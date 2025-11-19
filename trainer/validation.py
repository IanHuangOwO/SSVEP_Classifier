import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

from IO.dataset import build_dataset_from_config
from trainer.trainer import Trainer
from trainer.loss import LossCalculator
from trainer.metrics import MetricsCalculator
from utils.plot_results import plot_kfold_results, plot_subject_accuracies, plot_accuracy_matrix, plot_confusion_matrix
from utils.visualization import visualize_eeg_data, plot_attention_maps, plot_2d_attention_heatmap

def _build_model(model_name: str, main_config_path: str, dataset_metadata_path: str, logger):
    """
    Factory function to build a model based on its name from the config.
    """
    logger.info(f"Attempting to build model: {model_name}")
    if model_name == 'SSVEPformer':
        from models.SSVEPformer import build_model_from_config, preprocess_collate_fn
        model = build_model_from_config(main_config_path)
        collate_fn = preprocess_collate_fn(dataset_metadata_path)
        return model, collate_fn
    elif model_name == 'SSVEP_CASViT':
        from models.SSVEP_CASViT import build_model_from_config, preprocess_collate_fn
        model = build_model_from_config(main_config_path)
        collate_fn = preprocess_collate_fn(dataset_metadata_path)
        return model, collate_fn
    else:
        error_msg = f"Unknown model name '{model_name}' specified in config."
        logger.error(error_msg)
        raise ValueError(error_msg)

def _run_training_fold(train_dataset, val_dataset, config, main_config_path: str, dataset_metadata_path: str, logger):
    """
    Helper function to run a single training and validation fold.
    Initializes the model, optimizer, trainer and runs the training.
    """
    train_cfg = config['training_params']
    loss_cfg = config['loss_params']
    metrics_cfg = config['metrics_params']
    model_name = train_cfg['model_name']

    # --- Re-initialize model and optimizer for each fold ---
    logger.info("Building fresh model for this fold...")
    model_build_result = _build_model(model_name, main_config_path, dataset_metadata_path, logger)
    if isinstance(model_build_result, tuple):
        model, collate_fn = model_build_result
    else:
        model, collate_fn = model_build_result, None # Default collate if model doesn't provide one

    # --- Build DataLoaders internally ---
    train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg['batch_size'], shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['learning_rate'])

    # --- Build Calculators and Trainer ---
    device = train_cfg['device'] if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    loss_calculator = LossCalculator(config=loss_cfg, device=device)
    metrics_calculator = MetricsCalculator(config=metrics_cfg)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        loss_calculator=loss_calculator,
        metrics_calculator=metrics_calculator
    )

    fit_results = trainer.fit(train_loader, val_loader, num_epochs=train_cfg['epochs'])

    return fit_results, collate_fn


def run_inter_subject_validation(args, config, subject_list, dataset_metadata_path, logger, results_dir):
    """
    Performs leave-one-subject-out cross-validation.
    """
    if len(subject_list) < 2:
        logger.error("Inter-subject validation requires at least 2 subjects.")
        return
    
    subject_accuracies = {}

    logger.info(f"Starting Leave-One-Subject-Out Cross-Validation for subjects: {subject_list}...")

    for i, val_subject_id in enumerate(subject_list):
        logger.info(f"========== FOLD {i + 1}/{len(subject_list)} ==========")
        
        # --- Create datasets for this fold ---
        train_subject_ids = [s for s in subject_list if s != val_subject_id]
        logger.info(f"Training on subjects: {train_subject_ids}, Validating on subject: {val_subject_id}")

        train_dataset = build_dataset_from_config(
            main_config_path=args.config, dataset_metadata_path=dataset_metadata_path, subject_list=train_subject_ids
        )
        val_dataset = build_dataset_from_config(
            main_config_path=args.config, dataset_metadata_path=dataset_metadata_path, subject_list=[val_subject_id]
        )
        
        if args.visualization:
            logger.info("--- Visualization Step ---")
            visualize_eeg_data(
                dataset=val_dataset,
                dataset_metadata_path=dataset_metadata_path,
                save_dir=results_dir
            )

        # Run the training fold with the datasets
        fit_results, collate_fn = _run_training_fold(train_dataset, val_dataset, config, args.config, dataset_metadata_path, logger)
        _, best_score, best_preds, best_labels, trained_model = fit_results
        subject_accuracies[val_subject_id] = best_score

        # --- Visualization for this fold ---
        if args.visualization and best_preds is not None:
            # Plot confusion matrix
            plot_confusion_matrix(best_labels, best_preds, save_dir=results_dir, subject_id=val_subject_id, title_prefix="Inter-Subject")
            
            # Plot attention maps
            logger.info(f"Plotting attention maps for validation on S{val_subject_id}...")
            plot_attention_maps(model=trained_model,
                                dataset=val_dataset,
                                dataset_metadata_path=dataset_metadata_path,
                                save_dir=results_dir,
                                subject_id=val_subject_id,
                                preprocess_fn=collate_fn)
            
            # Also plot a detailed 2D heatmap for the first trial as an example
            logger.info("Plotting detailed 2D attention map for the first validation trial...")
            plot_2d_attention_heatmap(model=trained_model,
                                      dataset=val_dataset,
                                      dataset_metadata_path=dataset_metadata_path,
                                      preprocess_fn=collate_fn,                                      
                                      save_dir=results_dir,
                                      subject_id=val_subject_id,
                                      train_subject_id=train_subject_ids[0])

    # --- Aggregate and Plot Results ---
    if args.visualization:
        logger.info("Inter-subject validation complete. Plotting results...")
        plot_subject_accuracies(
            accuracies=subject_accuracies,
            title='Inter-Subject (LOSO) Validation Accuracy',
            save_dir=results_dir
        )
    else:
        logger.info("Inter-subject validation complete. Skipping result plotting as --visualization flag was not set.")

    logger.info(f"Final per-subject accuracies: {subject_accuracies}")

def run_intra_subject_validation(args, config, subject_list, dataset_metadata_path, logger, results_dir):
    """
    Performs K-Fold cross-validation on the data from the specified subjects.
    """

    subject_accuracies = {}

    for subject_id in subject_list:
        logger.info(f"--- Starting Intra-Subject Validation for Subject {subject_id} ---")

        # --- Build Dataset for the current subject ---
        full_subject_dataset = build_dataset_from_config(
            main_config_path=args.config, dataset_metadata_path=dataset_metadata_path,
            subject_list=[subject_id]
        )

        # --- Optional: Visualize Data for the current subject ---
        if args.visualization:
            logger.info("--- Visualization Step ---")
            visualize_eeg_data(
                dataset=full_subject_dataset,
                dataset_metadata_path=dataset_metadata_path,
                save_dir=results_dir
            )
        
        # --- Set up K-Fold Cross-Validation ---
        train_cfg = config['training_params']
        k_folds = train_cfg['k_folds']
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        all_fold_histories = []
        all_fold_best_scores = []

        logger.info(f"Starting {k_folds}-Fold Cross-Validation...")

        for fold, (train_indices, val_indices) in enumerate(kf.split(full_subject_dataset)):
            logger.info(f"========== FOLD {fold + 1}/{k_folds} ==========")

            train_subset = Subset(full_subject_dataset, train_indices)
            val_subset = Subset(full_subject_dataset, val_indices)
            
            fit_results, _ = _run_training_fold(train_subset, val_subset, config, args.config, dataset_metadata_path, logger)
            full_history, best_score, _, __, _ = fit_results
            all_fold_histories.append(full_history)
            all_fold_best_scores.append(best_score)

        # --- Report, Calculate, and Store Results for the Subject ---
        logger.info("\n--- K-Fold Final Report for Subject %s ---", subject_id)
        for i, score in enumerate(all_fold_best_scores):
            logger.info(f"Fold {i+1}: Best Validation Accuracy = {score:.4f}")
        
        mean_acc = np.mean(all_fold_best_scores)
        std_acc = np.std(all_fold_best_scores)
        logger.info("\n--- Summary for Subject %s ---", subject_id)
        logger.info(f"Average Validation Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        
        # Store the average accuracy for the final summary plot
        avg_accuracy = mean_acc
        subject_accuracies[subject_id] = avg_accuracy

        # --- Plot K-Fold results for the current subject ---
        if args.visualization:
            plot_kfold_results(
                histories=all_fold_histories,
                save_dir=results_dir,
                subject_id=subject_id
            )
            
            # --- Plot attention map for the best overall model for this subject ---
            logger.info("Plotting attention maps using the best model from all folds...")
            # Re-run training on the full dataset to get a final model for visualization
            # Use the full dataset for both "training" and "validation" loaders to find the best model on all data
            fit_results, collate_fn = _run_training_fold(full_subject_dataset, full_subject_dataset, config, args.config, dataset_metadata_path, logger)
            _, _, _, _, trained_model = fit_results
            
            # Use the returned model directly for plotting
            plot_attention_maps(model=trained_model,
                                dataset=full_subject_dataset,
                                dataset_metadata_path=dataset_metadata_path,
                                save_dir=results_dir,
                                subject_id=subject_id,
                                preprocess_fn=collate_fn)

    # --- Aggregate and Plot Results ---
    if args.visualization:
        logger.info("Intra-subject validation complete. Plotting results...")
        plot_subject_accuracies(
            accuracies=subject_accuracies,
            title='Intra-Subject Mean Validation Accuracy',
            save_dir=results_dir
        )

    logger.info(f"Final per-subject average accuracies: {subject_accuracies}")

def run_subject_transfer_matrix(args, config, subject_list, dataset_metadata_path, logger, results_dir):
    """
    Performs subject-to-subject transfer learning validation.
    Trains a model on each subject and validates it against every other subject,
    generating an accuracy matrix.
    """
    if len(subject_list) < 1:
        logger.error("Subject transfer matrix requires at least 1 subject.")
        return

    # Initialize a nested dictionary to store the accuracy matrix
    accuracy_matrix = {train_id: {} for train_id in subject_list}
    logger.info(f"Starting Subject-to-Subject Transfer Matrix validation for subjects: {subject_list}...")

    for train_subject_id in subject_list:
        logger.info(f"--- TRAINING on Subject {train_subject_id} ---")

        # --- Create training dataset for the current training subject ---
        train_dataset = build_dataset_from_config(
            main_config_path=args.config, dataset_metadata_path=dataset_metadata_path, subject_list=[train_subject_id]
        )

        for val_subject_id in subject_list:
            logger.info(f"--- VALIDATING on Subject {val_subject_id} ---")

            # --- Create validation dataset for the current validation subject ---
            val_dataset = build_dataset_from_config(
                main_config_path=args.config, dataset_metadata_path=dataset_metadata_path, subject_list=[val_subject_id]
            )

            # --- Run a single training and validation pass ---
            fit_results, _ = _run_training_fold(train_dataset, val_dataset, config, args.config, dataset_metadata_path, logger)
            _, best_score, _, _, _ = fit_results
            accuracy_matrix[train_subject_id][val_subject_id] = best_score
            logger.info(f"Accuracy (Train: S{train_subject_id}, Val: S{val_subject_id}): {best_score:.4f}")

    if args.visualization:
        plot_accuracy_matrix(accuracy_matrix, subject_list, save_dir=results_dir)