import logging
import json
import os
import argparse
import numpy as np
from torch.utils.data import Subset
from sklearn.model_selection import KFold

from IO.dataset import build_dataset_from_config
from trainer.engine import run_training
from utils.plot_results import plot_kfold_results, plot_subject_accuracies, plot_confusion_matrix
from utils.visualization import visualize_eeg_data, plot_grad_cam_visuals

def main():
    """
    Main function to run intra-subject k-fold cross-validation.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Intra-Subject SSVEP Training Script")
    parser.add_argument('--config', type=str, default='./config/config_intra_subject.json', help='Path to the configuration file for intra-subject validation.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        main_config = json.load(f)

    training_params = main_config['training_params']
    dataset_path = training_params['dataset_path']
    model_name = training_params['model_name']
    subject_list = training_params['subjects']
    dataset_metadata_path = os.path.join(dataset_path, 'metadata.json')

    with open(dataset_metadata_path, 'r') as f:
        metadata_config = json.load(f)

    # --- Create a unified config object ---
    config = {**main_config, **metadata_config}

    results_dir = os.path.join(dataset_path, f"{model_name}_results_intra_subject")
    os.makedirs(results_dir, exist_ok=True)

    logger.info("--- Intra-Subject Training Configuration ---")
    logger.info(f"Model: {model_name}, Subjects: {subject_list}")
    logger.info(f"Results will be saved to: {results_dir}")
    logger.info("-------------------------------------------")

    subject_accuracies = {}

    for subject_id in subject_list:
        logger.info(f"--- Starting Validation for Subject {subject_id} ---")

        full_subject_dataset = build_dataset_from_config(
            config=config, subject_list=[subject_id]
        )

        visualize_eeg_data(full_subject_dataset, config, save_dir=results_dir)

        k_folds = training_params['k_folds']
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        all_fold_histories = []
        all_fold_best_scores = []

        logger.info(f"Starting {k_folds}-Fold Cross-Validation...")

        for fold, (train_indices, val_indices) in enumerate(kf.split(full_subject_dataset)):
            logger.info(f"========== FOLD {fold + 1}/{k_folds} ==========")
            train_subset = Subset(full_subject_dataset, train_indices)
            val_subset = Subset(full_subject_dataset, val_indices)
            
            fit_results, _ = run_training(train_subset, val_subset, config, logger)
            full_history, best_score, _, __, _ = fit_results
            all_fold_histories.append(full_history)
            all_fold_best_scores.append(best_score)

        mean_acc = np.mean(all_fold_best_scores)
        std_acc = np.std(all_fold_best_scores)
        logger.info(f"\n--- Summary for Subject {subject_id} ---")
        logger.info(f"Average Validation Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        subject_accuracies[subject_id] = mean_acc

        plot_kfold_results(histories=all_fold_histories, save_dir=results_dir, subject_id=subject_id)
        
        logger.info("Plotting Grad-CAM maps using a model trained on all data for this subject...")
        fit_results, collate_fn = run_training(full_subject_dataset, full_subject_dataset, config, logger)
        _, _, best_preds, best_labels, trained_model = fit_results

        if best_preds is not None:
            plot_confusion_matrix(best_labels, best_preds, save_dir=results_dir, subject_id=subject_id, title_prefix="Intra-Subject (Full Data)")

        if model_name == 'SSVEP_CASViT':
            logger.info(f"Generating Grad-CAM visualizations for subject {subject_id}...")
            plot_grad_cam_visuals(
                model=trained_model, dataset=full_subject_dataset, config=config,
                preprocess_fn=collate_fn, save_dir=results_dir, subject_id=subject_id
            )

    logger.info("Intra-subject validation complete. Plotting final summary...")
    plot_subject_accuracies(
        accuracies=subject_accuracies,
        title='Intra-Subject Mean Validation Accuracy',
        save_dir=results_dir
    )

    logger.info(f"Final per-subject average accuracies: {subject_accuracies}")

if __name__ == '__main__':
    main()