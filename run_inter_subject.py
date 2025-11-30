import logging
import json
import os
import argparse

from IO.dataset import build_dataset_from_config
from trainer.engine import run_training
from utils.plot_results import plot_subject_accuracies, plot_confusion_matrix
from utils.visualization import visualize_eeg_data, plot_grad_cam_visuals

def main():
    """
    Main function to run inter-subject (leave-one-out) cross-validation.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Inter-Subject SSVEP Training Script")
    parser.add_argument('--config', type=str, default='./config/config_inter_subject.json', help='Path to the configuration file for inter-subject validation.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        main_config = json.load(f)

    training_params = main_config['training_params']
    dataset_path = training_params['dataset_path']
    model_name = training_params['model_name']
    dataset_metadata_path = os.path.join(dataset_path, 'metadata.json')

    with open(dataset_metadata_path, 'r') as f:
        metadata_config = json.load(f)

    # --- Create a unified config object ---
    config = {**main_config, **metadata_config}

    # --- Determine Subject List ---
    if training_params.get('subjects') == ["all"]:
        num_subjects = config['data_metadata']['Number_of_Subjects']
        subject_list = list(range(1, num_subjects + 1))
    else:
        subject_list = training_params['subjects']

    results_dir = os.path.join(dataset_path, f"{model_name}_results_inter_subject")
    os.makedirs(results_dir, exist_ok=True)

    logger.info("--- Inter-Subject Training Configuration ---")
    logger.info(f"Model: {model_name}, Subjects: {subject_list}")
    logger.info(f"Results will be saved to: {results_dir}")
    logger.info("-------------------------------------------")

    if len(subject_list) < 2:
        logger.error("Inter-subject validation requires at least 2 subjects.")
        return

    subject_accuracies = {}

    for i, val_subject_id in enumerate(subject_list):
        logger.info(f"========== FOLD {i + 1}/{len(subject_list)} ==========")
        
        train_subject_ids = [s for s in subject_list if s != val_subject_id]
        logger.info(f"Training on subjects: {train_subject_ids}, Validating on subject: {val_subject_id}")

        train_dataset = build_dataset_from_config(
            config=config, subject_list=train_subject_ids
        )
        val_dataset = build_dataset_from_config(
            config=config, subject_list=[val_subject_id]
        )
        
        visualize_eeg_data(val_dataset, config, save_dir=results_dir)

        fit_results, collate_fn = run_training(train_dataset, val_dataset, config, logger)
        _, best_score, best_preds, best_labels, trained_model = fit_results
        subject_accuracies[val_subject_id] = best_score

        if best_preds is not None:
            plot_confusion_matrix(best_labels, best_preds, save_dir=results_dir, subject_id=val_subject_id, title_prefix="Inter-Subject")
            
            # List of models for which Grad-CAM should be generated
            if model_name in ['SSVEP_CASViT', 'SSVEPformer']:
                logger.info(f"Generating Grad-CAM visualizations for validation on S{val_subject_id}...")
                plot_grad_cam_visuals(
                    model=trained_model, 
                    dataset=val_dataset, 
                    config=config,
                    preprocess_fn=collate_fn, 
                    save_dir=results_dir, 
                    subject_id=val_subject_id
                )


    logger.info("Inter-subject validation complete. Plotting results...")
    plot_subject_accuracies(
        accuracies=subject_accuracies,
        title='Inter-Subject (LOSO) Validation Accuracy',
        save_dir=results_dir
    )

    logger.info(f"Final per-subject accuracies: {subject_accuracies}")

if __name__ == '__main__':
    main()