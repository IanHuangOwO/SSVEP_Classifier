import logging
import json
import os
import argparse

from IO.dataset import build_dataset_from_config
from trainer.engine import run_training
from utils.plot_results import plot_accuracy_matrix

def main():
    """
    Main function to run subject-to-subject transfer matrix validation.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Subject Transfer Matrix SSVEP Training Script")
    parser.add_argument('--config', type=str, default='./config/config_transfer_matrix.json', help='Path to the configuration file for transfer matrix validation.')
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

    config = {**main_config, **metadata_config}

    results_dir = os.path.join(dataset_path, f"{model_name}_results_transfer_matrix")
    os.makedirs(results_dir, exist_ok=True)

    logger.info("--- Subject Transfer Matrix Configuration ---")
    logger.info(f"Model: {model_name}, Subjects: {subject_list}")
    logger.info(f"Results will be saved to: {results_dir}")
    logger.info("-------------------------------------------")

    if len(subject_list) < 1:
        logger.error("Subject transfer matrix requires at least 1 subject.")
        return

    accuracy_matrix = {train_id: {} for train_id in subject_list}

    for train_subject_id in subject_list:
        logger.info(f"--- TRAINING on Subject {train_subject_id} ---")

        train_dataset = build_dataset_from_config(
            config=config, subject_list=[train_subject_id]
        )

        for val_subject_id in subject_list:
            logger.info(f"--- VALIDATING on Subject {val_subject_id} ---")

            val_dataset = build_dataset_from_config(
                config=config, subject_list=[val_subject_id]
            )

            fit_results, _ = run_training(train_dataset, val_dataset, config, logger)
            _, best_score, _, _, _ = fit_results
            accuracy_matrix[train_subject_id][val_subject_id] = best_score
            logger.info(f"Accuracy (Train: S{train_subject_id}, Val: S{val_subject_id}): {best_score:.4f}")

    plot_accuracy_matrix(accuracy_matrix, subject_list, save_dir=results_dir)

if __name__ == '__main__':
    main()