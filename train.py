import logging
import json
import os
import argparse

from trainer.validation import run_intra_subject_validation, run_inter_subject_validation, run_subject_transfer_matrix

def main():
    """
    Main function to configure logging and test the dataset loading.
    """
    # --- 1. Configure Logging ---
    # Change to logging.INFO to see only the high-level status messages.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # --- 2. Parse Command-Line Arguments ---
    parser = argparse.ArgumentParser(description="SSVEP Model Training Script")
    parser.add_argument('--config', type=str, default='./config.json', help='Path to the main configuration file.')
    parser.add_argument('--mode', type=str, default='intra-subject', choices=['intra-subject', 'inter-subject', 'transfer-matrix'], help='Validation strategy.')
    parser.add_argument('--visualization', action='store_true', help='If set, visualizes the dataset before training and plots results after training.')
    args = parser.parse_args()
    
    # --- Load All Configurations ---
    with open(args.config, 'r') as f:
        config = json.load(f)

    # --- Get Data Parameters from Config ---
    training_params = config['training_params']
    dataset_path = training_params['dataset_path']
    model_name = training_params['model_name']
    subject_list = training_params['subjects']
    dataset_metadata_path = os.path.join(dataset_path, 'metadata.json')

    # --- Create Results Directory ---
    results_dir = os.path.join(dataset_path, f"{model_name}_results")
    os.makedirs(results_dir, exist_ok=True)

    logger.info("--- Training Configuration ---")
    logger.info(f"Model: {model_name}")
    logger.info(f"Validation Mode: {args.mode}")
    logger.info(f"Subjects: {subject_list}")
    logger.info(f"Dataset Path: {dataset_path}")
    logger.info(f"Visualization: {args.visualization}")
    logger.info(f"Results will be saved to: {results_dir}")
    logger.info("------------------------------")

    if args.mode == 'intra-subject':
        run_intra_subject_validation(args, config, subject_list, dataset_metadata_path, logger, results_dir)
    elif args.mode == 'inter-subject':
        run_inter_subject_validation(args, config, subject_list, dataset_metadata_path, logger, results_dir)
    elif args.mode == 'transfer-matrix':
        run_subject_transfer_matrix(args, config, subject_list, dataset_metadata_path, logger, results_dir)

if __name__ == '__main__':
    main()
