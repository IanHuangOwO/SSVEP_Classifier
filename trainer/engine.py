import torch
from torch.utils.data import DataLoader

from trainer.trainer import Trainer
from trainer.loss import LossCalculator
from trainer.metrics import MetricsCalculator

def _build_model(model_name: str, config: dict, logger):
    """
    Factory function to build a model based on its name from the config.
    """
    logger.info(f"Attempting to build model: {model_name}")
    if model_name == 'SSVEPformer':
        from models.SSVEPformer import build_model_from_config, preprocess_collate_fn
        model = build_model_from_config(config)
        collate_fn = preprocess_collate_fn(config)
        return model, collate_fn
    elif model_name == 'SSVEP_CASViT':
        from models.SSVEP_CASViT import build_model_from_config, preprocess_collate_fn
        model = build_model_from_config(config)
        collate_fn = preprocess_collate_fn(config)
        return model, collate_fn
    else:
        error_msg = f"Unknown model name '{model_name}' specified in config."
        logger.error(error_msg)
        raise ValueError(error_msg)

def run_training(train_dataset, val_dataset, config: dict, logger):
    """
    Helper function to run a single training and validation pass.
    Initializes the model, data loaders, optimizer, and trainer, then runs the training.
    """
    train_cfg = config['training_params']
    loss_cfg = config['loss_params']
    metrics_cfg = config['metrics_params']
    model_name = train_cfg['model_name']

    # --- Re-initialize model and optimizer for each fold ---
    logger.info("Building fresh model for this fold...")
    model_build_result = _build_model(model_name, config, logger)
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