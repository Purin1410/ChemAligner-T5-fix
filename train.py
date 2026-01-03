from argparse import ArgumentParser, Namespace
import math

import lightning as pl
import optuna
from sconf import Config
from transformers import AutoTokenizer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import seed_everything

from src.dataset_module import get_dataloaders
from src.lighning_module_base import T5Model as T5ModelBase
from src.lightning_module import T5Model as T5ModelContrastive


METHOD_MAP = {
    "base": T5ModelBase,
    "chemaligner": T5ModelContrastive,
}

DATASET_MAP = {
    "lpm-24": "duongttr/LPM-24-extend",
    "lpm-24-extra": "Neeze/LPM-24-extra-extend",
    "lpm-24-smoke": "Neeze/LPM-24-smoke-extend",
    "chebi-20": "duongttr/chebi-20-new",
}


def inject_config_to_args(args: Namespace, config: Config, tokenizer, train_dataloader) -> Namespace:
    """
    Inject config values into args.* so downstream code can keep using args.xxx
    without breaking your current code structure.
    """

    # Model backbone
    args.t5 = Namespace()
    args.t5.pretrained_model_name_or_path = config.t5.pretrained_model_name_or_path

    # Trainer related
    args.lr = float(config.trainer_init.lr)
    args.warmup_ratio = float(config.trainer_init.warmup_ratio)
    args.max_epochs = int(config.trainer_init.max_epochs)
    args.grad_accum = int(config.trainer_init.grad_accum)

    # Steps per epoch for scheduler
    steps_per_epoch = int(math.ceil(len(train_dataloader) / max(1, args.grad_accum)))
    args.train_data_len = steps_per_epoch

    # Token ids used in LightningModule forward
    args.pad_token_id = int(tokenizer.pad_token_id)

    # Method weights
    args.seq2seq_loss_weight = float(config.method_init.seq2seq_loss_weight)
    args.contrastive_loss_weight = float(config.method_init.contrastive_loss_weight)

    # Eval settings (used by LightningModule)
    args.eval_batch_size = int(config.dataset_init.eval_batch_size)
    args.eval_max_length = int(config.eval_init.max_length)
    args.eval_num_beams = int(config.eval_init.num_beams)
    args.eval_use_amp = bool(config.eval_init.use_amp)
    args.eval_max_samples = int(config.eval_init.max_samples)
    args.eval_num_proc = int(config.dataset_init.num_workers)
    args.eval_chunk_size = int(config.eval_init.chunk_size)
    args.eval_compute_fcd = bool(config.eval_init.compute_fcd)
    args.eval_run_text2mol_metrics = bool(config.eval_init.run_text2mol_metrics)

    # Optimizer settings (optional but recommended)
    args.optimizer_name = str(config.trainer_init.optimizer.name)
    args.optimizer_weight_decay = float(config.trainer_init.optimizer.weight_decay)
    args.optimizer_momentum = float(config.trainer_init.optimizer.momentum)

    return args


def train_once(args, config, trial=None):
    # ============================== Init ==============================
    seed_everything(int(config.trainer_init.seed_everything))

    tokenizer = AutoTokenizer.from_pretrained(config.t5.pretrained_model_name_or_path)

    # ============================== Dataloaders ==============================
    if config.dataset_init.dataset_name not in DATASET_MAP:
        raise ValueError(f"Invalid dataset_name. Choose in: {', '.join(DATASET_MAP.keys())}")

    args.dataset_name_or_path = DATASET_MAP[config.dataset_init.dataset_name]

    train_dataloader = get_dataloaders(
        args,
        tokenizer,
        batch_size=int(config.dataset_init.train_batch_size),
        num_workers=int(config.dataset_init.num_workers),
        split="train",
        task=str(config.dataset_init.task),
    )

    val_dataloader = get_dataloaders(
        args,
        tokenizer,
        batch_size=int(config.dataset_init.eval_batch_size),
        num_workers=int(config.dataset_init.num_workers),
        split="validation",
        task=str(config.dataset_init.task),
    )

    # ============================== Inject config to args ==============================
    args = inject_config_to_args(args, config, tokenizer, train_dataloader)

    # ============================== Build model ==============================
    T5Model = METHOD_MAP[str(config.method_init.method)]
    model = T5Model(args)

    # Pass HF tokenizer object for decoding in validation metrics
    model.tokenizer = tokenizer

    # ============================== Callbacks & Logger ==============================
    on_best_eval_loss_callback = ModelCheckpoint(
        dirpath=str(config.trainer_init.output_folder),
        filename=str(config.trainer_init.filename),
        save_top_k=int(config.trainer_init.save_top_k),
        verbose=bool(config.trainer_init.verbose),
        monitor=str(config.trainer_init.monitor),
        mode=str(config.trainer_init.mode),
    )

    wandb_logger = WandbLogger(
        log_model=False,
        project=str(config.trainer_init.wandb.project),
        name=str(config.trainer_init.wandb.name),
        config=dict(config),
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [on_best_eval_loss_callback, lr_monitor]

    # ============================== Trainer ==============================
    trainer = pl.Trainer(
        callbacks=callbacks,
        max_epochs=int(config.trainer_init.max_epochs),
        accelerator="cuda" if bool(config.trainer_init.cuda) else "cpu",
        strategy=str(config.trainer_init.strategy) if int(config.trainer_init.num_devices) > 1 else "auto",
        devices=int(config.trainer_init.num_devices),
        precision=str(config.trainer_init.precision),
        gradient_clip_val=float(config.trainer_init.gradient_clip_val),
        gradient_clip_algorithm=str(config.trainer_init.gradient_clip_algorithm),
        logger=[wandb_logger],
        accumulate_grad_batches=int(config.trainer_init.grad_accum),
        deterministic=bool(config.trainer_init.deterministic),
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    if trial is not None:
        return float(trainer.callback_metrics["eval_loss"].item())


def train_with_optuna(config, args):
    def objective(trial):
        trial_args = Namespace(**vars(args))

        # Use lr_sweep from config.yaml
        trial_args.lr = trial.suggest_categorical(
            "lr", list(config.trainer_init.optuna.lr_sweep)
        )

        print(f"[Trial {trial.number}] lr = {trial_args.lr}")
        return train_once(trial_args, config, trial=trial)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=int(config.trainer_init.optuna.n_trials))

    best_trial = study.best_trial
    print("Best trial:")
    print(f"  eval_loss = {best_trial.value}")
    print("  params:")
    for k, v in best_trial.params.items():
        print(f"    {k}: {v}")


def main(args, config):
    if bool(config.trainer_init.optuna.activate):
        train_with_optuna(config, args)
    else:
        train_once(args, config)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_config", type=str, default="src/configs/config_lpm24_train.yaml")
    args = parser.parse_args()

    config = Config(args.model_config)
    main(args, config)
