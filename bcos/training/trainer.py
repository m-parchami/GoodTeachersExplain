import os
from pathlib import Path
from typing import Dict

import pytorch_lightning as pl
import pytorch_lightning.callbacks as pl_callbacks
import pytorch_lightning.loggers as pl_loggers
import pytorch_lightning.plugins.environments as pl_env_plugins
import torch
import torchmetrics
from pytorch_lightning.utilities import rank_zero_info
from contextlib import nullcontext
try:
    import rich  # noqa: F401

    HAS_RICH = True
except ImportError:
    HAS_RICH = False

import bcos.training.callbacks as custom_callbacks
from bcos.experiments.utils import CHECKPOINT_LAST_FILENAME, Experiment, sanitize_config, ExpMethod
from bcos.training.agc import adaptive_clip_grad_

# Only for Debug:
from matplotlib import pyplot as plt
from bcos.common import param_to_buffer
from bcos.modules.bcoslinear import BcosLinear
import numpy as np

from bcos.experiments.utils.experiment_utils.loading_utils import device_safe_load_state_dict_from_path
from torch.nn import Sequential
from os.path import join as ospj

class ClassificationLitModel(pl.LightningModule):
    def __init__(self, dataset, base_network, experiment_name, get_target_name=None):
        super().__init__()
        self.experiment = Experiment(dataset, base_network, experiment_name)
        self.get_target_name = get_target_name # Just a method of datamodule in case given!
        config = self.experiment.config

        num_classes = config["data"]["num_classes"]

        rank_zero_info("Initializing the model from scratch!")
        model = self.experiment.get_model()

        self.save_hyperparameters()  # passed arguments
        self.save_hyperparameters(sanitize_config(config))  # the config as well

        self.experiment_name = experiment_name
        self.config = config
        self.model = model

        self.is_bcos = self.config["model"].get("is_bcos", False)
        self.full_eval_every = self.config.get('full_eval_every_n_epochs', 1)

        self.criterion = self.config["criterion"] or \
            self.config['criterion_module'](self.config['criterion_config'])
        
        if self.config.get('test_criterion_same_as_train', False):
            self.test_criterion = self.criterion
        else:
            self.test_criterion = self.config["test_criterion"] or \
                self.config['criterion_module'](self.config['test_criterion_config'])

        self.ffkd_training = self.config.get('ffkd_training', False)
        self.livekd_training = self.config.get('livekd_training', False)


        metric = config.get('metric', None)
        metric_kwargs = config.get('metric_kwargs', None)
        if metric is not None:
            self.train_acc1 = metric['train_acc1'](**metric_kwargs['train_acc1'])
            self.train_acc5 = metric['train_acc5'](**metric_kwargs['train_acc5'])
            self.eval_acc1 = metric['eval_acc1'](**metric_kwargs['eval_acc1'])
            self.eval_acc5 = metric['eval_acc5'](**metric_kwargs['eval_acc5'])
        else:
            self.train_acc1 = torchmetrics.Accuracy(
                task="multiclass", top_k=1, num_classes=num_classes, compute_on_cpu=True
            )
            self.train_acc5 = torchmetrics.Accuracy(
                task="multiclass", top_k=5, num_classes=num_classes, compute_on_cpu=True
            )
            self.eval_acc1 = torchmetrics.Accuracy(
                task="multiclass", top_k=1, num_classes=num_classes, compute_on_cpu=True
            )
            self.eval_acc5 = torchmetrics.Accuracy(
                task="multiclass", top_k=5, num_classes=num_classes, compute_on_cpu=True
            )

        self.no_train_acc = config.get('no_train_acc', False)
        if self.no_train_acc:
            self.train_acc1 = None
            self.train_acc5 = None

        if self.livekd_training or self.ffkd_training:
            self.distillation_trainstep = self.config.get('distillation_trainstep')
            self.distillation_evalstep = self.config.get('distillation_evalstep')
            self.distillation_init = self.config.get('distillation_init')
            
            # Also add agreement!
            self.train_agree_acc1 = torchmetrics.Accuracy(
                task="multiclass", top_k=1, num_classes=num_classes, compute_on_cpu=True
            )
            self.eval_agree_acc1 = torchmetrics.Accuracy(
                task="multiclass", top_k=1, num_classes=num_classes, compute_on_cpu=True
            )
        else:
            self.train_agree_acc1 = None
            self.eval_agree_acc1 = None

        self.teacher_model = None
        self.teacher_ctx, self.maybe_no_grad_teacher = nullcontext, nullcontext # Might be set to teacher.explanation_mode

        if self.ffkd_training or self.livekd_training:
            if self.config.get('data').get('mixup_alpha', 0.0) != 0.0:
                assert not self.ffkd_training, 'This does not work with MixUp!'
            
            if self.livekd_training:
                rank_zero_info("~~~~~~~ Using LIVE Teacher ~~~~~~~")
                criterion_config = self.config["criterion_config"]
                teacher_experiment = Experiment(
                    criterion_config['live_teacher_dataset'],
                    criterion_config['live_teacher_base_network'],
                    criterion_config['live_teacher_experiment_name']
                )

                teacher_ckpt_name = criterion_config.get('live_teacher_ckpt_name', None)
                if teacher_ckpt_name is None:
                    rank_zero_info("loading best checkpoint for teacher!")
                    self.teacher_model = teacher_experiment.load_trained_model(reload='best')
                else:
                    ckpt_path = ospj(teacher_experiment.save_dir, teacher_ckpt_name)
                    rank_zero_info(f"loading given checkpoint from {ckpt_path}")
                    if ckpt_path.endswith(('.pt', 'pth')):
                        model_state_dict = device_safe_load_state_dict_from_path(ckpt_path) 
                        if 'model' in model_state_dict:
                            model_state_dict = model_state_dict['model']
                        
                        try: # Everything matches
                            self.teacher_model = teacher_experiment.get_model()
                            self.teacher_model.load_state_dict(model_state_dict, strict=True)
                        except RuntimeError:
                            try: # In case it's wrapped in a Sequential
                                seq_model = Sequential(teacher_experiment.get_model())
                                seq_model.load_state_dict(model_state_dict, strict=True)
                                self.teacher_model = seq_model[0]
                            except RuntimeError: # Assume that it has a certain transform to be applied
                                tm = teacher_experiment.get_model()
                                tm.load_state_dict(tm._transform_state_dict(model_state_dict), strict=True)
                                self.teacher_model = tm
                    else:
                        raise NotImplementedError

                self.teacher_model.eval()
                param_to_buffer(self.teacher_model)
                self.teacher_ctx = getattr(self.teacher_model, 'explanation_mode', nullcontext)

        self.lr_warmup_epochs = self.config["lr_scheduler"].warmup_epochs

        self.use_agc = self.config.get("use_agc", False)
        if self.use_agc:
            rank_zero_info("Adaptive Gradient Clipping is enabled!")

        self.sync_ctx = nullcontext # Will be set up later
        assert self.config.get("ema", None) is None, 'Using EMA! Not implemented yet!'

        if self.livekd_training or self.ffkd_training:
            self.distillation_init(self) # Run the distillation_init of the Criterion.

    def setup(self, stage: str) -> None:
        if stage != "fit":
            return

    def configure_optimizers(self):
        if self.config.get('criterion_has_params', False):
            optimizer = self.config["optimizer"].create(
                    torch.nn.ModuleList([self.model, self.criterion])
            )
        else:
            optimizer = self.config["optimizer"].create(self.model)
            
        scheduler = self.config["lr_scheduler"].create(
            optimizer,
            # this is total as in "whole" training
            total_steps=self.trainer.estimated_stepping_batches,
        )
        return dict(optimizer=optimizer, lr_scheduler=scheduler)

    def forward(self, in_tensor):
        model_ret = self.model(in_tensor)
        if self.livekd_training:
            with self.maybe_no_grad_teacher():
                teacher_ret = self.teacher_model(in_tensor)
                return model_ret, teacher_ret
        return model_ret


    def training_step(self, batch, batch_idx):
        if not (self.ffkd_training or self.livekd_training):
            # Load and process things normally
            images, labels = batch
            outputs = self(images)
            loss = self.criterion(outputs, labels)
            all_losses_dict = dict(loss=loss.item(), gt_logit_loss=loss.item(), logit_loss=loss.item())
        else: 
            loss, outputs, labels, teacher_preds, all_losses_dict = self.distillation_trainstep(self, batch, batch_idx)

        all_losses_dict = {f"train_{key}": value for key, value in all_losses_dict.items()}

        with torch.no_grad():
            if labels.ndim == 2 and self.config.get('data').get('mixup_alpha', 0.0) != 0.0:
                # b/c of mixup/cutmix or sparse labels in general. See
                # https://github.com/pytorch/vision/blob/9851a69f6d294f5d672d973/references/classification/utils.py#L179
                labels = labels.argmax(dim=1)
            if not self.no_train_acc: #e.g. in data-free we don't have train accuracy!
                
                self.train_acc1(outputs, labels)
                self.train_acc5(outputs, labels)

                self.log_dict(all_losses_dict)
                self.log("train_acc1", self.train_acc1,
                    on_step=True, on_epoch=True, prog_bar=True,
                    )
                self.log("train_acc5", self.train_acc5,
                    on_step=True, on_epoch=True, prog_bar=True,
                )
            if self.ffkd_training or self.livekd_training:
                self.train_agree_acc1(outputs, teacher_preds)
                self.log(
                    "train_agree_acc1", self.train_agree_acc1,
                    on_step=True, on_epoch=True, prog_bar=True,
                )

        return loss

    def eval_step(self, batch, _batch_idx, val_or_test):
        if not (self.ffkd_training or self.livekd_training):
            # Load and process things normally
            images, labels = batch
            outputs = self(images)
            loss = self.test_criterion(outputs, labels)
            all_losses_dict = {
                f"loss": loss.item(),
                f"logit_loss": loss.item(),
                f"gt_logit_loss": loss.item()}
        else:
            loss, outputs, labels, teacher_preds, all_losses_dict = self.distillation_evalstep(self, batch, _batch_idx)

        all_losses_dict = {f"{val_or_test}_{key}": value for key, value in all_losses_dict.items()}

        with torch.no_grad():
            if labels.ndim == 2 and self.config.get('data').get('mixup_alpha', 0.0) != 0.0:
                # b/c of mixup/cutmix or sparse labels in general. See
                # https://github.com/pytorch/vision/blob/9851a69f6d294f5d672d973/references/classification/utils.py#L179
                labels = labels.argmax(dim=1)
            self.eval_acc1(outputs, labels)
            self.eval_acc5(outputs, labels)

            self.log_dict(all_losses_dict)
            self.log(f"{val_or_test}_acc1", self.eval_acc1, on_epoch=True, prog_bar=True)
            self.log(f"{val_or_test}_acc5", self.eval_acc5, on_epoch=True, prog_bar=True)

            if self.ffkd_training or self.livekd_training:
                self.eval_agree_acc1(outputs, teacher_preds.detach())
                self.log(f"{val_or_test}_agree_acc1", self.eval_agree_acc1, on_epoch=True, prog_bar=True)
                

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "test")

    def configure_gradient_clipping(
        self,
        optimizer,
        optimizer_idx,
        gradient_clip_val=None,
        gradient_clip_algorithm=None,
    ) -> None:
        # Note: this is called even if gradient_clip_val etc. is None
        if not self.use_agc:
            self.clip_gradients(optimizer, gradient_clip_val, gradient_clip_algorithm)
        else:
            adaptive_clip_grad_(self.parameters())

    def log_grad_norm(self, grad_norm_dict: Dict[str, float]) -> None:
        # we only care about total grad norm
        norm_type = float(self.trainer.track_grad_norm)
        total_norm = grad_norm_dict[f"grad_{norm_type}_norm_total"]
        del grad_norm_dict
        self.log(
            "gradients/total_norm",
            total_norm,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        if self.trainer.gradient_clip_val is not None:
            clipped_total_norm = min(
                float(self.trainer.gradient_clip_val), float(total_norm)
            )
            self.log(
                "gradients/clipped_total_norm",
                clipped_total_norm,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )


def put_trainer_args_into_trainer_config(args, trainer_config):
    if args.distributed:
        # https://github.com/Lightning-AI/lightning/discussions/6761#discussioncomment-2614296
        trainer_config["strategy"] = "ddp_find_unused_parameters_false"

    if args.fast_dev_run:
        trainer_config["fast_dev_run"] = True

    if hasattr(args, "nodes"):  # on slurm
        trainer_config["num_nodes"] = args.nodes

    if args.track_grad_norm:
        trainer_config["track_grad_norm"] = 2.0

    if hasattr(args, "amp") and args.amp:
        trainer_config["precision"] = 16

    if args.debug:
        trainer_config["deterministic"] = True


def run_training(args):
    """
    Instantiates everything and runs the training.
    """
    base_directory = args.base_directory
    dataset = args.dataset
    base_network = args.base_network
    experiment_name = args.experiment_name
    save_dir = Path(base_directory, dataset, base_network, experiment_name)
    save_dir.mkdir(parents=True, exist_ok=True)

    # set up loggers early so that WB starts capturing output asap
    loggers = setup_loggers(args)

    # get config
    exp = Experiment(dataset, base_network, experiment_name)
    config = exp.config.copy()

    # get and set seed
    seed = exp.config.get("seed", 42)
    pl.seed_everything(seed, workers=True)

    # init model
    model = ClassificationLitModel(
        dataset,
        base_network,
        experiment_name,
        get_target_name=None #Might be set below after datamodule creation
    )
    rank_zero_info(f"Model: {repr(model.model)}")

    # jit the internal model if specified
    if args.jit:
        model.model = torch.jit.script(model.model)
        rank_zero_info("Jitted the model!")

    # init datamodule
    datamodule = model.experiment.get_datamodule(
        cache_dataset=getattr(args, "cache_dataset", None),
    )

    if hasattr(datamodule, 'get_target_name'):
        model.get_target_name = datamodule.get_target_name
    
    # callbacks
    callbacks = setup_callbacks(args, config)

    # init trainer
    trainer_config = config["trainer"]
    put_trainer_args_into_trainer_config(args, trainer_config)

    # plugin for slurm
    if "SLURM_JOB_ID" in os.environ:  # we're on slurm
        # let submitit handle requeuing
        trainer_config["plugins"] = [
            pl_env_plugins.SLURMEnvironment(auto_requeue=False)
        ]

    trainer = pl.Trainer(
        default_root_dir=save_dir,
        accelerator="auto",
        devices="auto",
        logger=loggers,
        callbacks=callbacks,
        # profiler='advanced',
        # limit_train_batches=0.1, # Can be set for debugging
        **trainer_config,
    )

    # decide whether to resume
    ckpt_path = None
    if args.resume:
        ckpt_path = save_dir / CHECKPOINT_LAST_FILENAME
        ckpt_path = ckpt_path if ckpt_path.exists() else None

    # start training
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


def setup_loggers(args):
    loggers = []
    save_dir = Path(
        args.base_directory, args.dataset, args.base_network, args.experiment_name
    )

    if args.wandb_logger:
        wandb_logger = pl_loggers.WandbLogger(
            name=args.wandb_name or args.experiment_name,
            save_dir=str(save_dir),
            project=args.wandb_project,
            id=args.wandb_id,
        )
        loggers.append(wandb_logger)

    if args.csv_logger:
        csv_logger = pl_loggers.CSVLogger(
            save_dir=str(save_dir / "csv_logs"),
            name="",
            flush_logs_every_n_steps=1000,
        )
        loggers.append(csv_logger)

    if args.tensorboard_logger:
        tensorboard_logger = pl_loggers.TensorBoardLogger(
            save_dir=Path(
                "tb_logs",
                args.base_directory,
                args.dataset,
                args.base_network,
                args.experiment_name,
            ),
            name=args.experiment_name,
        )
        loggers.append(tensorboard_logger)

    return loggers


def setup_callbacks(args, config):
    callbacks = []
    save_dir = Path(
        args.base_directory, args.dataset, args.base_network, args.experiment_name
    )

    # the most important one
    save_callback = pl_callbacks.ModelCheckpoint(
        dirpath=save_dir,
        monitor="val_acc1",
        mode="max",
        filename="{epoch}-{val_acc1:.4f}",
        save_last=True,
        save_top_k=3,
        verbose=True,
    )
    callbacks.append(save_callback)

    use_ema = config.get("ema", None) is not None
    if use_ema:
        raise NotImplementedError

    # lr monitor
    has_logger = args.wandb_logger or args.tensorboard_logger or args.csv_logger
    if has_logger:  # ow it's useless
        callbacks.append(pl_callbacks.LearningRateMonitor())
    slurm_or_submitit = hasattr(args, "nodes") or "SLURM_JOB_ID" in os.environ
    refresh_rate = args.refresh_rate or (20 if slurm_or_submitit else 5)
    if HAS_RICH and not slurm_or_submitit:
        callbacks.append(pl_callbacks.RichProgressBar(refresh_rate=refresh_rate))
    else:
        callbacks.append(pl_callbacks.TQDMProgressBar(refresh_rate=refresh_rate))

    # save metrics to checkpoint
    callbacks.append(custom_callbacks.MetricsTracker())

    # TODO Not sure which of the following callbacks are actually needed anymore.
    callbacks.append(SetNoSync())
    callbacks.append(TeacherAlwaysEvalMode())
    callbacks.append(FreezeTeacher())

    # do explanation logging
    if args.explanation_logging:
        log_every = args.explanation_logging_every_n_epochs
        rank_zero_info(f"Will log explanations every {log_every} epoch(s)!")
        callbacks.append(
            custom_callbacks.ExplanationsLogger(log_every_n_epochs=log_every)
        )
    else:
        rank_zero_info("Explanation logging is disabled!")

    # for debugging purposes
    if args.debug:
        callbacks.append(custom_callbacks.ModelUpdateHasher())

    return callbacks

class SetNoSync(pl_callbacks.Callback):
    def on_train_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if hasattr(pl_module.trainer.model, 'no_sync'):
            pl_module.sync_ctx = pl_module.trainer.model.no_sync
        else:
            pl_module.sync_ctx = nullcontext

class TeacherAlwaysEvalMode(pl_callbacks.Callback):
    def on_train_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if pl_module.livekd_training:
            pl_module.teacher_model.eval()

class FreezeTeacher(pl_callbacks.BaseFinetuning):
    def freeze_before_training(self, pl_module):
        if pl_module.livekd_training:
            self.freeze(pl_module.teacher_model)

    def finetune_function(self, pl_module, current_epoch, optimizer, opt_idx):
        pass # Just HAD to be implemented :)