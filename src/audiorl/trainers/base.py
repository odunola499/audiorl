import lightning as pl
from src.audiorl.models import PolicyModel
from typing import Literal, List, Union
from src.audiorl.configs import TrainArgs
from src.audiorl.trainers import OPTIMIZERS, SCHEDULERS
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.strategies import Strategy
from typing import Optional

class BaseModule(pl.LightningModule):
    def __init__(
            self,
            model: PolicyModel,
            args:TrainArgs,
            train_loader:DataLoader,
            valid_loader:DataLoader = None,
            test_loader:DataLoader = None,
    ):
        super().__init__()
        self.model = model
        self.ref_model = None
        self.args = args
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.save_hyperparameters(args.to_dict())

    def configure_optimizers(self):
        if self.ref_model:
            for param in self.ref_model.parameters():
                param.requires_grad = False

        optimizer_class = OPTIMIZERS[self.args.optimizer]
        opt_kwargs = {
            'lr': self.args.lr,
            'weight_decay': self.args.weight_decay,
            'fused': True if self.device.type == 'cuda' else False
        }
        optimizer = optimizer_class(
            self.model.parameters(),
            **opt_kwargs
        )
        scheduler_class = SCHEDULERS[self.args.scheduler]
        if self.args.scheduler in ['cosine_warmup']:
            assert self.args.warmup_steps > 0, 'Warmup steps must be > 0 for cosine scheduler'

            scheduler = scheduler_class(
                optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=self.args.max_steps if self.args.max_steps else len(
                    self.train_loader) * self.args.num_epochs
            )
        elif self.args.scheduler in ['token_cosine_warmup']:
            assert self.args.warmup_steps > 0 or self.args.warmup_ratio > 0, 'Either warmup steps or warmup ratio must be > 0 for token cosine scheduler'
            assert self.args.max_train_tokens > 0, 'Total tokens must be > 0 for token cosine scheduler'
            assert self.global_batch_size > 0, 'Global batch size must be > 0 for token cosine scheduler'

            scheduler = scheduler_class(
                optimizer,
                total_tokens=self.args.max_train_tokens,
                global_batch_size=self.global_batch_size,
                seq_len=1024,
                warmup_steps=self.args.warmup_steps,
                warmup_ratio=self.args.warmup_ratio
            )
        else:
            raise ValueError(f'Unknown scheduler {self.args.scheduler}')


        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def train_dataloader(self):
        if self.train_loader is None:
            raise ValueError("Train dataloader is not provided.")
        return self.train_loader

    def val_dataloader(self):
        if self.valid_loader:
            return self.valid_loader
        return None

    def test_dataloader(self):
        if self.test_loader:
            return self.test_loader
        return None

    def validation_step(self, batch):
        pass


    def training_step(self, batch):
        pass

    def compute_metrics(self, batch, mode: Literal['train', 'valid', 'test'] = 'valid') -> dict:
        pass


class Trainer:
    def __init__(self, module:BaseModule, strategy: Union[ str,Strategy] = 'ddp', callbacks:Optional[List[Callback]] = None):
        self.logger = None
        self.checkpoint = None
        self.callbacks = [] if not callbacks else callbacks
        self.args = module.args
        self.module = module
        self.strategy = strategy

    def setup_logger(self):
        if self.args.log_to == 'wandb':
            try:
                from lightning.pytorch.loggers import WandbLogger
            except:
                raise ValueError('wandb not install, please pip install')
            logger = WandbLogger(
                name = self.args.experiment_name,
                project = self.args.project_name,
            )
        elif self.args.log_to == 'comet':
            try:
                from lightning.pytorch.loggers import CometLogger
            except:
                raise ValueError('comet not installed, please pip install')
            logger = CometLogger(
                experiment_name=self.args.experiment_name,
                project_name=self.args.project_name
            )
        elif self.args.log_to == 'neptune':
            try:
                from lightning.pytorch.loggers import NeptuneLogger
            except:
                raise ValueError('neptune not installed, please pip install')
            logger = NeptuneLogger(
                name = self.args.experiment_name,
                project = self.args.project_name
            )
        else:
            print('No logger specified, defaulting to csv logger')
            from lightning.pytorch.loggers import CSVLogger
            logger = CSVLogger(
                name = self.args.experiment_name,
                save_dir = self.args.project_name
            )

        self.logger = logger

    def setup_callbacks(self):
        checkpoint = ModelCheckpoint(
            monitor='train/loss',
            dirpath=self.args.snapshot_path,
            filename=f'model-{{epoch:02d}}-{{steps:02d}}-{{val_loss:.3f}}',
            save_top_k=self.args.save_num_checkpoints,
            mode='min',
            every_n_train_steps=self.args.save_num_checkpoints,
            save_last=True,
            verbose=True
        )
        lr_monitor = LearningRateMonitor(logging_interval='step')
        self.callbacks.append([checkpoint, lr_monitor])



    def fit(self):
        self.setup_logger()
        self.setup_callbacks()
        engine = pl.Trainer(
            devices = self.args.devices,
            accelerator = self.args.accelerator,
            precision = self.args.precision,
            max_epochs = self.args.num_epochs if self.args.max_steps is None else None,
            max_steps = self.args.max_steps,
            accumulate_grad_batches= self.args.gradient_accumulation_steps,
            val_check_interval = self.args.eval_every,
            log_every_n_steps = self.args.log_every,
            enable_checkpointing=True,
            logger = self.logger,
            callbacks = self.callbacks,
            enable_model_summary = True,
            limit_val_batches = self.args.num_eval_steps
        )

        engine.fit(self.module)




