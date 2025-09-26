import os
import torch
from abc import ABC, abstractmethod
from torch import nn
from src.models import ASRBaseModel
from typing import  Union, Tuple, Literal
from src.configs import DPOConfig as DPOArguments
from src.trainers import OPTIMIZERS, SCHEDULERS

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from src.loggers import WandbLogger, CSVLogger


def _build_distributed_loader(loader: DataLoader, mode="train"):
    sampler = DistributedSampler(loader.dataset, shuffle=(mode == "train"))
    new_loader = DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        sampler=sampler,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        collate_fn=loader.collate_fn,
        drop_last=loader.drop_last
    )
    return new_loader


class BaseTrainer(ABC):
    def __init__(self,
                 model:ASRBaseModel,
                 loaders:Union[Tuple[DataLoader], DataLoader],
                 args:DPOArguments,
                 ):
        super().__init__()

        self.global_batch_size = None
        self.test_loader = None
        self.valid_loader = None
        self.train_loader = None
        self.model = None
        self.device = None
        self.args = args
        self.raw_model = model
        self.ref_model = None
        self.raw_loaders = loaders
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.logger = None
        self.cur_epoch = 0
        self.cur_step = 0
        self.gpu_id = int(os.environ.get("LOCAL_RANK", "0"))


    def forward(self):
        pass

    @abstractmethod
    def train_step(self, batch):
        pass

    @abstractmethod
    def validation_step(self, batch):
        pass

    @abstractmethod
    def compute_metrics(self, model: nn.Module = None, batch=None, model_output=None,
                        mode: Literal['train', 'valid', 'test'] = 'valid') -> dict:
        pass
    
    
    def _load_snapshot(self, path:str):
        checkpoint = torch.load(path, map_location = f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu")
        self.model.module.load_state_dict(checkpoint["model"])
        if self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler and checkpoint["scheduler"]:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        if "global_step" in checkpoint:
            self.cur_step = checkpoint["global_step"]
        if "epoch" in checkpoint:
            self.cur_epoch = checkpoint["epoch"]
        if "args" in checkpoint:
            self.args = checkpoint["args"]

    def _save_snapshot(self, path:str):
        checkpoint = {
            "model": self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "global_step": self.cur_step,
            "epoch": self.cur_epoch,
            "args": self.args,
        }
        torch.save(checkpoint, path)
        if self.args.push_to_hub and self.gpu_id == 0:
            try:
                self.model.module.model.push_to_hub(self.args.hub_model_id, commit_message=f"Saving model at step {self.cur_step}", private= True)
                if self.model.module.tokenizer is not None:
                    self.model.module.tokenizer.push_to_hub(self.args.hub_model_id, commit_message=f"Saving tokenizer at step {self.cur_step}", private= True)
                if self.model.module.feature_extractor is not None:
                    self.model.module.feature_extractor.push_to_hub(self.args.hub_model_id, commit_message=f"Saving feature extractor at step {self.cur_step}", private= True)
            except Exception as e:
                print(f"Failed to push model to hub: {e}")

    
    def _save_checkpoint(self, epoch:int, step:int):
        pass

    def log(self, logs:dict) -> None:
        self.logger.log_metrics(logs, step = self.cur_step)

    def log_audio(self, audios:list, captions:list) -> None:
        self.logger.log_audio_artifacts(audios, captions, step = self.cur_step, sample_rate = 16000)

    def configure_model_and_optimizer(self):
        if torch.cuda.is_available() and self.args.accelerator in ['gpu','auto']:
            torch.cuda.set_device(self.gpu_id)
            self.device = torch.device(f"cuda:{self.gpu_id}")

            self.model = self.raw_model.to(self.device)
            self.model = DDP(self.model, device_ids = [self.gpu_id])

            if self.ref_model:
                self.ref_model = self.ref_model.to(self.device)
                self.ref_model = DDP(self.ref_model, device_ids = [self.gpu_id])
                for param in self.ref_model.parameters():
                    param.requires_grad = False
        else:
            self.device = torch.device("cpu")
            self.model = self.raw_model.to(self.device)
            self.model = DDP(self.model)
            if self.ref_model:
                self.ref_model = self.ref_model.to(self.device)
                self.ref_model = DDP(self.ref_model)
                for param in self.ref_model.parameters():
                    param.requires_grad = False

        del self.raw_model

        optimizer_cls = OPTIMIZERS[self.args.optimizer]
        scheduler_cls = SCHEDULERS[self.args.scheduler]

        opt_kwargs = {
            'lr': self.args.lr,
            'weight_decay': self.args.weight_decay,
            'fused': True if self.device.type == 'cuda' else False
        }

        optimizer = optimizer_cls(self.model.parameters(), **opt_kwargs)

        if self.args.scheduler in ['cosine_warmup']:
            assert self.args.warmup_steps > 0, 'Warmup steps must be > 0 for cosine scheduler'

            scheduler = scheduler_cls(
                optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=self.args.max_steps if self.args.max_steps else len(
                    self.train_loader) * self.args.num_epochs
            )
        elif self.args.scheduler in ['token_cosine_warmup']:
            assert self.args.warmup_steps > 0 or self.args.warmup_ratio > 0, 'Either warmup steps or warmup ratio must be > 0 for token cosine scheduler'
            assert self.args.max_train_tokens > 0, 'Total tokens must be > 0 for token cosine scheduler'
            assert self.global_batch_size > 0, 'Global batch size must be > 0 for token cosine scheduler'

            scheduler = scheduler_cls(
                optimizer,
                total_tokens=self.args.max_train_tokens,
                global_batch_size=self.global_batch_size,
                seq_len=1024,
                warmup_steps=self.args.warmup_steps,
                warmup_ratio=self.args.warmup_ratio
            )
        else:
            raise ValueError(f'Unknown scheduler {self.args.scheduler}')

        if self.args.precision == 'bf16_mixed':
            self.scaler = torch.cuda.amp.GradScaler()

        self.optimizer = optimizer
        self.scheduler = scheduler



    def setup_train_dataloader(self, train_loader):
        if not hasattr(train_loader, "__iter__"):
            raise TypeError("train_loader must be iterable")
        self.train_loader = train_loader

    def setup_valid_dataloader(self, valid_loader):
        if not hasattr(valid_loader, "__iter__"):
            raise TypeError("valid_loader must be iterable")
        self.valid_loader = valid_loader

    def _run_batch(self, batch, batch_idx = None):
        self.optimizer.zero_grad()
        loss = self.train_step(batch)
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()
        print(f"Epoch {self.cur_epoch} Step {self.cur_step} Loss {loss.item():.4f}")

    def _run_eval_batch(self, batch, batch_idx = None):
        loss = self.validation_step(batch)

        print(f"Validation Step {batch_idx} Loss {loss.item():.4f}")
        return loss

    def _run_epoch(self):
        device = self.device
        self.model.train()

        for index, batch in enumerate(self.train_loader):
            self.cur_step = index
            batch = {k:v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
            self._run_batch(batch, batch_idx = index)

            if index % self.args.eval_every == 0 and index > 0:
                self._run_eval_epoch()

    def _run_eval_epoch(self):
        device = self.device
        self.model.eval()

        with torch.no_grad():
            for val_index, val_batch in enumerate(self.valid_loader):
                if val_index >= self.args.num_eval_steps:
                    break

                val_batch = {k:v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k,v in val_batch.items()}
                self._run_eval_batch(val_batch, batch_idx = val_index)
        self.model.train()

    def _configure_logger(self):
        if self.args.log_to == 'wandb':
            self.logger = WandbLogger(self.args)
        else:
            self.logger = CSVLogger(self.args)
        self.logger.init_and_log_args(self.args)

    def train(self):

        if self.args.accelerator == 'gpu':
            backend = 'nccl'
        elif self.args.accelerator == 'cpu':
            backend = 'gloo'
            self.args.precision = 'float32' #force float32 for cpu
        elif self.args.accelerator == 'auto':
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        else:
            raise ValueError(f'Unknown accelerator {self.args.accelerator}')

        init_process_group(backend = backend)

        try:
            self.configure_model_and_optimizer()

            if isinstance(self.raw_loaders, (tuple, list)):
                n = len(self.raw_loaders)
                if n >= 1: self.train_loader = _build_distributed_loader(self.raw_loaders[0], mode="train")
                if n >= 2: self.valid_loader = _build_distributed_loader(self.raw_loaders[1], mode="valid")
                if n == 3: self.test_loader = _build_distributed_loader(self.raw_loaders[2], mode="test")
            else:
                self.train_loader = _build_distributed_loader(self.raw_loaders, mode="train")

            self.global_batch_size = self.train_loader.batch_size * torch.distributed.get_world_size()

            if os.path.exists(self.args.snapshot_path):
                self._load_snapshot(self.args.snapshot_path)

            rank = torch.distributed.get_rank()

            for epoch in range(self.cur_epoch, self.args.num_epochs):
                self.cur_epoch = epoch
                if isinstance(self.train_loader.sampler, DistributedSampler):
                    self.train_loader.sampler.set_epoch(epoch)
                self._run_epoch()
                if (epoch + 1) % self.args.save_every == 0 and self.gpu_id == 0:
                    self._save_snapshot(self.args.snapshot_path)

            if rank == 0:
                self._save_snapshot(self.args.snapshot_path)

        finally:
            destroy_process_group()





