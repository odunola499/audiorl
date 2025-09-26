import os
import csv
import numpy as np
from abc import ABC, abstractmethod
from src.configs import TrainArgs
from typing import List

class BaseLogger(ABC):
    def __init__(self, args:TrainArgs, **kwargs):
        self.args = args

    @abstractmethod
    def init_and_log_args(self, args:TrainArgs):
        pass

    @abstractmethod
    def log_metrics(self, metrics:dict, step:int, prefix:str = ''):
        pass

    @abstractmethod
    def log_audio_artifacts(self, audios: List[np.ndarray],
                            captions: List[str],
                            step:int,
                            sample_rate:int = 16000,
                            prefix:str = ''):
        pass

class CSVLogger(BaseLogger):
    def __init__(self, args: TrainArgs, **kwargs):
        super().__init__(args, **kwargs)
        os.makedirs(args.log_dir, exist_ok=True)
        self.csv_path = os.path.join(args.log_dir, "metrics.csv")
        self._file = open(self.csv_path, "w", newline="")
        self._writer = None

    def init_and_log_args(self, args: TrainArgs):
        args_path = os.path.join(self.args.log_dir, "hparams.txt")
        with open(args_path, "w") as f:
            for k, v in vars(args).items():
                f.write(f"{k}: {v}\n")

    def log_metrics(self, metrics: dict, step: int, prefix: str = ""):
        if step % self.args.log_every != 0:
            return
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        metrics = {"step": step, **metrics}
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(metrics.keys()))
            self._writer.writeheader()
        self._writer.writerow(metrics)
        self._file.flush()

    def log_audio_artifacts(
        self,
        audios: List[np.ndarray],
        captions: List[str],
        step: int,
        sample_rate: int = 16000,
        prefix: str = "",
    ):
        return

    def __del__(self):
        if hasattr(self, "_file") and not self._file.closed:
            self._file.close()


class WandbLogger(BaseLogger):
    def __init__(self, args:TrainArgs, **kwargs):
        super().__init__(args, **kwargs)
        import wandb
        self.wandb = wandb


    def init_and_log_args(self, args:TrainArgs):
        self.wandb.init(project=args.project_name, name=args.experiment_name, config=vars(args),
                        entity = args.wandb_entity)

    def log_metrics(self, metrics:dict, step:int, prefix:str = None):
        if prefix:
            first_nest = list(metrics.values())[0].split('/')[0]
            if first_nest != prefix:
                metrics = {f'{prefix}/{k}': v for k, v in metrics.items()}

        if step % self.args.log_every == 0:
            self.wandb.log(metrics, step=step)

    def log_audio_artifacts(self, audios: List[np.ndarray],
                            captions: List[str],
                            step:int,
                            sample_rate:int = 16000,
                            prefix:str = ''):
        if step % self.args.log_audio_every == 0 and prefix == 'valid':
            for i, (audio, caption) in enumerate(zip(audios, captions)):
                self.wandb.log({f'{prefix}/audio_{i}':
                                    self.wandb.Audio(audio,
                                                     caption=caption,
                                                     sample_rate=sample_rate)
                                }, step=step
                               )
                if i + 1 >= self.args.log_n_audio:
                    break

