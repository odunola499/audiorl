import torch
from torch import nn
from torch.nn import functional as F
from src.trainers import BaseTrainer
from src.configs import DPOConfig
from typing import Union, Tuple
from src.models import get_model
from src.utils import pad_to_length
from torch.utils.data import DataLoader

class DPOTrainer(BaseTrainer):
    def __init__(self,
                 model,
                 loaders:Union[Tuple[DataLoader], DataLoader],
                 train_config:DPOConfig,
                 ):
        super().__init__(
            model = model,
            loaders = loaders,
            args = train_config,
        )
        self.args = train_config
        if not self.args.ref_model:
            self.ref_model = get_model(self.model.model_name, size=self.model.model_size)().eval()
        else:
            self.ref_model = get_model(self.args.ref_model, size=self.model.model_size)().eval()


        self.pad_token_id = self.model.pad_token_id


    def concatenated_inputs(self, batch, padding_value:int = None) -> dict:
        if not padding_value:
            padding_value = self.pad_token_id

        output ={}
        output['audio_features'] = torch.cat([batch['audio_features'], batch['audio_features']], dim = 0)

        if batch['prompt_input_ids'] is not None:
            output['prompt_input_ids'] = torch.cat([batch['prompt_input_ids'], batch['prompt_input_ids']], dim = 0)
            output['prompt_attention_mask'] = torch.cat([batch['prompt_attention_mask'], batch['prompt_attention_mask']], dim = 0)
        else:
            output['prompt_input_ids'] = None
            output['prompt_attention_mask'] = None

        max_completion_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1])
        output['completion_input_ids'] = torch.cat(
            [
                pad_to_length(batch['chosen_input_ids'], max_completion_length, padding_value),
                pad_to_length(batch['rejected_input_ids'], max_completion_length, padding_value)
            ]
        )
        output['completion_attention_mask'] = torch.cat(
            [
                pad_to_length(batch['chosen_attention_mask'], max_completion_length, padding_value),
                pad_to_length(batch['rejected_attention_mask'], max_completion_length, padding_value)
            ]
        )
        return output


    def compute_ref_log_probs(self, batch):
        with torch.no_grad():
            ref_model_output = self.concatenated_forward(batch, is_ref_model=True)
        return ref_model_output['chosen_logps'], ref_model_output['rejected_logps']

    def concatenated_forward(self,batch = None, is_ref_model:bool = False, logits:torch.FloatTensor = None, labels:torch.LongTensor = None) -> dict:
        if is_ref_model:
            model = self.ref_model
        else:
            model = self.model

        num_examples = batch['chosen_input_ids'].shape[0]
        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.pad_token_id)

        model_kwargs = {'use_cache':False}
        model_kwargs['audio_features'] = concatenated_batch['audio_features']

        completion_input_ids = concatenated_batch['completion_input_ids']
        completion_attention_mask = concatenated_batch['completion_attention_mask']

        if concatenated_batch['prompt_input_ids'] is not None:
            prompt_input_ids = concatenated_batch['prompt_input_ids']
            prompt_attention_mask = concatenated_batch['prompt_attention_mask']

            input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
            attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
            loss_mask = torch.cat(
                (torch.zeros_like(prompt_attention_mask), completion_attention_mask), dim = 1
            )

        else:
            input_ids = completion_input_ids
            attention_mask = completion_attention_mask

            loss_mask = completion_attention_mask

        model_kwargs['attention_mask'] = attention_mask

        if is_ref_model:
            with torch.no_grad():
                model_output = model(
                    input_ids = input_ids,
                    **model_kwargs,
                )
        else:
            model_output = model(
                input_ids = input_ids,
                **model_kwargs,
            )

        logits = model_output.logits
        labels = torch.roll(
            input_ids, shifts = -1, dims = 1
        )
        loss_mask = torch.roll(
            loss_mask, shifts = -1, dims = 1
        ).bool()

        if logits.shape[:2] != labels.shape:
            seq_len = labels.shape[1]
            logits = logits[:, -seq_len:, :]

        labels[~loss_mask] = -100
        per_token_logps = torch.gather(logits.log_softmax(-1), dim = -1, index = labels.unsqueeze(-1)).squeeze(-1)
        per_token_logps[~loss_mask] = 0.0
        per_token_logps = torch.roll(per_token_logps, shifts = 1, dims = 1)

        all_logps = per_token_logps[:, 1:].sum(-1)

        output = {}
        output['chosen_logps'] = all_logps[:num_examples]
        output['rejected_logps'] = all_logps[num_examples:]

        mean_chosen_logits = logits[:num_examples][loss_mask[:num_examples]].mean()
        mean_rejected_logits = logits[num_examples:][loss_mask[num_examples:]].mean()

        output['mean_chosen_logits'] = mean_chosen_logits
        output['mean_rejected_logits'] = mean_rejected_logits

        return output

    def dpo_loss(self,
                 chosen_logps:torch.FloatTensor,
                 rejected_logps:torch.FloatTensor,
                 ref_chosen_logps:torch.FloatTensor,
                 ref_rejected_logps:torch.FloatTensor,
                 loss_type:str = 'sigmoid',
                 model_output:dict = None) -> tuple:

        logratios = chosen_logps - rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps

        logits = logratios - ref_logratios

        assert loss_type == 'sigmoid', 'Only sigmoid is currently supported'
        losses = (
                -F.logsigmoid(self.args.beta * logits) * (1 - self.args.label_smoothing)
                - F.logsigmoid(-self.args.beta * logits) * self.args.label_smoothing
            )

        chosen_rewards = (self.args.beta * chosen_logps) - ref_chosen_logps.detach()
        rejected_rewards = (self.args.beta * rejected_logps) - ref_rejected_logps.detach()

        return losses, chosen_rewards, rejected_rewards


    def compute_metrics(self, model:nn.Module = None, batch = None, model_output = None, mode = 'train'):
        metrics = {}

        if batch:
            model_output = self.concatenated_forward(batch, is_ref_model = False) #pass one
        else:
            if model_output is None:
                raise ValueError('Either batch or model_output must be provided')

        with torch.no_grad():
            ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch) # pass two



        losses, chosen_rewards, rejected_rewards =self.dpo_loss(
            model_output['chosen_logps'],
            model_output['rejected_logps'],
            ref_chosen_logps,
            ref_rejected_logps,
            model_output = model_output
        )

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        metrics[f'{mode}/rewards/chosen'] = chosen_rewards.mean().item()
        metrics[f'{mode}/rewards/rejected'] = rejected_rewards.mean().item()
        metrics[f'{mode}/rewards/accuracy'] = reward_accuracies.mean().item()
        metrics[f'{mode}/margins'] = (chosen_rewards - rejected_rewards).mean().item()
        metrics[f'{mode}/logps/chosen'] = (
            model_output['chosen_logps'].detach().mean().item()
        )
        metrics[f'{mode}/logps/rejected'] = (
            model_output['rejected_logps'].detach().mean().item()
        )
        metrics[f'{mode}/logits/chosen'] = (
            model_output['mean_chosen_logits'].detach().mean().item()
        )
        metrics[f'{mode}/logits/rejected'] = (
            model_output['mean_rejected_logits'].detach().mean().item()
        )
        metrics[f'{mode}/loss'] = losses.mean().item()
        return losses.mean(), metrics


    def train_step(self, batch):
        loss, metrics = self.compute_metrics(batch = batch)
        #todo: logging to wandb or csv,
        self.log(metrics)
        return loss

    def validation_step(self, batch):
        with torch.no_grad():
            loss, metrics = self.compute_metrics(batch = batch, mode = 'valid')
        self.log(metrics)
        return loss



