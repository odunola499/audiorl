import torch
from torch.nn import functional as F
from .bco import BCOModule
from .utils import RunningMoments
from ..configs.mpo import MPOArgs



def cross_entropy_loss(logits:torch.Tensor, labels:torch.Tensor, lengths:torch.Tensor):
    log_probs = F.log_softmax(logits, dim = -1)
    nll = F.nll_loss(
        log_probs.transpose(1,2), labels, reduction = 'none'
    )
    mask = torch.arange(logits.size(1), device = lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
    nll *= mask

    per_seq_loss = nll.sum(dim = 1) / lengths
    return per_seq_loss.mean()



class MPOTrainer(BCOModule):
    def __init__(self,*args, train_config:MPOArgs,  **kwargs):
        super().__init__(*args, train_config = train_config, **kwargs)

        self.running = RunningMoments()
        self.args = train_config


    def mpo_loss(self,
                 chosen_logps:torch.FloatTensor,
                 rejected_logps:torch.FloatTensor,
                 ref_chosen_logps:torch.FloatTensor,
                 ref_rejected_logps:torch.FloatTensor,
                 chosen_logits:torch.FloatTensor,
                 chosen_labels:torch.LongTensor,
                 chosen_lengths:torch.LongTensor
                 ) -> tuple:

        generation_loss = cross_entropy_loss(chosen_logits, chosen_labels,chosen_lengths)

        dpo_losses, dpo_chosen_rewards, dpo_rejected_rewards = self.dpo_loss(
            chosen_logps,
            rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
        )

        bco_losses, bco_chosen_rewards, bco_rejected_rewards = self.bco_loss(
            chosen_logps,
            rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
        )

        losses = (self.args.loss_weights[0] * generation_loss +
                  self.args.loss_weights[1] * dpo_losses +
                  self.args.loss_weights[2] * bco_losses)

        chosen_rewards = (self.args.loss_weights[1] * dpo_chosen_rewards +
                          self.args.loss_weights[2] * bco_chosen_rewards)
        rejected_rewards = (self.args.loss_weights[1] * dpo_rejected_rewards +
                            self.args.loss_weights[2] * bco_rejected_rewards)

        return losses, chosen_rewards, rejected_rewards

    def compute_metrics(self, batch, mode = 'train'):
        metrics = {}
        model_output = self.concatenated_forward(batch, is_ref_model=False)

        chosen_logits = model_output['chosen_logits']
        chosen_labels = model_output['chosen_input_ids']
        chosen_lengths = model_output['chosen_lengths']

        with torch.no_grad():
            ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch)  # pass two

        losses, chosen_rewards, rejected_rewards = self.mpo_loss(
            model_output['chosen_logps'],
            model_output['rejected_logps'],
            ref_chosen_logps,
            ref_rejected_logps,
            chosen_logits = chosen_logits,
            chosen_labels = chosen_labels,
            chosen_lengths = chosen_lengths

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


