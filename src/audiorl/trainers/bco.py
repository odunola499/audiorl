import torch
from torch.nn import functional as F
from .dpo import DPOModule
from .utils import RunningMoments



class BCOModule(DPOModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)

        self.running = RunningMoments()

    def bco_loss(self,
                 chosen_logps: torch.FloatTensor,
                 rejected_logps: torch.FloatTensor,
                 ref_chosen_logps: torch.FloatTensor,
                 ref_rejected_logps: torch.FloatTensor,
                 ) -> tuple:

        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps
        chosen_rewards = self.args.beta * chosen_logratios
        rejected_rewards = self.args.beta * rejected_logratios
        rewards = torch.cat((chosen_rewards, rejected_rewards), dim=0).mean().detach()

        global_rewards = self.all_gather(rewards)
        global_rewards = global_rewards.reshape(-1, *global_rewards.shape[2:])

        rewards_var, rewards_mean = torch.var_mean(global_rewards, unbiased = False)
        rewards_count = global_rewards.numel()
        self.running.update(rewards_mean, rewards_var, rewards_count)
        delta = self.running.mean
        losses =  -F.logsigmoid((self.args.beta * chosen_logratios) - delta) - F.logsigmoid(
                -(self.args.beta * rejected_logratios - delta)
            )
        chosen_rewards = self.args.beta * (chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = (self.args.beta * rejected_logps) - ref_rejected_logps.detach()

        return losses, chosen_rewards, rejected_rewards

    def compute_metrics(self, batch, mode = 'train'):
        metrics = {}
        model_output = self.concatenated_forward(batch, is_ref_model=False)  # pass one

        with torch.no_grad():
            ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch)  # pass two

        losses, chosen_rewards, rejected_rewards = self.bco_loss(
            model_output['chosen_logps'],
            model_output['rejected_logps'],
            ref_chosen_logps,
            ref_rejected_logps,
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