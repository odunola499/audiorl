import torch
from torch.nn import functional as F
from .dpo import DPOTrainer

class RunningMoments:
    def __init__(self):
        self.mean = 0
        self.std = 1
        self.var = 1
        self.count = 1e-24

    def update(self, xs_mean, xs_var, xs_count):
        delta = xs_mean - self.mean
        tot_count = self.count + xs_count
        new_sum = xs_var * xs_count
        old_sum = self.var * self.count + delta ** 2 * self.count * xs_count / tot_count
        tot_sum = old_sum + new_sum

        self.mean += (delta * xs_count / tot_count).item()
        new_var = tot_sum / tot_count
        self.std = (new_var * tot_count / (tot_count - 1)).float().sqrt().item()
        self.var = new_var.item()
        self.count = tot_count

        return xs_mean.item(), (xs_var * xs_count / (xs_count - 1)).float().sqrt().item()




class BCOTrainer(DPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)

        self.running = RunningMoments()

    def dpo_loss(self,
                 chosen_logps: torch.FloatTensor,
                 rejected_logps: torch.FloatTensor,
                 ref_chosen_logps: torch.FloatTensor,
                 ref_rejected_logps: torch.FloatTensor,
                 loss_type: str = 'sigmoid',
                 model_output: dict = None) -> tuple:

        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps
        chosen_rewards = self.args.beta * chosen_logratios
        rejected_rewards = self.args.beta * rejected_logratios
        rewards = torch.cat((chosen_rewards, rejected_rewards), dim=0).mean().detach()
        #todo: rewards_mean, rewards_count, rewards_var needs to be distributed with all_reduce
        rewards_var, rewards_mean = torch.var_mean(rewards, unbiased = False)
        rewards_count = rewards.numel()
        self.running.update(rewards_mean, rewards_var, rewards_count)
        delta = self.running.mean
        losses =  -F.logsigmoid((self.args.beta * chosen_logratios) - delta) - F.logsigmoid(
                -(self.args.beta * rejected_logratios - delta)
            )
        chosen_rewards = self.args.beta * (chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = (self.args.beta * rejected_logps) - ref_rejected_logps.detach()

        return losses, chosen_rewards, rejected_rewards
