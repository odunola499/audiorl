import torch

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