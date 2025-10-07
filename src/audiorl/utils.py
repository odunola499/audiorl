import torch


def pad_to_length(tensor:torch.Tensor, length:int, pad_value:int, dim = -1):
    B, seq_len = tensor.shape
    if tensor.size(dim) >= length:
        return tensor

    pad_amount = length - seq_len
    result = torch.nn.functional.pad(tensor, (0, pad_amount), value=pad_value)
    return result