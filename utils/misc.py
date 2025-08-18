import torch
from collections import OrderedDict
import torch.distributed as dist

class ListAverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None
        self.avg = []
        self.sum = []
        self.count = 0

    def update(self, val):

        if self.val is None:
            # initialize the self.val list with 0 values, use length of val
            self.val = [0] * len(val)
            self.sum = [0] * len(val)
            self.avg = [0] * len(val)

        print("val:",len(val))
        print("self.val:",len(self.val))
        

        for i in range(len(val)):
            self.count += 1
            self.sum[i] = self.val[i] + val[i] 
            self.avg[i] = self.sum[i] / self.count
        self.val = val

def reduce_tensor(tensor):
    import time
    #print("Inside reduce:",tensor)
    rt = tensor.clone()
    start = time.time()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    #print("time for reduce op", time.time()-start)

    rt /= dist.get_world_size()
    return rt

def parse_losses(losses, mean=True):
    '''Parse the losses dict and return the total loss and log_vars.'''
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(f'{loss_name} is not a tensor or list of tensors')

    if not mean: 
        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)
    else:
        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key) / len(log_vars)

    return loss, log_vars

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item()**norm_type
    total_norm = total_norm**(1. / norm_type)
    return total_norm
