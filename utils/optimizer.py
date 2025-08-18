from torch import optim as optim
from utils import get_logger

def check_items_in_string(items, key):
    for item in items:
        if item in key:
            return True
    return False

def set_gradient(model, finetune):
    """Set gradient for finetuning."""
    #get the logger
    logger = get_logger()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #get all the items from config.finetune, for every key in the config, freeze all the layers having the name in the corresponding value list
    if finetune:
        #logger.info(f'Finetuning the following layers: {finetune}')
        for _, value in finetune.items():
            for name, param in model.named_parameters():
                #print(name)
                if check_items_in_string(value, name):
                    param.requires_grad = False
                    #print(f"Freezing {name}")
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model


def build_optimizer(config, model):
    """Build optimizer, set weight decay of normalization to 0 by default."""

    #For finetuning, freeze the selected layers
    if config.finetune:
        model = set_gradient(model, config.finetune)
    
    parameters = set_weight_decay(model, {}, {})
    
    opt_name = config.optimizer.name
    optimizer = None
    if opt_name == 'adamw':
        optimizer = optim.AdamW(
            parameters,
            lr=config.optimizer.base_lr,
            eps=config.optimizer.eps,
            betas=config.optimizer.betas,
            weight_decay=config.weight_decay)
    else:
        raise ValueError(f'Unsupported optimizer: {opt_name}')
    
    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith('.bias') or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay}, {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin