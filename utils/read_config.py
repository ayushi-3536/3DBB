import os
import os.path as osp
from omegaconf import OmegaConf

def load_config(cfg_file):
    cfg = OmegaConf.load(cfg_file)
    if '_base_' in cfg:
        if isinstance(cfg._base_, str):
            base_cfg = OmegaConf.load(osp.join(osp.dirname(cfg_file), cfg._base_))
        else:
            base_cfg = OmegaConf.merge(OmegaConf.load(f) for f in cfg._base_)
        cfg = OmegaConf.merge(base_cfg, cfg)
    return cfg

def get_config(args):
    cfg = load_config(args.cfg)
    OmegaConf.set_struct(cfg, True)

    if args.opts is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.opts))
    if hasattr(args, 'batch_size') and args.batch_size:
        cfg.data.batch_size = args.batch_size
        
    if hasattr(args, 'path_imgref_test') and args.path_imgref_test:
        cfg.data.meta.tryGBIF.path_imgref_test = args.path_imgref_test
    
    if hasattr(args, 'seed') and args.seed:
        cfg.seed= args.seed

    if hasattr(args, 'resume') and args.resume:
        cfg.checkpoint.resume = args.resume

    if hasattr(args, 'eval') and args.eval:
        cfg.evaluate.eval_only = args.eval

    if not cfg.model_name:
        cfg.model_name = osp.splitext(osp.basename(args.cfg))[0]

    world_size = int(os.environ.get('WORLD_SIZE', 1))
    cfg.model_name = cfg.model_name + f'_bs{cfg.data.batch_size}x{world_size}'

    if hasattr(args, 'output') and args.output:
        cfg.output = args.output
    else:
        cfg.output = osp.join('output', cfg.model_name)

    if hasattr(args, 'tag') and args.tag:
        cfg.tag = args.tag
        cfg.output = osp.join(cfg.output, cfg.tag)

    if hasattr(args, 'wandb') and args.wandb:
        cfg.wandb = args.wandb


    cfg.local_rank = int(os.environ.get('LOCAL_RANK', 0))

    OmegaConf.set_readonly(cfg, True)

    return cfg
