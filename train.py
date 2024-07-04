import argparse
from utils import dist_util, logger
from utils.image_datasets import load_data
from utils.resample import create_named_schedule_sampler
from utils.script_util import (
    model_and_diffusion_defaults,
    args_to_dict,
    create_model_and_diffusion,
)
from utils.train_util import TrainLoop
import torch as th
from attrdict import AttrDict
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default='./cfg/train_cfg.yaml',
                        help='config file path')
    parser = parser.parse_args()
    with open(parser.cfg_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = AttrDict(create_cfg(cfg))
    train_step = cfg.train_step
    total_train_step = cfg.total_train_step
    classifier_free = cfg.classifier_free
    cfg.__delattr__('train_step')
    cfg.__delattr__('total_train_step')
    cfg.__delattr__('classifier_free')

    dist_util.setup_dist()

    model_save_dir = cfg.model_save_dir  

    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    logger.configure(dir=model_save_dir, format_strs=['stdout', 'log', 'csv']) 

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(cfg, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(cfg.schedule_sampler, diffusion)


    logger.log("creating data loader...")
    data = load_data(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        image_size=cfg.image_size,
        classifier_free=classifier_free,
        source_data_dir=cfg.source_data_dir
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=cfg.batch_size,
        microbatch=cfg.microbatch,
        lr=cfg.lr,
        ema_rate=cfg.ema_rate,
        log_interval=cfg.log_interval,
        save_interval=cfg.save_interval,
        test_interval=cfg.test_interval,
        train_step=train_step,
        resume_checkpoint=cfg.resume_checkpoint,
        use_fp16=cfg.use_fp16,
        fp16_scale_growth=cfg.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=cfg.weight_decay,
        classifier_free=classifier_free,
        total_train_step=total_train_step
    ).run_loop()


def create_cfg(cfg):
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=250,
        save_interval=20000,
        test_intertal=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        attention_resolutions='40, 20, 10',
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(cfg)
    return defaults


if __name__ == "__main__":
    import os
    main()