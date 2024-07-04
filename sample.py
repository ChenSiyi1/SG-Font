import argparse
import os

import numpy as np
import torch
import torch as th
import torch.distributed as dist
import blobfile as bf

from utils import dist_util, logger
from utils.script_util import (
    model_and_diffusion_defaults,
    args_to_dict,
    create_model_and_diffusion,
)
from PIL import Image
from attrdict import AttrDict
import yaml

def img_pre_pros(img_path, image_size):
    pil_image = Image.open(img_path).resize((image_size, image_size))
    pil_image.load()
    pil_image = pil_image.convert("RGB")
    arr = np.array(pil_image)
    arr = arr.astype(np.float32) / 127.5 - 1
    return np.transpose(arr, [2, 0, 1])

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default='./cfg/test_cfg.yaml',
                        help='config file path')
    parser.add_argument('--sty_img_path', type=str, default='data/imgs/Seen_TRAIN800/id_0')
    parser.add_argument('--img_save_path', type=str, default='./result/models/sf40uc800/id_0')
    parser.add_argument('--gen_txt_file', type=str, default='./TEST.txt')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--num_ref', type=int, default=1)
    # parser.add_argument('--sty_scale', type=int, default=3)
    # parser.add_argument('--cont_scale', type=int, default=3)
    parser = parser.parse_args()
    with open(parser.cfg_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = AttrDict(create_cfg(cfg))
    model_path = cfg.model_path
    cont_img_path = cfg.cont_img_path
    total_txt_file = cfg.total_txt_file
    classifier_free = cfg.classifier_free
    cond_scale = cfg.cond_scale

    num_samples = parser.num_samples
    num_ref = parser.num_ref
    sty_img_path = parser.sty_img_path  ##
    img_save_path = parser.img_save_path ##
    gen_txt_file = parser.gen_txt_file  ##
    # cont_guidance_scale = parser.cont_scale
    # sty_guidance_scale = parser.sty_scale

    cfg.__delattr__('model_path')
    cfg.__delattr__('total_txt_file')
    cfg.__delattr__('classifier_free')
    cfg.__delattr__('cond_scale')


    dist_util.setup_dist()

    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(cfg, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if cfg.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    noise = None

    # gen txt
    char2idx = {}
    with open(total_txt_file, 'r', encoding='utf-8') as f:
        chars = f.readlines()
        for idx, char in enumerate(chars[0]):
            char2idx[char] = idx
        f.close()

    char_idx = []
    with open(gen_txt_file, 'r', encoding="utf-8") as f1:
        genchars = f1.readlines()
        for char in genchars[0]:
            char_idx.append(char2idx[char])
        f1.close()
    source_files = _list_image_files_recursively(cont_img_path)
    all_images = []
    all_labels = []

    ch_idx = 0
    while len(all_images) * cfg.batch_size < num_samples:
        model_kwargs = {}
        classes_lable = th.tensor([i for i in char_idx[ch_idx:ch_idx + cfg.batch_size]], device=dist_util.dev())
        ch_idx += cfg.batch_size

        sty_sum = []
        ref_imgs = _list_image_files_recursively(sty_img_path)
        ref_imgs = ref_imgs[:num_ref]
        for i, ref_img_path in enumerate(ref_imgs):
            img = th.tensor(img_pre_pros(ref_img_path, cfg.image_size), requires_grad=False).cuda().repeat(
                cfg.batch_size, 1, 1, 1)
            # sty_sum += model.sty_encoder(img)
            sty_sum = model.encode(img)['cond']
        # sty_feat = sty_sum / len(ref_imgs)
        sty_feat = sty_sum
        model_kwargs["sty"] = sty_feat
        cond = [sty_feat, model.encode(th.zeros_like(img))['cond']]

        cont_imgs = []
        for idx in classes_lable:
            cont_img_path = source_files[idx]
            cont_img = th.tensor(img_pre_pros(cont_img_path, cfg.image_size), requires_grad=False).cuda()
            cont_imgs.append(cont_img)
        batch_imgs = th.stack(cont_imgs, dim=0)

        model_kwargs['source'] = batch_imgs


        def model_fn(x_t, ts, cond, cond_scale, **model_kwargs):
            if classifier_free:
                cond_output = model(x_t, ts, cond=cond, cond_scale=cond_scale, prob=1)
                uncond_output = model(x_t, ts, cond=cond, cond_scale=cond_scale, prob=0)
                model_output = uncond_output + cond_scale * (cond_output - uncond_output)

                # c_cond_output = model(x_t , ts, cond=cond, cond_scale=cond_scale, prob=0)
                # x = x_t[:, :3]
                # x_t = torch.cat([x, torch.ones_like(x)], dim=1)
                # s_cond_output = model(x_t, ts, cond=cond, cond_scale=cond_scale, prob=1)
                # uncond_output = model(x_t, ts, cond=cond, cond_scale=cond_scale, prob=0)
                # model_output = uncond_output + cond_scale * (c_cond_output - uncond_output) + cond_scale * (s_cond_output - uncond_output)

            else:
                model_output = model(x_t, ts, cond=cond, prob=1)
            return model_output

        # noise_schedule = NoiseScheduleVP(schedule='liner', continuous_beta_0=0.1, continuous_beta_1=20)
        # model_fn = model_wrapper(
        #     model,
        #     noise_schedule,
        #     model_type="noise",  # or "x_start" or "v" or "score"
        #     model_kwargs=model_kwargs,
        #     guidance_type="classifier-free",
        #     condition=condition,
        #     unconditional_condition=unconditional_condition,
        #     guidance_scale=cont_guidance_scale,
        # )
        #
        # dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++", correcting_x0_fn="dynamic_thresholding")
        # x_sample = dpm_solver.sample(
        #     x_T,
        #     steps=20,
        #     order=2,
        #     skip_type="time_uniform",
        #     method="multistep",
        # )

        sample_fn = (
            diffusion.p_sample_loop if not cfg.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fn,
            cond,
            cond_scale,
            (cfg.batch_size, 3, cfg.image_size, cfg.image_size),
            clip_denoised=cfg.clip_denoised,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
            noise=noise,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

        gathered_labels = [
            th.zeros_like(classes_lable) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(gathered_labels, classes_lable)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * cfg.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: num_samples]
    if dist.get_rank() == 0:
        for idx, (img_sample, img_cls) in enumerate(zip(arr, label_arr)):
            img = Image.fromarray(img_sample).convert("RGB")
            img_name = "%04d.png" % (idx)
            img.save(os.path.join(img_save_path, img_name))

    dist.barrier()
    logger.log("sampling complete")


def create_cfg(cfg):
    defaults = dict(
        clip_denoised=True,
        num_samples=100,
        batch_size=16,
        use_ddim=False,
        model_path="",
        cont_scale=1.0,
        sk_scale=1.0,
        sty_img_path="",
        stroke_path=None,
        attention_resolutions='40, 20, 10',
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(cfg)
    return defaults


if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    end_time = time.time()
    exe_time = end_time - start_time
    print(exe_time)
