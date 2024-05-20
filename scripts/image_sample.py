"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import time

from PIL import Image
import numpy as np
import torch as th
import torch.distributed as dist
from tqdm import tqdm

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

BASE = os.path.abspath("../..")

th.backends.cuda.matmul.allow_tf32 = True
attacks = ["clean",
           "poisoning_simple_replacement-Mouth_Slightly_Open-Wearing_Lipstick",
           "poisoning_simple_replacement-High_Cheekbones-Male"]

def main():
    args = create_argparser().parse_args()

    if args.convert_to_images:
        print("Making images")
        for attack in attacks:
            for i in range(1, 11):
                save_path = os.path.join(BASE, "results", "DDPM-IP", "celeba", "DDPM-IP", attack, "noDef", str(i))
                save_npz = os.path.join(save_path, "samples_10000x64x64x3.npz")
                npz_to_images(save_npz, os.path.join(save_path, "images"))
        return

    dist_util.setup_dist()
    logger.configure()

    model_path = args.model_path
    model_name = os.path.basename(model_path).split(".")[0]

    os.makedirs(args.save_path, exist_ok=True)
    out_path = os.path.join(args.save_path, f"{model_name}_samples_{args.num_samples}x64x64x3.npz")
    if os.path.exists(out_path):
        logger.log(f"samples at {out_path} already exist.")
        return

    if not os.path.exists(model_path):
        logger.log(f"model at {model_path} did not exist")
        return

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    #model_path = args.model_path
    model.load_state_dict(
        dist_util.load_state_dict(model_path, map_location=dist_util.dev())
    )
    logger.log(f"loading checkpoint: {model_path}")
    logger.log(f"timesteps: {args.timestep_respacing}")
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []

    with tqdm(total=args.num_samples) as pbar:
        while len(all_images) * args.batch_size < args.num_samples:
            model_kwargs = {}
            if args.class_cond:
                classes = th.randint(
                    low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
                )
                model_kwargs["y"] = classes
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )

            #######Here is the heavy lifting##########
            sample = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                progress=True,
                device=dist_util.dev(),
            )

            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()



            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

            if args.class_cond:
                gathered_labels = [
                    th.zeros_like(classes) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(gathered_labels, classes)
                all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
            #logger.log(f"created {len(all_images) * args.batch_size} samples in {time_full2 - time_full1} seconds")
            pbar.update(args.batch_size)

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        #shape_str = "x".join([str(x) for x in arr.shape])
        #out_path = os.path.join(args.save_path, f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")

def npz_to_images(npz_path, images_path):
    images_npz = np.load(npz_path)["arr_0"]
    print(images_npz.shape)

    os.makedirs(images_path, exist_ok=True)

    _N, H, W, C = images_npz.shape
    images = images_npz.transpose(0, 1, 2, 3)
    print(f"{images.shape = }")
    with tqdm(total=len(images)) as pbar:
        for i, image in enumerate(images):
            fname = os.path.join(images_path, f"{i+1:06}.png")
            Image.fromarray(image, 'RGB').save(fname)
            #print(f"Converting npz to file {fname}", end="\r")
            pbar.update(1)

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        save_path="",
        convert_to_images=False,
        images_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
