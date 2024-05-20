# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import PIL.Image


import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

BASE = os.path.abspath("../..")

def save_image_grid(img, fname, drange, grid_size):
    img = np.asarray(img, dtype=np.uint8)
    """
    lo, hi = drange
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)
    """

    gw, gh = grid_size
    _N, W, H, C = img.shape
    img = img.reshape(gh, gw, H, W, C)
    print(f"{img.shape = }")
    img = img.transpose(0, 2, 1, 3, 4)
    print(f"{img.shape = }")
    img = img.reshape(gh * H, gw * W, C)
    print(f"{img.shape = }")

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()
    attacks = ["clean",
               "poisoning_simple_replacement-High_Cheekbones-Male",
               "poisoning_simple_replacement-Mouth_Slightly_Open-Wearing_Lipstick"]
    base = os.path.join(BASE, "results", "example_images", "CelebA", "StyleGAN")

    for attack in attacks:
        data_file_path = os.path.join(base, f"{attack}_6x6_selected_images.npz")
        if not os.path.exists(data_file_path):
            logger.log("creating model and diffusion...")
            model, diffusion = create_model_and_diffusion(
                **args_to_dict(args, model_and_diffusion_defaults().keys())
            )
            model_path = args.model_path
            model.load_state_dict(
                dist_util.load_state_dict(model_path, map_location="cpu")
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
                sample = sample_fn(
                    model,
                    (args.batch_size, 3, args.image_size, args.image_size),
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
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
                logger.log(f"created {len(all_images) * args.batch_size} samples")

            arr = np.concatenate(all_images, axis=0)
            arr = arr[: args.num_samples]
            if args.class_cond:
                label_arr = np.concatenate(all_labels, axis=0)
                label_arr = label_arr[: args.num_samples]
            if dist.get_rank() == 0:
                shape_str = "x".join([str(x) for x in arr.shape])
                out_path = os.path.join(args.save_path, f"samples_{shape_str}.npz")
                logger.log(f"saving to {out_path}")
                if args.class_cond:
                    np.savez(out_path, arr, label_arr)
                else:
                    np.savez(out_path, arr)
        else:
            arr = np.load(data_file_path)["arr_0"]

        print(arr.shape)

        rnd = np.random.RandomState(args.seed)
        num_per_width = 6
        num_per_height = 6
        image_width = 64
        image_height = 64
        gw = np.clip(7680 // image_width, 7, num_per_width)
        gh = np.clip(4320 // image_height, 4, num_per_height)
        # No labels => show random subset of training samples.
        all_indices = list(range(len(arr)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]
        images = [arr[i] for i in grid_indices]
        print(len(images))
        grid_path = os.path.join(base, f"{attack}_grid_6x6.png")
        save_image_grid(images, grid_path, drange=[0, 255], grid_size=(gw, gh))

        dist.barrier()
        logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        save_path="",
        seed=999,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()