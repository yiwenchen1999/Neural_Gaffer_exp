"""
Inference script for Neural Gaffer on polyhaven-format relighting data.

Usage:
    accelerate launch --main_process_port 25539 \
        --config_file configs/1_16fp.yaml \
        neural_gaffer_inference_polyhaven.py \
        --output_dir logs/neural_gaffer_res256 \
        --mixed_precision fp16 \
        --resume_from_checkpoint latest \
        --polyhaven_data_root ./source_data_polyhaven \
        --save_dir ./polyhaven_relighting_results
"""

import os
from pathlib import Path
import numpy as np
import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm import tqdm

from dataset.dataset_relighting_eval_polyhaven import Relighting_Data_Polyhaven

from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPVisionModelWithProjection

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    UNet2DConditionModel,
)
from pipeline_neural_gaffer import Neural_Gaffer_StableDiffusionPipeline

import torchvision

logger = get_logger(__name__)

from parse_args import parse_args


def compute_psnr(pred, gt):
    mse = np.mean((pred - gt) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(1.0 / np.sqrt(mse))


def log_validation(
    validation_dataloader, vae, image_encoder, feature_extractor, unet,
    args, accelerator, weight_dtype, split="val"
):
    logger.info("Running {} validation... ".format(split))

    scheduler = DDIMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    pipeline = Neural_Gaffer_StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae).eval(),
        image_encoder=accelerator.unwrap_model(image_encoder).eval(),
        feature_extractor=feature_extractor,
        unet=accelerator.unwrap_model(unet).eval(),
        scheduler=scheduler,
        safety_checker=None,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    all_psnr = []

    for valid_step, batch in tqdm(enumerate(validation_dataloader)):
        if args.num_validation_batches is not None and valid_step >= args.num_validation_batches:
            break

        input_image = batch["image_cond"].to(dtype=weight_dtype)
        target_image = batch["image_target"].to(dtype=weight_dtype)
        target_envmap_ldr = batch["envir_map_target_ldr"].to(dtype=weight_dtype)
        target_envmap_hdr = batch["envir_map_target_hdr"].to(dtype=weight_dtype)
        cond_image_names = batch["cond_img_name"]
        target_view_idx = batch["target_view_idx"]
        target_envir_map_names = batch["target_envir_map_name"]

        batchsize, _, h, w = input_image.shape

        generator_list = [
            torch.Generator(device=accelerator.device).manual_seed(args.seed)
            for _ in range(batchsize)
        ]

        with torch.autocast("cuda"):
            pipeline_output_images = pipeline(
                input_imgs=input_image,
                prompt_imgs=input_image,
                first_target_envir_map=target_envmap_hdr,
                second_target_envir_map=target_envmap_ldr,
                poses=None,
                height=h, width=w,
                guidance_scale=args.guidance_scale,
                num_inference_steps=50,
                generator=generator_list,
            ).images

        input_npy = 0.5 * (input_image.permute(0, 2, 3, 1).cpu().float().numpy() + 1.0)
        target_npy = 0.5 * (target_image.permute(0, 2, 3, 1).cpu().float().numpy() + 1.0)
        envmap_ldr_npy = 0.5 * (target_envmap_ldr.permute(0, 2, 3, 1).cpu().float().numpy() + 1.0)

        for i in range(batchsize):
            pred_np = np.array(pipeline_output_images[i], dtype=np.float32) / 255.0
            gt_np = target_npy[i]

            psnr_val = compute_psnr(pred_np, gt_np)
            all_psnr.append(psnr_val)

            name = cond_image_names[i]
            view_name = target_envir_map_names[i]
            view_idx = target_view_idx[i]

            sample_dir = os.path.join(save_dir, name)
            for sub in ["input_image", "target_envmap_ldr", "pred_image", "gt_image"]:
                os.makedirs(os.path.join(sample_dir, sub), exist_ok=True)

            tag = f"{view_name}_{view_idx:05d}"
            Image.fromarray((input_npy[i] * 255).astype(np.uint8)).save(
                os.path.join(sample_dir, "input_image", f"{tag}.png")
            )
            Image.fromarray((envmap_ldr_npy[i] * 255).astype(np.uint8)).save(
                os.path.join(sample_dir, "target_envmap_ldr", f"{tag}.png")
            )
            Image.fromarray((pred_np * 255).astype(np.uint8)).save(
                os.path.join(sample_dir, "pred_image", f"{tag}.png")
            )
            Image.fromarray((gt_np * 255).astype(np.uint8)).save(
                os.path.join(sample_dir, "gt_image", f"{tag}.png")
            )

    if all_psnr:
        mean_psnr = np.mean(all_psnr)
        print(f"\n{'='*50}")
        print(f"  Mean PSNR: {mean_psnr:.4f} dB  ({len(all_psnr)} samples)")
        print(f"{'='*50}\n")

        with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
            f.write(f"mean_psnr: {mean_psnr:.4f}\n")
            f.write(f"num_samples: {len(all_psnr)}\n")
            for idx, p in enumerate(all_psnr):
                f.write(f"sample_{idx}: {p:.4f}\n")

    return True


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.seed is not None:
        set_seed(args.seed)

    # Load models
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision
    )
    feature_extractor = None
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)

    conv_in_16 = torch.nn.Conv2d(
        16, unet.conv_in.out_channels,
        kernel_size=unet.conv_in.kernel_size, padding=unet.conv_in.padding
    )
    conv_in_16.requires_grad_(False)
    unet.conv_in.requires_grad_(False)
    torch.nn.init.zeros_(conv_in_16.weight)
    conv_in_16.weight[:, :8, :, :].copy_(unet.conv_in.weight)
    conv_in_16.bias.copy_(unet.conv_in.bias)
    unet.conv_in = conv_in_16
    unet.requires_grad_(False)

    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"UNet loaded as datatype {accelerator.unwrap_model(unet).dtype}. "
            "Please make sure to always have all model weights in full float32 precision when starting training"
        )

    image_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((args.resolution, args.resolution), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    validation_dataset = Relighting_Data_Polyhaven(
        data_root=args.polyhaven_data_root,
        image_transforms=image_transforms,
    )

    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        shuffle=False,
        batch_size=4,
        num_workers=1,
        pin_memory=True,
    )

    unet = accelerator.prepare(unet)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    # Load checkpoint
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist !!!"
            )
            os._exit(1)
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
    else:
        print("No checkpoint found. Validation Failed")

    print("Loading checkpoint finished!!!!")

    if validation_dataloader is not None:
        log_validation(
            validation_dataloader, vae, image_encoder, feature_extractor, unet,
            args, accelerator, weight_dtype, split="polyhaven",
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
