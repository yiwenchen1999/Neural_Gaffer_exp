"""
Relight a single input image with 25 randomly chosen environment maps.

Usage:
    accelerate launch --main_process_port 25539 \
        --config_file configs/1_16fp.yaml \
        neural_gaffer_relight_single.py \
        --output_dir neural_gaffer_res256 \
        --mixed_precision fp16 \
        --resume_from_checkpoint latest \
        --polyhaven_data_root /home/ubuntu/LVSMExp/source_data_polyhaven \
        --input_image /data/polyhaven_lvsm/test/images/dining_chair_02_env_0/00063.png \
        --save_dir ./single_relight_results \
        --num_relights 25
"""

import os
import glob
import numpy as np
import torch
import torchvision
from torchvision import transforms
from pathlib import Path
from PIL import Image

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from transformers import CLIPVisionModelWithProjection

from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from pipeline_neural_gaffer import Neural_Gaffer_StableDiffusionPipeline
from parse_args import parse_args


def load_rgba_with_bg(path, bg=(1.0, 1.0, 1.0)):
    img = Image.open(path).convert("RGBA")
    arr = np.array(img, dtype=np.float32) / 255.0
    alpha = arr[:, :, 3:4]
    rgb = arr[:, :, :3]
    bg_arr = np.array(bg, dtype=np.float32).reshape(1, 1, 3)
    composited = rgb * alpha + bg_arr * (1.0 - alpha)
    return Image.fromarray(np.uint8(composited * 255.0))


def collect_envmaps(data_root):
    """Collect all (hdr, ldr) envmap pairs from the polyhaven dataset, excluding white_env."""
    pairs = []
    for subdir in sorted(os.listdir(data_root)):
        if "_white_env_" in subdir:
            continue
        envmap_dir = os.path.join(data_root, subdir, "envmaps")
        if not os.path.isdir(envmap_dir):
            continue
        hdr_files = sorted(glob.glob(os.path.join(envmap_dir, "*_hdr.png")))
        for hdr in hdr_files:
            ldr = hdr.replace("_hdr.png", "_ldr.png")
            if os.path.exists(ldr):
                view_id = os.path.basename(hdr).replace("_hdr.png", "")
                pairs.append(dict(
                    hdr=hdr, ldr=ldr, subdir=subdir, view=view_id
                ))
    return pairs


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(
            project_dir=args.output_dir, logging_dir=logging_dir
        ),
    )
    device = accelerator.device

    if args.seed is not None:
        set_seed(args.seed)

    # ---- load model ----
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision
    )
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
        kernel_size=unet.conv_in.kernel_size, padding=unet.conv_in.padding,
    )
    conv_in_16.requires_grad_(False)
    unet.conv_in.requires_grad_(False)
    torch.nn.init.zeros_(conv_in_16.weight)
    conv_in_16.weight[:, :8, :, :].copy_(unet.conv_in.weight)
    conv_in_16.bias.copy_(unet.conv_in.bias)
    unet.conv_in = conv_in_16
    unet.requires_grad_(False)

    unet = accelerator.prepare(unet)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(device, dtype=weight_dtype)
    image_encoder.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)

    # checkpoint
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            ckpt_path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = sorted(
                [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")],
                key=lambda x: int(x.split("-")[1]),
            )
            ckpt_path = dirs[-1] if dirs else None
        if ckpt_path is None:
            accelerator.print("Checkpoint not found!")
            os._exit(1)
        accelerator.print(f"Resuming from checkpoint {ckpt_path}")
        accelerator.load_state(os.path.join(args.output_dir, ckpt_path))
    else:
        print("No checkpoint found. Exiting.")
        os._exit(1)

    print("Checkpoint loaded.")

    # ---- pipeline ----
    scheduler = DDIMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    pipeline = Neural_Gaffer_StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae).eval(),
        image_encoder=accelerator.unwrap_model(image_encoder).eval(),
        feature_extractor=None,
        unet=accelerator.unwrap_model(unet).eval(),
        scheduler=scheduler,
        safety_checker=None,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    # ---- transforms ----
    img_tf = torchvision.transforms.Compose([
        torchvision.transforms.Resize(
            (args.resolution, args.resolution), antialias=True
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # ---- load input image ----
    input_path = args.input_image
    print(f"Input image: {input_path}")

    if Image.open(input_path).mode == "RGBA":
        input_pil = load_rgba_with_bg(input_path)
    else:
        input_pil = Image.open(input_path).convert("RGB")

    input_tensor = img_tf(input_pil).unsqueeze(0).to(device, dtype=weight_dtype)

    # ---- collect and sample envmaps ----
    all_envmaps = collect_envmaps(args.polyhaven_data_root)
    print(f"Found {len(all_envmaps)} envmap pairs in dataset.")

    rng = np.random.RandomState(args.seed)
    num = args.num_relights
    chosen_idx = rng.choice(len(all_envmaps), size=num, replace=False)
    chosen = [all_envmaps[i] for i in chosen_idx]

    # ---- save dir ----
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    input_pil.resize((args.resolution, args.resolution), Image.LANCZOS).save(
        os.path.join(save_dir, "input.png")
    )

    # ---- relight ----
    for idx, env in enumerate(chosen):
        hdr_t = img_tf(Image.open(env["hdr"]).convert("RGB")) \
            .unsqueeze(0).to(device, dtype=weight_dtype)
        ldr_t = img_tf(Image.open(env["ldr"]).convert("RGB")) \
            .unsqueeze(0).to(device, dtype=weight_dtype)

        gen = [torch.Generator(device=device).manual_seed(args.seed + idx)]
        with torch.autocast("cuda"):
            pil_out = pipeline(
                input_imgs=input_tensor,
                prompt_imgs=input_tensor,
                first_target_envir_map=hdr_t,
                second_target_envir_map=ldr_t,
                poses=None,
                height=args.resolution, width=args.resolution,
                guidance_scale=args.guidance_scale,
                num_inference_steps=50,
                generator=gen,
            ).images[0]

        tag = f"{idx:03d}_{env['subdir']}_{env['view']}"
        pil_out.save(os.path.join(save_dir, f"{tag}.png"))

        # also save the LDR envmap for reference
        ldr_pil = Image.open(env["ldr"]).convert("RGB").resize(
            (args.resolution, args.resolution), Image.LANCZOS
        )
        ldr_pil.save(os.path.join(save_dir, f"{tag}_envmap.png"))

        print(f"[{idx+1}/{num}] {env['subdir']}/{env['view']}  -> saved {tag}.png")

    print(f"\nDone. {num} relit images saved to {save_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
