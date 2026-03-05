"""
Iterative relighting experiment matching the (scene, envmap) pairs from
a reference CSV (e.g. degrade_detail_image_space.csv).

For each (scene_name, envmap_name) pair the SAME envmap is applied
repeatedly for num_chain_steps steps:

    I (from scene_name) --envmap--> I_0 --envmap--> I_1 --envmap--> ...

At each step, metrics are computed against the GT target of envmap_name.

Usage:
    accelerate launch --main_process_port 25539 \
        --config_file configs/1_16fp.yaml \
        neural_gaffer_iterative_relight.py \
        --output_dir neural_gaffer_res256 \
        --mixed_precision fp16 \
        --resume_from_checkpoint latest \
        --polyhaven_data_root /home/ubuntu/LVSMExp/source_data_polyhaven \
        --save_dir ./iterative_relight_results \
        --num_chain_steps 20 \
        --reference_csv degrade_detail_image_space.csv
"""

import os
import csv
from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lpips

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from transformers import CLIPVisionModelWithProjection

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    UNet2DConditionModel,
)
from pipeline_neural_gaffer import Neural_Gaffer_StableDiffusionPipeline
from parse_args import parse_args

logger = get_logger(__name__)


# --------------- metrics ---------------

def compute_psnr(pred, gt):
    mse = np.mean((pred - gt) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(1.0 / np.sqrt(mse))


def _gaussian_kernel_2d(size=11, sigma=1.5, channels=3, device="cpu"):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g1d = g1d / g1d.sum()
    g2d = g1d[:, None] * g1d[None, :]
    return g2d.expand(channels, 1, size, size).contiguous()


def compute_ssim_torch(pred, gt, window_size=11, sigma=1.5):
    """SSIM for NCHW tensors in [0, 1]. Returns shape [N]."""
    C = pred.shape[1]
    kernel = _gaussian_kernel_2d(window_size, sigma, C, pred.device)
    pad = window_size // 2
    mu1 = F.conv2d(pred, kernel, padding=pad, groups=C)
    mu2 = F.conv2d(gt, kernel, padding=pad, groups=C)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1_sq = F.conv2d(pred * pred, kernel, padding=pad, groups=C) - mu1_sq
    s2_sq = F.conv2d(gt * gt, kernel, padding=pad, groups=C) - mu2_sq
    s12 = F.conv2d(pred * gt, kernel, padding=pad, groups=C) - mu12
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu12 + C1) * (2 * s12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (s1_sq + s2_sq + C2))
    return ssim_map.mean(dim=[1, 2, 3])


# --------------- data helpers ---------------

def load_rgba_with_bg(path, bg=(1.0, 1.0, 1.0)):
    img = np.array(Image.open(path).convert("RGBA"), dtype=np.float32) / 255.0
    alpha = img[:, :, 3:4]
    rgb = img[:, :, :3]
    bg_arr = np.array(bg, dtype=np.float32).reshape(1, 1, 3)
    composited = rgb * alpha + bg_arr * (1.0 - alpha)
    return Image.fromarray(np.uint8(composited * 255.0))


def parse_pairs_from_csv(csv_path):
    """Extract unique (scene_name, envmap_name) pairs from reference CSV."""
    pairs = OrderedDict()
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["scene_name"], row["envmap_name"])
            if key not in pairs:
                pairs[key] = int(row["step"]) + 1  # track max steps
            else:
                pairs[key] = max(pairs[key], int(row["step"]) + 1)
    return pairs


def find_view_id(data_root, subdir_name):
    """Find the first available view ID in a subdirectory."""
    input_dir = os.path.join(data_root, subdir_name, "input_images")
    if not os.path.isdir(input_dir):
        return None
    view_ids = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(input_dir)
        if f.endswith(".png") or f.endswith(".jpg")
    )
    return view_ids[0] if view_ids else None


# --------------- main ---------------

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
        feature_extractor=feature_extractor,
        unet=accelerator.unwrap_model(unet).eval(),
        scheduler=scheduler,
        safety_checker=None,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()

    # ---- transforms ----
    img_tf = torchvision.transforms.Compose([
        torchvision.transforms.Resize(
            (args.resolution, args.resolution), antialias=True
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # ---- parse reference CSV for (scene, envmap) pairs ----
    data_root = args.polyhaven_data_root
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    num_steps = args.num_chain_steps

    pairs = parse_pairs_from_csv(args.reference_csv)
    print(f"Loaded {len(pairs)} (scene, envmap) pairs from {args.reference_csv}")

    all_rows = []
    pair_idx = 0

    for (scene_name, envmap_name), csv_steps in pairs.items():
        pair_idx += 1
        steps_to_run = num_steps

        # find a common view_id between scene and envmap subdirs
        view_id = find_view_id(data_root, scene_name)
        if view_id is None:
            print(f"  SKIP {scene_name}: no input images found")
            continue

        # build paths
        inp_path = os.path.join(data_root, scene_name, "input_images", f"{view_id}.png")
        hdr_path = os.path.join(data_root, envmap_name, "envmaps", f"{view_id}_hdr.png")
        ldr_path = os.path.join(data_root, envmap_name, "envmaps", f"{view_id}_ldr.png")
        gt_path  = os.path.join(data_root, envmap_name, "target_images", f"{view_id}.png")

        missing = [p for p in [inp_path, hdr_path, ldr_path, gt_path] if not os.path.exists(p)]
        if missing:
            print(f"  SKIP {scene_name} / {envmap_name}: missing {missing}")
            continue

        # load input
        init_img = load_rgba_with_bg(inp_path)
        current = img_tf(init_img).unsqueeze(0).to(device, dtype=weight_dtype)

        # load envmap (same for every step)
        hdr_t = img_tf(Image.open(hdr_path).convert("RGB")) \
            .unsqueeze(0).to(device, dtype=weight_dtype)
        ldr_t = img_tf(Image.open(ldr_path).convert("RGB")) \
            .unsqueeze(0).to(device, dtype=weight_dtype)

        # load GT (same for every step)
        gt_img = load_rgba_with_bg(gt_path)
        gt_np = np.array(gt_img.resize(
            (args.resolution, args.resolution), Image.LANCZOS
        ), dtype=np.float32) / 255.0
        gt_01 = torch.from_numpy(gt_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
        gt_11 = gt_01 * 2.0 - 1.0

        pair_save = os.path.join(save_dir, f"{scene_name}__{envmap_name}")
        os.makedirs(pair_save, exist_ok=True)

        print(f"[{pair_idx}/{len(pairs)}] scene={scene_name}  envmap={envmap_name}  "
              f"view={view_id}  steps={steps_to_run}")

        for step in range(steps_to_run):
            gen = [torch.Generator(device=device).manual_seed(args.seed)]
            with torch.autocast("cuda"):
                pil_out = pipeline(
                    input_imgs=current,
                    prompt_imgs=current,
                    first_target_envir_map=hdr_t,
                    second_target_envir_map=ldr_t,
                    poses=None,
                    height=args.resolution, width=args.resolution,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=50,
                    generator=gen,
                ).images[0]

            # feed output back
            current = img_tf(pil_out.convert("RGB")) \
                .unsqueeze(0).to(device, dtype=weight_dtype)

            # metrics against GT
            pred_np = np.array(pil_out, dtype=np.float32) / 255.0
            psnr_val = compute_psnr(pred_np, gt_np)

            pred_01 = torch.from_numpy(pred_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
            with torch.no_grad():
                ssim_val = compute_ssim_torch(pred_01, gt_01).item()
                lpips_val = lpips_fn(pred_01 * 2 - 1, gt_11).item()

            all_rows.append(dict(
                scene_name=scene_name, step=step,
                psnr=psnr_val, ssim=ssim_val, lpips=lpips_val,
                envmap_name=envmap_name,
            ))

            pil_out.save(os.path.join(pair_save, f"step_{step:03d}.png"))

            print(f"  step {step:3d}  "
                  f"PSNR={psnr_val:6.2f}  SSIM={ssim_val:.4f}  LPIPS={lpips_val:.4f}")

    # ---- save CSV (same column order as reference) ----
    csv_path = os.path.join(save_dir, "iterative_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["scene_name", "step", "psnr", "ssim", "lpips", "envmap_name"]
        )
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nCSV saved to {csv_path}")

    # ---- aggregate & plot ----
    steps = sorted(set(r["step"] for r in all_rows))
    mean_psnr  = [np.mean([r["psnr"]  for r in all_rows if r["step"] == s]) for s in steps]
    mean_ssim  = [np.mean([r["ssim"]  for r in all_rows if r["step"] == s]) for s in steps]
    mean_lpips = [np.mean([r["lpips"] for r in all_rows if r["step"] == s]) for s in steps]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(steps, mean_psnr, "o-", markersize=3, linewidth=1.2)
    axes[0].set_xlabel("Chain Step")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("PSNR vs. Iterative Relighting Step")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, mean_ssim, "s-", markersize=3, linewidth=1.2, color="tab:orange")
    axes[1].set_xlabel("Chain Step")
    axes[1].set_ylabel("SSIM")
    axes[1].set_title("SSIM vs. Iterative Relighting Step")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, mean_lpips, "^-", markersize=3, linewidth=1.2, color="tab:red")
    axes[2].set_xlabel("Chain Step")
    axes[2].set_ylabel("LPIPS")
    axes[2].set_title("LPIPS vs. Iterative Relighting Step")
    axes[2].grid(True, alpha=0.3)

    n_pairs = len(pairs)
    fig.suptitle(
        f"Iterative Relighting Degradation ({n_pairs} pairs, {num_steps} steps)",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()
    plot_path = os.path.join(save_dir, "iterative_metrics.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")

    print(f"\n{'Step':>5s}  {'PSNR':>8s}  {'SSIM':>8s}  {'LPIPS':>8s}")
    print("-" * 35)
    for s, p, ss, lp in zip(steps, mean_psnr, mean_ssim, mean_lpips):
        print(f"{s:5d}  {p:8.4f}  {ss:8.4f}  {lp:8.4f}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
