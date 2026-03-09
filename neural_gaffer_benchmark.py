"""
Benchmark Neural Gaffer inference: measure end-to-end wall time,
per-sample latency, and peak GPU/CPU memory for 100 samples.

Usage:
    accelerate launch --main_process_port 25539 \
        --config_file configs/1_16fp.yaml \
        neural_gaffer_benchmark.py \
        --output_dir neural_gaffer_res256 \
        --mixed_precision fp16 \
        --resume_from_checkpoint latest \
        --polyhaven_data_root /home/ubuntu/LVSMExp/source_data_polyhaven \
        --save_dir ./benchmark_results \
        --benchmark_samples 100
"""

import os
import time
import json
import resource
from pathlib import Path
import numpy as np
import torch
import torch.utils.checkpoint
import torchvision
from torchvision import transforms
from PIL import Image

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from transformers import CLIPVisionModelWithProjection

from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from pipeline_neural_gaffer import Neural_Gaffer_StableDiffusionPipeline

from dataset.dataset_relighting_eval_polyhaven import Relighting_Data_Polyhaven
from parse_args import parse_args


def get_gpu_memory_mb():
    if not torch.cuda.is_available():
        return 0.0, 0.0
    allocated = torch.cuda.max_memory_allocated() / 1024 ** 2
    reserved = torch.cuda.max_memory_reserved() / 1024 ** 2
    return allocated, reserved


def get_cpu_rss_mb():
    """Peak RSS of the current process in MB (Linux/macOS)."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    if hasattr(os, 'uname') and os.uname().sysname == 'Darwin':
        return ru.ru_maxrss / 1024 ** 2  # macOS returns bytes
    return ru.ru_maxrss / 1024  # Linux returns KB


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

    # ---- dataset ----
    image_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(
            (args.resolution, args.resolution), antialias=True
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    dataset = Relighting_Data_Polyhaven(
        data_root=args.polyhaven_data_root,
        image_transforms=image_transforms,
    )

    num_samples = min(args.benchmark_samples, len(dataset))
    subset = torch.utils.data.Subset(dataset, list(range(num_samples)))

    dataloader = torch.utils.data.DataLoader(
        subset,
        shuffle=False,
        batch_size=100,
        num_workers=1,
        pin_memory=True,
    )

    print(f"\n{'='*60}")
    print(f"  Benchmarking {num_samples} samples  (resolution={args.resolution})")
    print(f"  Mixed precision: {args.mixed_precision}")
    print(f"  Guidance scale: {args.guidance_scale}")
    print(f"  Diffusion steps: 50")
    print(f"{'='*60}\n")

    # ---- reset GPU peak counters ----
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # ---- warmup (not timed) ----
    print("Warmup run...")
    warmup_batch = next(iter(dataloader))
    warmup_bsz = warmup_batch["image_cond"].shape[0]
    with torch.autocast("cuda"), torch.no_grad():
        pipeline(
            input_imgs=warmup_batch["image_cond"].to(dtype=weight_dtype),
            prompt_imgs=warmup_batch["image_cond"].to(dtype=weight_dtype),
            first_target_envir_map=warmup_batch["envir_map_target_hdr"].to(dtype=weight_dtype),
            second_target_envir_map=warmup_batch["envir_map_target_ldr"].to(dtype=weight_dtype),
            poses=None,
            height=args.resolution, width=args.resolution,
            guidance_scale=args.guidance_scale,
            num_inference_steps=50,
            generator=[torch.Generator(device=device).manual_seed(args.seed + i) for i in range(warmup_bsz)],
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    print("Warmup done.\n")

    # ---- timed benchmark ----
    per_sample_times = []
    total_images = 0

    t_total_start = time.perf_counter()

    for batch_idx, batch in enumerate(dataloader):
        if total_images >= num_samples:
            break

        input_image = batch["image_cond"].to(dtype=weight_dtype)
        envmap_hdr = batch["envir_map_target_hdr"].to(dtype=weight_dtype)
        envmap_ldr = batch["envir_map_target_ldr"].to(dtype=weight_dtype)
        batchsize = input_image.shape[0]

        generator_list = [
            torch.Generator(device=device).manual_seed(args.seed)
            for _ in range(batchsize)
        ]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.autocast("cuda"), torch.no_grad():
            pipeline(
                input_imgs=input_image,
                prompt_imgs=input_image,
                first_target_envir_map=envmap_hdr,
                second_target_envir_map=envmap_ldr,
                poses=None,
                height=args.resolution, width=args.resolution,
                guidance_scale=args.guidance_scale,
                num_inference_steps=50,
                generator=generator_list,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        elapsed = t1 - t0
        for _ in range(batchsize):
            per_sample_times.append(elapsed / batchsize)
        total_images += batchsize

        if (batch_idx + 1) % 10 == 0:
            print(f"  [{total_images}/{num_samples}]  "
                  f"last batch: {elapsed:.3f}s  "
                  f"avg: {np.mean(per_sample_times):.3f}s/sample")

    t_total_end = time.perf_counter()
    total_wall_time = t_total_end - t_total_start

    # ---- collect resource stats ----
    gpu_peak_alloc_mb, gpu_peak_reserved_mb = get_gpu_memory_mb()
    cpu_peak_rss_mb = get_cpu_rss_mb()

    gpu_name = "N/A"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    times = np.array(per_sample_times[:num_samples])

    results = {
        "num_samples": int(num_samples),
        "resolution": args.resolution,
        "mixed_precision": args.mixed_precision,
        "diffusion_steps": 50,
        "guidance_scale": args.guidance_scale,
        "batch_size": 100,
        "gpu": gpu_name,
        "total_wall_time_s": round(total_wall_time, 3),
        "avg_time_per_sample_s": round(float(times.mean()), 3),
        "median_time_per_sample_s": round(float(np.median(times)), 3),
        "std_time_per_sample_s": round(float(times.std()), 3),
        "min_time_per_sample_s": round(float(times.min()), 3),
        "max_time_per_sample_s": round(float(times.max()), 3),
        "throughput_samples_per_s": round(num_samples / total_wall_time, 3),
        "peak_gpu_allocated_mb": round(gpu_peak_alloc_mb, 1),
        "peak_gpu_reserved_mb": round(gpu_peak_reserved_mb, 1),
        "peak_cpu_rss_mb": round(cpu_peak_rss_mb, 1),
    }

    # ---- print report ----
    print(f"\n{'='*60}")
    print(f"  BENCHMARK RESULTS")
    print(f"{'='*60}")
    for k, v in results.items():
        print(f"  {k:35s}: {v}")
    print(f"{'='*60}\n")

    # ---- save ----
    json_path = os.path.join(save_dir, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {json_path}")

    txt_path = os.path.join(save_dir, "benchmark_results.txt")
    with open(txt_path, "w") as f:
        f.write("Neural Gaffer Inference Benchmark\n")
        f.write("=" * 50 + "\n")
        for k, v in results.items():
            f.write(f"{k:35s}: {v}\n")
        f.write("\nPer-sample times (seconds):\n")
        for i, t in enumerate(times):
            f.write(f"  sample {i:4d}: {t:.4f}\n")
    print(f"Detailed log saved to {txt_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
