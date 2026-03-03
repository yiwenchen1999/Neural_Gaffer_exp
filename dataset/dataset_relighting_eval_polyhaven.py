import os
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys


class Relighting_Data_Polyhaven(Dataset):
    """
    Dataset for polyhaven-format relighting evaluation data.

    Expected directory layout under `data_root`:
        {obj}_{env_tag}/
            input_images/{view_id}.png      (RGBA, 512x512)
            target_images/{view_id}.png     (RGBA, 512x512)
            envmaps/{view_id}_hdr.png       (RGB,  512x256)
            envmaps/{view_id}_ldr.png       (RGB,  512x256)

    Each (subdirectory, view_id) pair becomes one sample.
    """

    def __init__(
        self,
        data_root,
        image_transforms=None,
        bg_color=(1.0, 1.0, 1.0),
    ):
        super().__init__()
        self.data_root = data_root
        self.tform = image_transforms
        self.bg_color = bg_color

        self.samples = []  # list of (subdir_name, view_id)

        for subdir in sorted(os.listdir(data_root)):
            input_dir = os.path.join(data_root, subdir, "input_images")
            if not os.path.isdir(input_dir):
                continue
            view_ids = sorted(
                os.path.splitext(f)[0]
                for f in os.listdir(input_dir)
                if f.endswith(".png") or f.endswith(".jpg")
            )
            for vid in view_ids:
                hdr_path = os.path.join(data_root, subdir, "envmaps", f"{vid}_hdr.png")
                ldr_path = os.path.join(data_root, subdir, "envmaps", f"{vid}_ldr.png")
                if os.path.exists(hdr_path) and os.path.exists(ldr_path):
                    self.samples.append((subdir, vid))

        print(f"[Polyhaven Dataset] Found {len(self.samples)} samples "
              f"across {len(set(s[0] for s in self.samples))} subdirectories")

    def __len__(self):
        return len(self.samples)

    def _load_rgba_with_bg(self, path):
        """Load RGBA image and alpha-blend onto self.bg_color background."""
        img = np.array(Image.open(path).convert("RGBA")).astype(np.float32) / 255.0
        alpha = img[:, :, 3:4]
        rgb = img[:, :, :3]
        bg = np.array(self.bg_color, dtype=np.float32).reshape(1, 1, 3)
        composited = rgb * alpha + bg * (1.0 - alpha)
        return Image.fromarray(np.uint8(composited * 255.0))

    def _load_rgb(self, path):
        img = Image.open(path).convert("RGB")
        return img

    def _process(self, img):
        img = img.convert("RGB")
        if self.tform is not None:
            return self.tform(img)
        return transforms.ToTensor()(img)

    def __getitem__(self, index):
        subdir, view_id = self.samples[index]
        base = os.path.join(self.data_root, subdir)

        input_path = os.path.join(base, "input_images", f"{view_id}.png")
        target_path = os.path.join(base, "target_images", f"{view_id}.png")
        hdr_path = os.path.join(base, "envmaps", f"{view_id}_hdr.png")
        ldr_path = os.path.join(base, "envmaps", f"{view_id}_ldr.png")

        cond_im = self._process(self._load_rgba_with_bg(input_path))
        target_im = self._process(self._load_rgba_with_bg(target_path))
        envmap_hdr = self._process(self._load_rgb(hdr_path))
        envmap_ldr = self._process(self._load_rgb(ldr_path))

        data = {
            "image_cond": cond_im,
            "image_target": target_im,
            "envir_map_target_hdr": envmap_hdr,
            "envir_map_target_ldr": envmap_ldr,
            "cond_img_name": f"{subdir}",
            "target_envir_map_name": view_id,
            "target_view_idx": int(view_id),
        }
        return data


if __name__ == "__main__":
    image_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    ds = Relighting_Data_Polyhaven(
        data_root="./source_data_polyhaven",
        image_transforms=image_transforms,
    )
    print(f"Dataset length: {len(ds)}")
    sample = ds[0]
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}, range=[{v.min():.2f}, {v.max():.2f}]")
        else:
            print(f"  {k}: {v}")
