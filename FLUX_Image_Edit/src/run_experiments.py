#!/usr/bin/env python3
"""
Batch runner for RF‑Solver‑Edit (a.k.a. RF‑Edit).

Reads a dataset JSON (image_path / source_prompt / target_prompt),
applies the RF‑Edit pipeline, records metrics, and writes a CSV.

The core editing code is a verbatim copy of the Gradio demo, so
re‑running the demo and this script with identical seeds produces
bit‑for‑bit identical outputs.
"""

import argparse
import csv
import json
import os
import re
import time
from glob import iglob
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torchvision.transforms as T
from einops import rearrange
from PIL import Image, ImageChops, ExifTags
from transformers import CLIPModel, CLIPProcessor

# RF‑Solver / FLUX imports (same as demo)
from flux.sampling import denoise, get_schedule, prepare, unpack
from flux.util import (
    configs,
    embed_watermark,
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
)

import lpips

# ------------------------------------------------------------
# 1.  Metric helpers (with channel‑averaged pixel distances)
# ------------------------------------------------------------
class MetricComputer:
    def __init__(self, device: torch.device):
        self.device = device
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.lpips_fn = lpips.LPIPS(net="vgg").to(device)
        self.lpips_transform = T.Compose(
            [T.Resize((1024, 1024)), T.ToTensor(), T.Normalize((0.5,), (0.5,))]
        )

    @torch.inference_mode()
    def clip_similarity(self, image: Image.Image, text: str) -> float:
        inp = self.clip_processor(text=[text], images=image, return_tensors="pt", padding=True
                                  ).to(self.clip_model.device)
        return self.clip_model(**inp).logits_per_image.item()

    @torch.inference_mode()
    def lpips_distance(self, img1: Image.Image, img2: Image.Image) -> float:
        t1 = self.lpips_transform(img1).unsqueeze(0).to(self.device)
        t2 = self.lpips_transform(img2).unsqueeze(0).to(self.device)
        return self.lpips_fn(t1, t2).item()

    # --- updated per‑pixel stats (RGB averaged) ------------
    @staticmethod
    def pixel_distances(img1: Image.Image, img2: Image.Image) -> Tuple[float, float, float, float]:
        a1, a2 = np.asarray(img1, np.float32), np.asarray(img2, np.float32)
        diff   = a1 - a2
        l1_map = np.mean(np.abs(diff), axis=-1)        # (H, W)
        l2_map = np.mean(diff ** 2,  axis=-1)          # (H, W)
        return l1_map.mean(), np.median(l1_map), l2_map.mean(), np.median(l2_map)


# ------------------------------------------------------------
# 2.  RF‑Edit (FluxEditor)  –  copied “as‑is”
# ------------------------------------------------------------

@dataclass
class SamplingOptions:
    source_prompt: str
    target_prompt: str
    # prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None

@torch.inference_mode()
def encode(init_image, torch_device, ae):
    init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1
    init_image = init_image.unsqueeze(0) 
    init_image = init_image.to(torch_device)
    with torch.no_grad():
        init_image = ae.encode(init_image.to()).to(torch.bfloat16)
    return init_image

class FluxEditor:
    def __init__(self, args):
        self.device  = torch.device(args.device)
        self.offload = args.offload
        self.name    = args.name
        self.is_schnell = self.name == "flux-schnell"

        self.feature_path = "feature"
        self.output_dir   = "result"

        if self.name not in configs:
            raise ValueError(f"Unknown model name {self.name}; choose from {list(configs.keys())}")

        # init components (identical to demo)
        self.t5    = load_t5(self.device, max_length=256 if self.is_schnell else 512)
        self.clip  = load_clip(self.device)
        self.model = load_flow_model(self.name, device="cpu")
        self.ae    = load_ae(self.name, device="cpu")
        for m in (self.t5, self.clip, self.model, self.ae):
            m.eval()

        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.encoder.to(self.device)

    # ------------- helper: encode to latent (demo code) -----
    @torch.inference_mode()
    def _encode(self, np_img):
        img = torch.from_numpy(np_img.copy()).permute(2, 0, 1).float() / 127.5 - 1
        img = img.unsqueeze(0).to(self.device)
        with torch.no_grad():
            lat = self.ae.encode(img).to(torch.bfloat16)
        return lat

    # ------------- main edit function (demo logic) ----------
    @torch.inference_mode()
    def edit(self, init_pil: Image.Image, source_prompt: str, target_prompt: str,
             *, num_steps: int, inject_step: int, guidance: float, seed: int):
        torch.manual_seed(seed)
        
        init_image = np.array(init_pil)
        shape = init_image.shape

        self.t5 = self.t5.to("cuda")
        self.clip = self.clip.to("cuda")
        self.ae = self.ae.to("cuda")
        self.model = self.model.to("cuda")

        new_h = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
        new_w = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16

        init_image = init_image[:new_h, :new_w, :]

        width, height = init_image.shape[0], init_image.shape[1]
        init_image = encode(init_image, self.device, self.ae)

        print(init_image.shape)

        opts = SamplingOptions(
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed,
        )
        if opts.seed is None:
            opts.seed = torch.Generator(device="cpu").seed()
        
        print(f"Generating with seed {opts.seed}:\n{opts.source_prompt}")

        opts.seed = None
        #############inverse#######################
        info = {}
        info['feature'] = {}
        info['inject_step'] = inject_step

        if not os.path.exists(self.feature_path):
            os.mkdir(self.feature_path)

        with torch.no_grad():
            inp = prepare(self.t5, self.clip, init_image, prompt=opts.source_prompt)
            inp_target = prepare(self.t5, self.clip, init_image, prompt=opts.target_prompt)
        timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(self.name != "flux-schnell"))

        # inversion initial noise
        with torch.no_grad():
            z, info = denoise(self.model, **inp, timesteps=timesteps, guidance=1, inverse=True, info=info)
        
        inp_target["img"] = z

        timesteps = get_schedule(opts.num_steps, inp_target["img"].shape[1], shift=(self.name != "flux-schnell"))

        # denoise initial noise
        x, _ = denoise(self.model, **inp_target, timesteps=timesteps, guidance=guidance, inverse=False, info=info)

        # decode latents to pixel space
        x = unpack(x.float(), opts.width, opts.height)

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # bring into PIL format and save
        x = x.clamp(-1, 1)
        x = embed_watermark(x.float())
        x = rearrange(x[0], "c h w -> h w c")

        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        return img


# ------------------------------------------------------------
# 3.  Batch “run_edit” wrapper
# ------------------------------------------------------------
def run_edit(
    editor: FluxEditor,
    metric: MetricComputer,
    image_path: Path,
    source_prompt: str,
    target_prompt: str,
    *,
    save_dir: Path,
    idx: int,
    num_steps: int,
    inject_step: int,
    guidance: float,
    seed: int,
) -> Dict[str, float]:
    init_pil = Image.open(image_path).convert("RGB")

    edited = editor.edit(
        init_pil,
        source_prompt,
        target_prompt,
        num_steps=num_steps,
        inject_step=inject_step,
        guidance=guidance,
        seed=seed,
    )

    # ---- save images ----
    stem, ext = image_path.stem, image_path.suffix or ".png"
    save_dir.mkdir(parents=True, exist_ok=True)
    edit_path = save_dir / f"{idx:04d}_{stem}_edited{ext}"
    diff_path = save_dir / f"{idx:04d}_{stem}_diff{ext}"
    edited.save(edit_path)
    ImageChops.difference(init_pil.resize(edited.size, Image.Resampling.LANCZOS), edited).save(diff_path)

    # ---- metrics -------
    clip_edit = metric.clip_similarity(edited, target_prompt)
    clip_src  = metric.clip_similarity(init_pil, target_prompt)
    lpips_val = metric.lpips_distance(init_pil, edited)
    l1_mean, l1_med, l2_mean, l2_med = metric.pixel_distances(init_pil.resize(edited.size), edited)

    return {
        "clip_target_edit": clip_edit,
        "clip_target_src":  clip_src,
        "lpips":            lpips_val,
        "l1_mean":          l1_mean,
        "l1_median":        l1_med,
        "l2_mean":          l2_mean,
        "l2_median":        l2_med,
        "edited_path":      str(edit_path),
        "diff_path":        str(diff_path),
        "seed":             seed,
    }


# ------------------------------------------------------------
# 4.  CLI & main loop
# ------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser("RF‑Edit Experiment Runner")
    p.add_argument("--data", type=str, default="dataset.json")

    p.add_argument("--name",   type=str, default="flux-dev", choices=list(configs.keys()))
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--offload", action="store_true")

    # hyper‑params (demo defaults)
    p.add_argument("--num_steps",   type=int,   default=25)
    p.add_argument("--inject_step", type=int,   default=5)
    p.add_argument("--guidance",    type=float, default=2.0)

    # misc
    p.add_argument("--save_dir", type=str, default="edited_rfedit")
    p.add_argument("--csv_out",  type=str, default="results_rfedit.csv")
    p.add_argument("--seed",     type=int, default=None, help="Global seed; else random per sample")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    editor = FluxEditor(args)
    metric = MetricComputer(device)

    with open(args.data) as fp:
        dataset: List[Dict] = json.load(fp)

    rows = []
    for idx, entry in enumerate(dataset):
        print(f"\n=== [{idx+1}/{len(dataset)}] {entry['image_path']} ===")
        seed = args.seed if args.seed is not None else int(torch.Generator(device="cpu").seed())

        stats = run_edit(
            editor,
            metric,
            Path(entry["image_path"]),
            entry["source_prompt"],
            entry["target_prompt"],
            save_dir=Path(args.save_dir),
            idx=idx,
            num_steps=args.num_steps,
            inject_step=args.inject_step,
            guidance=args.guidance,
            seed=seed,
        )
        for k, v in stats.items():
            print(f"{k:>20}: {v}")
        rows.append({**entry, **stats})

    out = Path(args.csv_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nFinished — detailed metrics stored in {out.resolve()}")


if __name__ == "__main__":
    main()
