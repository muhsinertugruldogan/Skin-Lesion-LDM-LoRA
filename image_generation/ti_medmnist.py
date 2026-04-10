#!/usr/bin/env python
# coding=utf-8

import os
import math
import random
import shutil
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm
from packaging import version

import transformers
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder

import medmnist
from medmnist import INFO
import safetensors

# ----------------------------------------------------------------------
# GLOBALS
# ----------------------------------------------------------------------

PIL_INTERPOLATION = {
    "linear": Image.Resampling.BILINEAR,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
    "nearest": Image.Resampling.NEAREST,
}

skin_disease_prompt = [
    "a dermoscopic image of a {}",
    "a dermatoscopic photo showing {} on human skin",
    "a magnified image of a {} under polarized light",
    "a close-up dermoscopic image of {}",
    "a dermoscopy image of {} lesion",
    "a clinical dermatoscopic photo of {}",
    "a polarized light photo showing {} on skin surface",
    "a macro dermoscopy photo of {}",
    "a high-resolution dermoscopic image showing {}",
    "a close-up clinical dermoscopy image of {}",
    "a digital dermatoscopic capture of {}",
]



# ----------------------------------------------------------------------
# DATASET
# ----------------------------------------------------------------------

class TextualInversionDataset(Dataset):
    def __init__(
        self,
        dataset_name,
        tokenizer,
        class_name=None,
        size=512,
        repeats=10,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
    ):
        info = INFO[dataset_name]
        DataClass = getattr(medmnist, info["python_class"])
        self.base_ds = DataClass(split=set, download=True, size=224)

        label_dict = {v.lower(): int(k) for k, v in info["label"].items()}

        alias_map = {
            "akiec": "actinic keratoses and intraepithelial carcinoma",
            "bcc": "basal cell carcinoma",
            "bkl": "benign keratosis-like lesions",
            "df": "dermatofibroma",
            "mel": "melanoma",
            "nv": "melanocytic nevi",
            "vasc": "vascular lesions"
        }

        if class_name is not None:
            cname = class_name.lower().strip()
            if cname in alias_map:
                cname = alias_map[cname]
            if cname not in label_dict:
                raise ValueError(f"Unknown class_name '{class_name}'. Available aliases: {list(alias_map.keys())}")
            class_id = label_dict[cname]
            imgs, labels = [], []
            for img, lbl in self.base_ds:
                if int(lbl) == class_id:
                    imgs.append(img)
                    labels.append(lbl)
            self.samples = list(zip(imgs, labels))
            print(f"[INFO] Loaded {len(self.samples)} samples for class '{class_name}' ({cname}) → id={class_id}")
        else:
            self.samples = list(self.base_ds)
            print(f"[INFO] Loaded full dataset ({len(self.samples)} samples, all classes)")


        self.tokenizer = tokenizer
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.num_images = len(self.samples)
        self._length = self.num_images * repeats if set == "train" else self.num_images
        self.interpolation = PIL_INTERPOLATION[interpolation]
        self.templates = skin_disease_prompt
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image, _ = self.samples[i % self.num_images]

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)
        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        img_np = np.array(image).astype(np.uint8)
        if self.center_crop:
            crop = min(img_np.shape[0], img_np.shape[1])
            h, w = img_np.shape
            img_np = img_np[(h - crop)//2:(h + crop)//2, (w - crop)//2:(w + crop)//2]
        image = Image.fromarray(img_np)
        image = image.resize((self.size, self.size), resample=self.interpolation)
        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example


# ----------------------------------------------------------------------
# SAVE / VALIDATION FUNCTIONS
# ----------------------------------------------------------------------

def save_progress(text_encoder, placeholder_token_ids, accelerator, args, save_path, safe_serialization=True):
    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight[min(placeholder_token_ids): max(placeholder_token_ids) + 1]
    )
    learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().cpu()}
    torch.save(learned_embeds_dict, save_path)
    print(f"[INFO] Saved embeddings → {save_path}")


def log_validation(text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, epoch):
    if args.validation_prompt is None:
        return []
    print(f"[INFO] Validation: generating {args.num_validation_images} images for prompt '{args.validation_prompt}'")

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=unet,
        vae=vae,
        safety_checker=None,
        torch_dtype=weight_dtype,
    ).to(accelerator.device)

    samples_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    generator = torch.Generator(device=accelerator.device).manual_seed(0)

    images = []
    for idx in range(args.num_validation_images):
        with torch.autocast("cuda"):
            img = pipeline(args.validation_prompt, num_inference_steps=25, generator=generator).images[0]
        img.save(os.path.join(samples_dir, f"sample_{epoch:03d}_{idx}.png"))
        images.append(img)
    torch.cuda.empty_cache()
    return images


# ----------------------------------------------------------------------
# ARGUMENTS
# ----------------------------------------------------------------------

# def parse_args():
#     parser = argparse.ArgumentParser(description="Textual inversion fine-tuning on MedMNIST datasets.")
#     parser.add_argument("--dataset_name", type=str, required=True, help="e.g., dermamnist, pathmnist")
#     parser.add_argument("--class_name", type=str, default=None, help="Target class name, e.g. 'mel', 'bcc', 'nv'")
#     parser.add_argument("--placeholder_token", type=str, required=True)
#     parser.add_argument("--initializer_token", type=str, default="lesion")
#     parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-2-1-base")
#     parser.add_argument("--output_dir", type=str, default="./ti_outputs")
#     parser.add_argument("--resolution", type=int, default=512)
#     parser.add_argument("--train_batch_size", type=int, default=4)
#     parser.add_argument("--repeats", type=int, default=10)
#     parser.add_argument("--learning_rate", type=float, default=5e-4)
#     parser.add_argument("--max_train_steps", type=int, default=500)
#     parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
#     parser.add_argument("--mixed_precision", type=str, default="fp16")
#     parser.add_argument("--save_steps", type=int, default=500)
#     parser.add_argument("--validation_prompt", type=str, default=None)
#     parser.add_argument("--num_validation_images", type=int, default=2)
#     parser.add_argument("--validation_steps", type=int, default=200)
#     return parser.parse_args()

def parse_args():
    parser = argparse.ArgumentParser(description="Simplified Textual Inversion training for DermMNIST.")
    parser.add_argument("--class_name", type=str, required=True, help="Class alias (e.g. mel, nv, bcc)")
    parser.add_argument("--max_train_steps", type=int, default=3000)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num_validation_images", type=int, default=8)
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    return parser.parse_args()


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(42)

    args.dataset_name = "dermamnist"
    args.placeholder_token = f"<{args.class_name}_lesion>"
    args.initializer_token = "skin"
    args.pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-1-base"
    args.output_dir = f"./outputs/ti_{args.class_name}_lesion"
    args.validation_prompt = f"a dermoscopic photo of {args.placeholder_token} on the skin"
    args.save_steps = 500
    args.validation_steps = 500

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "train_log.txt")

    # --- Accelerator setup ---
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=args.output_dir),
        log_with=None,
    )

    device = accelerator.device
    print(f"[INFO] Training '{args.class_name}' on DermMNIST with token={args.placeholder_token}")

    # --- Model and tokenizer loading ---
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet").to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    vae.requires_grad_(False)
    unet.requires_grad_(False)

    # --- Token setup ---
    placeholder_tokens = [args.placeholder_token]
    tokenizer.add_tokens(placeholder_tokens)
    text_encoder.resize_token_embeddings(len(tokenizer))
    init_id = tokenizer.encode(args.initializer_token, add_special_tokens=False)[0]
    ph_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
    with torch.no_grad():
        for token_id in ph_ids:
            text_encoder.get_input_embeddings().weight[token_id] = (
                text_encoder.get_input_embeddings().weight[init_id].clone()
            )

    # --- Dataset setup ---
    train_dataset = TextualInversionDataset(
        dataset_name=args.dataset_name,
        tokenizer=tokenizer,
        class_name=args.class_name,
        size=args.resolution,
        repeats=args.repeats,
        placeholder_token=(" ".join(tokenizer.convert_ids_to_tokens(ph_ids))),
        set="train",
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)

    optimizer = torch.optim.AdamW(text_encoder.get_input_embeddings().parameters(), lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        "constant", optimizer=optimizer, num_warmup_steps=0, num_training_steps=args.max_train_steps
    )

    text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32
    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)

    # --- Training setup ---
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    global_step, first_epoch = 0, 0
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    print(f"[INFO] Total steps = {args.max_train_steps}, LR = {args.learning_rate}, Batch = {args.train_batch_size}")

    # --- Training loop ---
    for epoch in range(first_epoch, args.num_train_epochs):
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(text_encoder):
                latents = vae.encode(batch["pixel_values"].to(device, dtype=weight_dtype)).latent_dist.sample().detach()
                latents = latents * 0.18215
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = text_encoder(batch["input_ids"])[0].to(dtype=weight_dtype)
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if global_step % 100 == 0 and accelerator.is_main_process:
                    with open(log_path, "a") as f:
                        f.write(f"Step {global_step:05d}: loss={loss.item():.6f}\n")

            progress_bar.update(1)
            global_step += 1

            if global_step % args.save_steps == 0:
                save_progress(
                    text_encoder, ph_ids, accelerator, args,
                    os.path.join(args.output_dir, f"learned_embeds_step{global_step}.pt")
                )
            if args.validation_prompt and global_step % args.validation_steps == 0:
                log_validation(text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, epoch)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    # --- Final save ---
    final_path = os.path.join(args.output_dir, "learned_embeds_final.pt")
    save_progress(text_encoder, ph_ids, accelerator, args, final_path)
    print(f"[DONE] Training complete. Embedding saved at {final_path}")


if __name__ == "__main__":
    main()
