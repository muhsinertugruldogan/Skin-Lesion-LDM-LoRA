#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import PIL
import safetensors
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version

import medmnist
from medmnist import INFO

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }

check_min_version("0.27.0.dev0")
logger = get_logger(__name__)

# ------------------------------------------------------------------------------
# PROMPTS
# ------------------------------------------------------------------------------

skin_disease_prompts = [
    "a dermoscopic image of {}",
    "a dermoscopic photo showing {} on human skin",
    # "a magnified dermoscopic image of {} under polarized light",
    "a close-up dermoscopic image of {}",
    "a dermoscopic image of {} lesion",
    "a clinical dermoscopic photo of {}",
    "a macro dermoscopic photo of {}",
]

# ------------------------------------------------------------------------------
# DATASET
# ------------------------------------------------------------------------------

class TextualInversionDataset(Dataset):
    def __init__(
        self,
        dataset_name,
        tokenizer,
        condition,
        learnable_property="object",
        size=512,
        repeats=10,
        interpolation="bicubic",
        flip_p=0.5,
        set_name="train",
        placeholder_token="*",
        center_crop=False,
    ):
        info = INFO[dataset_name]
        DataClass = getattr(medmnist, info["python_class"])
        self.base_ds = DataClass(split=set_name, download=True, size=224)
        
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

        cname = condition.lower().strip()
        if cname in alias_map:
            cname = alias_map[cname]
        if cname not in label_dict:
            raise ValueError(f"Unknown condition '{condition}'. Available: {list(alias_map.keys())}")
        
        class_id = label_dict[cname]
        
        self.samples = [(img, lbl) for img, lbl in self.base_ds if int(lbl) == class_id]
        
        if len(self.samples) == 0:
            raise ValueError(f"No samples found for condition '{condition}' in dataset '{dataset_name}'.")
            
        print(f"[INFO] Loaded {len(self.samples)} samples for condition '{condition}'.")

        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p
        
        self.num_images = len(self.samples)
        self._length = self.num_images * repeats if set_name == "train" else self.num_images

        self.interpolation = PIL_INTERPOLATION[interpolation]
        self.templates = skin_disease_prompts
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image, _ = self.samples[i % self.num_images]

        if not image.mode == "RGB":
            image = image.convert("RGB")

        text = random.choice(self.templates).format(self.placeholder_token)
        example["input_ids"] = self.tokenizer(
            text, padding="max_length", truncation=True, max_length=self.tokenizer.model_max_length, return_tensors="pt"
        ).input_ids[0]

        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)
        image = self.flip_transform(image)
        
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example

# ------------------------------------------------------------------------------
# SAVE & VALIDATION
# ------------------------------------------------------------------------------

def log_validation(text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, global_step):
    logger.info(f"Running validation at step {global_step}...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=unet,
        vae=vae,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    
    samples_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    autocast_device = "cuda" if accelerator.device.type == "cuda" else "cpu"
    
    val_prompts = [
        f"a dermoscopic image of {args.placeholder_token}",
        f"a dermoscopic photo of {args.placeholder_token}",
        f"a close-up dermoscopic image of {args.placeholder_token}",
        f"a clinical dermoscopic image of {args.placeholder_token}",
        f"a magnified dermoscopic image of {args.placeholder_token}",
        f"a dermoscopic capture of {args.placeholder_token}",
    ]

    for idx in range(args.num_validation_images):
        prompt = val_prompts[idx % len(val_prompts)]
        with torch.autocast(autocast_device, enabled=(accelerator.device.type == "cuda")):
            image = pipeline(prompt, num_inference_steps=25, generator=generator).images[0]
        
        save_path = os.path.join(samples_dir, f"val_step{global_step:05d}_{idx}.png")
        image.save(save_path)

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def save_progress(text_encoder, placeholder_token_ids, accelerator, args, save_path, safe_serialization=True):
    logger.info(f"Saving embeddings to {save_path}")
    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
    )
    learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().cpu()}

    if safe_serialization:
        safetensors.torch.save_file(learned_embeds_dict, save_path, metadata={"format": "pt"})
    else:
        torch.save(learned_embeds_dict, save_path)

# ------------------------------------------------------------------------------
# PARSER
# ------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Textual inversion training script adapted for MedMNIST.")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_as_full_pipeline", action="store_true")
    parser.add_argument("--num_vectors", type=int, default=1)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None, required=True)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    
    parser.add_argument("--dataset_name", type=str, default="dermamnist", help="MedMNIST dataset name")
    parser.add_argument("--class_name", type=str, required=True, help="Target class alias (e.g. mel, nv)")
    
    parser.add_argument("--placeholder_token", type=str, default=None, required=True)
    parser.add_argument("--initializer_token", type=str, default=None, required=True)
    parser.add_argument("--learnable_property", type=str, default="object")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="text-inversion-model")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=5000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--scale_lr", action="store_true")
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--num_validation_images", type=int, default=4)
    parser.add_argument("--validation_steps", type=int, default=100)
    parser.add_argument("--validation_epochs", type=int, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--no_safe_serialization", action="store_true")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant)

    placeholder_tokens = [args.placeholder_token]
    if args.num_vectors < 1:
        raise ValueError(f"--num_vectors has to be larger or equal to 1, but is {args.num_vectors}")

    additional_tokens = [f"{args.placeholder_token}_{i}" for i in range(1, args.num_vectors)]
    placeholder_tokens += additional_tokens

    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    if num_added_tokens != args.num_vectors:
        raise ValueError(f"The tokenizer already contains the token {args.placeholder_token}.")

    token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    initializer_token_id = token_ids[0]
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        for token_id in placeholder_token_ids:
            token_embeds[token_id] = token_embeds[initializer_token_id].clone()

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.train()
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning("xFormers 0.0.16 cannot be used for training in some GPUs.")
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available.")

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes

    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = TextualInversionDataset(
        dataset_name=args.dataset_name,
        tokenizer=tokenizer,
        condition=args.class_name,
        size=args.resolution,
        placeholder_token=(" ".join(tokenizer.convert_ids_to_tokens(placeholder_token_ids))),
        repeats=args.repeats,
        learnable_property=args.learnable_property,
        center_crop=args.center_crop,
        set_name="train",
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )

    if args.validation_epochs is not None:
        args.validation_steps = args.validation_epochs * len(train_dataset)

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
    )

    text_encoder.train()
    text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    embed_weight = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight
    orig_embeds_params = embed_weight.data.clone()
    index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool, device=embed_weight.device)
    index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False

    for epoch in range(first_epoch, args.num_train_epochs):
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(text_encoder):
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = text_encoder(batch["input_ids"])[0].to(dtype=weight_dtype)
                
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                accelerator.backward(loss)

                grads = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.grad
                if grads is not None:
                    grads[index_no_updates] = 0

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                
                with torch.no_grad():
                    accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                        index_no_updates
                    ] = orig_embeds_params[index_no_updates]
                
                progress_bar.update(1)
                global_step += 1
                
                if global_step % args.save_steps == 0:
                    weight_name = (
                        f"learned_embeds-steps-{global_step}.bin"
                        if args.no_safe_serialization
                        else f"learned_embeds-steps-{global_step}.safetensors"
                    )
                    save_path = os.path.join(args.output_dir, weight_name)
                    save_progress(
                        text_encoder,
                        placeholder_token_ids,
                        accelerator,
                        args,
                        save_path,
                        safe_serialization=not args.no_safe_serialization,
                    )

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]
                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)

                    if args.num_validation_images > 0 and global_step % args.validation_steps == 0:
                        log_validation(
                            text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, global_step
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
                
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.save_as_full_pipeline:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=accelerator.unwrap_model(text_encoder),
                vae=vae,
                unet=unet,
                tokenizer=tokenizer,
            )
            pipeline.save_pretrained(args.output_dir)
            
        weight_name = "learned_embeds.bin" if args.no_safe_serialization else "learned_embeds.safetensors"
        save_path = os.path.join(args.output_dir, weight_name)
        save_progress(
            text_encoder,
            placeholder_token_ids,
            accelerator,
            args,
            save_path,
            safe_serialization=not args.no_safe_serialization,
        )

    accelerator.end_training()

if __name__ == "__main__":
    main()