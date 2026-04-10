#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import math
import os
import random
from datetime import datetime
from collections import Counter

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import Dataset
from torch.utils.data.sampler import WeightedRandomSampler

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers

check_min_version("0.27.0.dev0")
logger = get_logger(__name__, log_level="INFO")

# ------------------------------------------------------------------------------
# PROMPTS & MAPS
# ------------------------------------------------------------------------------

skin_disease_prompts = [
    "a dermoscopic image of {}",
    "a dermoscopic photo of {}",
    "a close-up dermoscopic image of {}",
    "a clinical dermoscopic image of {}",
    "a magnified dermoscopic image of {}",
    "a dermoscopic capture of {}",
]

id_to_alias = {
    0: "akiec",
    1: "bcc",
    2: "bkl",
    3: "df",
    4: "mel",
    5: "nv",
    6: "vasc",
}

# ------------------------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------------------------

def log_validation(text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, epoch):
    n_images_per_class = args.num_validation_images
    total_images = len(id_to_alias) * n_images_per_class
    logger.info(f"Validation: generating {total_images} images ({n_images_per_class} per class)")

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        vae=vae,
        safety_checker=None,
        torch_dtype=weight_dtype,
    ).to(accelerator.device)

    samples_dir = os.path.join(args.output_dir, "samples", f"epoch_{epoch:03d}")
    os.makedirs(samples_dir, exist_ok=True)

    def get_full_token_str(alias: str) -> str:
        base_token = f"<{alias}_lesion>"
        tokens = [base_token]
        for i in range(1, args.num_vectors):
            next_token = f"{base_token}_{i}"
            if next_token in tokenizer.get_vocab():
                tokens.append(next_token)
        return " ".join(tokens)

    autocast_device = "cuda" if accelerator.device.type == "cuda" else "cpu"

    for _, alias in id_to_alias.items():
        class_generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
        full_token_str = get_full_token_str(alias)
        prompt = f"a dermoscopic photo of {full_token_str}"
        logger.info(f"Generating class '{alias}' with prompt: {prompt}")

        for i in range(n_images_per_class):
            with torch.autocast(autocast_device, enabled=(accelerator.device.type == "cuda")):
                image = pipeline(prompt, num_inference_steps=25, generator=class_generator).images[0]
            out_name = f"{alias}_{i:02d}_epoch{epoch:03d}.png"
            image.save(os.path.join(samples_dir, out_name))

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info(f"Validation results saved → {samples_dir}")

# ------------------------------------------------------------------------------
# PARSER
# ------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Stable Diffusion on DermMNIST.")

    parser.add_argument("--pretrained_model_name_or_path", type=str, default="Manojb/stable-diffusion-2-1-base")
    parser.add_argument("--embed_path", type=str, default="./outputs/merged_embeds.pt")
    parser.add_argument("--output_dir", type=str, default="./outputs/lora_derma_finetune")
    parser.add_argument("--logging_dir", type=str, default="logs")

    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--max_train_steps", type=int, default=3000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num_vectors", type=int, default=1, help="Max number of vectors per class from TI.")
    parser.add_argument("--num_validation_images", type=int, default=4)
    parser.add_argument("--validation_epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--log_interval", type=int, default=20)

    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--allow_tf32", action="store_true")

    return parser.parse_args()

# ------------------------------------------------------------------------------
# DATASET
# ------------------------------------------------------------------------------

class DermMultiDataset(Dataset):
    def __init__(self, images, labels, tokenizer, image_transforms, num_vectors=1):
        self.images = images
        self.labels = labels
        self.tokenizer = tokenizer
        self.image_transforms = image_transforms
        self.num_vectors = num_vectors

        self.class_tokens = {}
        vocab = self.tokenizer.get_vocab()
        for cls_id, alias in id_to_alias.items():
            base_token = f"<{alias}_lesion>"
            tokens = [base_token]
            for i in range(1, self.num_vectors):
                next_token = f"{base_token}_{i}"
                if next_token in vocab:
                    tokens.append(next_token)
            self.class_tokens[cls_id] = " ".join(tokens)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        lbl = self.labels[idx]

        full_token_str = self.class_tokens[lbl]
        prompt = random.choice(skin_disease_prompts).format(full_token_str)

        pixel_values = self.image_transforms(image)
        input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
        }

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------

def main():
    args = parse_args()

    # Deney klasörünü baştan oluştur
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        exp_name = (
            f"exp_{timestamp}_lr{args.learning_rate}_bs{args.train_batch_size}"
            f"_steps{args.max_train_steps}_rank{args.rank}"
        )
        args.output_dir = os.path.join(args.output_dir, exp_name)
        os.makedirs(args.output_dir, exist_ok=True)

    # Sadece dosyaya log yaz, terminale yazma
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []

    log_file = os.path.join(args.output_dir, "run.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    root_logger.addHandler(file_handler)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )

    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        hyperparam_path = os.path.join(args.output_dir, "hyperparams.txt")
        with open(hyperparam_path, "w") as f:
            for k, v in vars(args).items():
                f.write(f"{k}: {v}\n")
        logger.info(f"Experiment folder created → {args.output_dir}")

    # === Load pretrained components ===
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    # === Load merged TI embeddings with shape check ===
    if args.embed_path is not None:
        logger.info(f"Loading merged embeddings: {args.embed_path}")
        expected_dim = text_encoder.get_input_embeddings().weight.shape[1]

        merged = torch.load(args.embed_path, map_location="cpu")
        for token, tensor in merged.items():
            num_added = tokenizer.add_tokens(token)
            if num_added == 0:
                logger.warning(f"Token {token} already exists in tokenizer.")

            if tensor.ndim != 1 or tensor.shape[0] != expected_dim:
                raise ValueError(
                    f"Embedding for token '{token}' has shape {tuple(tensor.shape)}, "
                    f"expected ({expected_dim},)."
                )

            text_encoder.resize_token_embeddings(len(tokenizer))
            tok_id = tokenizer.convert_tokens_to_ids(token)
            text_encoder.get_input_embeddings().weight.data[tok_id] = tensor.to(
                text_encoder.get_input_embeddings().weight.dtype
            )

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # === LoRA setup ===
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)

    # === Precision & device ===
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=torch.float32)

    if args.mixed_precision in ["fp16", "bf16"]:
        cast_training_params(unet, dtype=torch.float32)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # === DermMNIST dataset & sampler ===
    import medmnist
    from medmnist import INFO

    info = INFO["dermamnist"]
    DataClass = getattr(medmnist, info["python_class"])
    base_ds = DataClass(split="train", download=True, size=224)

    all_images = []
    all_labels = []

    for img, lbl in base_ds:
        all_images.append(img.convert("RGB"))
        all_labels.append(int(lbl.item()) if hasattr(lbl, "item") else int(lbl))

    class_counts = Counter(all_labels)
    class_weights = {cls_id: 1.0 / math.sqrt(count) for cls_id, count in class_counts.items()}
    sample_weights = torch.DoubleTensor([class_weights[lbl] for lbl in all_labels])

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    if accelerator.is_main_process:
        logger.info("=== Sınıf Dağılımı ve Ağırlıklar (1/sqrt(count)) ===")
        for cls_id in sorted(class_counts.keys()):
            logger.info(
                f"Sınıf: {id_to_alias[cls_id].upper():<5} (ID: {cls_id}) | "
                f"Örnek: {class_counts[cls_id]:<4} | "
                f"Ağırlık: {class_weights[cls_id]:.4f}"
            )

    # train_transforms = transforms.Compose([
    #     transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    # ])
    
    train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(
        args.resolution,
        scale=(0.8, 1.0),
        ratio=(0.9, 1.1),
        interpolation=transforms.InterpolationMode.BILINEAR
    ),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(180),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

    train_dataset = DermMultiDataset(
        all_images,
        all_labels,
        tokenizer,
        train_transforms,
        args.num_vectors,
    )

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # === Optimizer & scheduler ===
    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())
    optimizer = torch.optim.AdamW(
        lora_layers,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    num_training_steps = args.max_train_steps * accelerator.num_processes
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=num_training_steps,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    logger.info("***** Running LoRA Fine-Tuning *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    for epoch in range(args.num_train_epochs):
        unet.train()
        for _, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                )

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                else:
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, unet.parameters()),
                        args.max_grad_norm,
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.log_interval == 0:
                    current_lr = lr_scheduler.get_last_lr()[0]
                    logger.info(
                        f"Step {global_step:05d} | loss={loss.item():.6f} | lr={current_lr:.8f}"
                    )

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process and (epoch + 1) % args.validation_epochs == 0:
            log_validation(text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, epoch)

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        unwrapped = accelerator.unwrap_model(unet)
        lora_state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped))
        StableDiffusionPipeline.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=lora_state,
            safe_serialization=True,
        )
        logger.info(f"Training complete. LoRA weights saved → {args.output_dir}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    accelerator.end_training()

if __name__ == "__main__":
    main()