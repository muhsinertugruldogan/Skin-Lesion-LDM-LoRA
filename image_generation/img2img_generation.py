#!/usr/bin/env python
# coding=utf-8

import os
import argparse
import random
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from medmnist import INFO
import medmnist
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import StableDiffusionImg2ImgPipeline

# =========================================================
# === CONFIGURATION MAPS ==================================
# =========================================================

id_to_alias = {
    0: "akiec",
    1: "bcc",
    2: "bkl",
    3: "df",
    4: "mel",
    5: "nv",
    6: "vasc"
}

CLASS_CONFIGS = {
    
    "nv":    {"strength": 0.45, "guidance_scale": 2.5},
    "mel":   {"strength": 0.50, "guidance_scale": 3.0},
    "bcc":   {"strength": 0.55, "guidance_scale": 3.0},
    "bkl":   {"strength": 0.55, "guidance_scale": 3.0},

    # sadece problemli sınıfları değiştir
    "akiec": {"strength": 0.50, "guidance_scale": 3.0},
    "vasc":  {"strength": 0.55, "guidance_scale": 3.0},
    "df":    {"strength": 0.60, "guidance_scale": 3.5},
}

prompt_templates = [
    "a dermoscopic image of {}",
    "a dermoscopic photo of {}",
    "a close-up dermoscopic image of {}",
    "a clinical dermoscopic image of {}",
]

# =========================================================
# === HELPERS =============================================
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Img2Img Generation with LoRA and TI for DermMNIST")
    parser.add_argument("--exp_name", type=str, default="experiment_v1", help="Name of the experiment for output folder")
    parser.add_argument("--model_name", type=str, default="Manojb/stable-diffusion-2-1-base")
    parser.add_argument("--embed_path", type=str, default="./outputs/merged_embeds.pt")
    parser.add_argument(
        "--lora_path", 
        type=str, 
        default="/home/edogan/Downloads/ertugrul/myenv/ti_lora_image_generation/outputs/lora_derma_finetune/exp_2026-03-15_00-29-56_lr0.0001_bs8_steps4500_rank8/pytorch_lora_weights.safetensors", 
        help="Path to the trained LoRA directory or file"
    )
    
    parser.add_argument("--strength", type=float, default=0.60, help="Fallback denoising strength")
    parser.add_argument("--guidance_scale", type=float, default=5.5, help="Fallback CFG Scale")
    parser.add_argument("--steps", type=int, default=40, help="Number of inference steps")
    parser.add_argument("--n_per_class", type=int, default=200, help="Number of images to generate per class")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for generation")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for reproducibility")
    
    return parser.parse_args()


def load_derma_embeddings(embed_path, model_path, device, dtype=torch.float32):
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")

    embeds = torch.load(embed_path, map_location="cpu")

    added = 0
    for token, tensor in embeds.items():
        if token not in tokenizer.get_vocab():
            tokenizer.add_tokens(token)
            added += 1

    if added > 0:
        text_encoder.resize_token_embeddings(len(tokenizer))

    text_encoder = text_encoder.to(device, dtype=dtype)

    embed_dtype = text_encoder.get_input_embeddings().weight.dtype
    for token, tensor in embeds.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        text_encoder.get_input_embeddings().weight.data[token_id] = tensor.to(
            device=device, dtype=embed_dtype
        )

    print(f"[INFO] Loaded {len(embeds)} textual inversion tokens (added {added}).")
    return tokenizer, text_encoder


def random_prompt(alias):
    return random.choice(prompt_templates).format(f"<{alias}_lesion>")

# =========================================================
# === MAIN ================================================
# =========================================================

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if os.path.isfile(args.lora_path):
        lora_dir = os.path.dirname(args.lora_path)
    else:
        lora_dir = args.lora_path.rstrip("/")

    folder_name = f"generated_images_{args.exp_name}"
    output_dir = os.path.join(lora_dir, folder_name)
    
    torch.set_grad_enabled(False)
    os.makedirs(output_dir, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("[INFO] Loading Pipeline...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
    ).to(device)

    print(f"[INFO] Loading LoRA from: {args.lora_path}")
    pipe.load_lora_weights(args.lora_path)

    print("[INFO] Loading Textual Inversion...")
    tokenizer, text_encoder = load_derma_embeddings(
        args.embed_path, args.model_name, device, dtype=torch.float32
    )

    pipe.tokenizer = tokenizer
    pipe.text_encoder = text_encoder

    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()

    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("[INFO] xFormers enabled.")
    except Exception as e:
        print(f"[WARN] xFormers not enabled: {e}")

    pipe.unet.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)

    print("[INFO] Loading DermMNIST...")
    info = INFO["dermamnist"]
    DataClass = getattr(medmnist, info["python_class"])
    base_ds = DataClass(split="train", download=True, size=224)

    to_pil = transforms.Compose([
        transforms.Lambda(lambda x: x if isinstance(x, Image.Image) else transforms.functional.to_pil_image(x)),
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC)
    ])

    total_images = args.n_per_class * len(id_to_alias)
    print(f"[INFO] Generating {args.n_per_class} images per class × {len(id_to_alias)} classes = {total_images} total")

    progress_bar = tqdm(total=total_images, desc="Overall progress", ncols=100)
    use_autocast = device == "cuda"

    for cls_idx, alias in id_to_alias.items():
        current_strength = CLASS_CONFIGS.get(alias, {}).get("strength", args.strength)
        current_guidance = CLASS_CONFIGS.get(alias, {}).get("guidance_scale", args.guidance_scale)
        
        print(f"\n[CLASS] generating: {alias.upper()} | Config -> Strength: {current_strength} | Guidance: {current_guidance}")
        
        cls_out = os.path.join(output_dir, alias.upper())
        os.makedirs(cls_out, exist_ok=True)

        indices = [
            i for i, (_, y) in enumerate(base_ds)
            if int(y.item() if hasattr(y, "item") else y) == cls_idx
        ]

        if not indices:
            print(f"[WARN] No samples for {alias}, skipping.")
            continue

        generated_count = 0
        n_batches = (args.n_per_class + args.batch_size - 1) // args.batch_size

        shuffled_indices = indices.copy()
        random.shuffle(shuffled_indices)
        ptr = 0

        for batch_id in range(n_batches):
            current_bs = min(args.batch_size, args.n_per_class - generated_count)

            batch_indices = []
            for _ in range(current_bs):
                batch_indices.append(shuffled_indices[ptr])
                ptr += 1
                if ptr >= len(shuffled_indices):
                    random.shuffle(shuffled_indices)
                    ptr = 0

            batch_imgs = [
                to_pil(base_ds[i][0]) if not isinstance(base_ds[i][0], Image.Image)
                else base_ds[i][0].resize((512, 512), Image.BICUBIC)
                for i in batch_indices
            ]

            prompts = [random_prompt(alias) for _ in batch_imgs]

            batch_seed = args.seed + cls_idx * 10000 + batch_id
            rng = torch.Generator(device=device).manual_seed(batch_seed)

            with torch.no_grad():
                if use_autocast:
                    with torch.autocast("cuda", dtype=pipe.unet.dtype):
                        results = pipe(
                            prompt=prompts,
                            image=batch_imgs,
                            strength=current_strength,
                            guidance_scale=current_guidance,
                            num_inference_steps=args.steps,
                            generator=rng
                        ).images
                else:
                    results = pipe(
                        prompt=prompts,
                        image=batch_imgs,
                        strength=current_strength,
                        guidance_scale=current_guidance,
                        num_inference_steps=args.steps,
                        generator=rng
                    ).images

            for img in results:
                img_id = generated_count
                img.save(os.path.join(cls_out, f"{alias}_{img_id:05d}.jpg"))
                generated_count += 1
                progress_bar.update(1)

            if generated_count >= args.n_per_class:
                break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[DONE] {alias.upper()}: {generated_count} images saved → {cls_out}")

    progress_bar.close()
    print(f"\nAll synthetic images generated successfully in: {output_dir}")

if __name__ == "__main__":
    main()