# **Skin Lesion Classification using Latent Diffusion Models with Low-Rank Adaptation for Synthetic Data Augmentation**

This repository contains the official PyTorch implementation of the framework presented in the paper:

**"Skin Lesion Classification using Latent Diffusion Models with Low-Rank Adaptation for Synthetic Data Augmentation"** (Submitted to SDS 2026).

**Authors:** Muhsin Ertuğrul Doğan and Çiğdem Eroğlu Erdem

## **Methodology Overview**

To address severe class imbalance in dermatological datasets (e.g., DermaMNIST), we propose a diffusion-based augmentation framework. This framework adapts a pretrained Stable Diffusion 2.1 model to the medical domain using Parameter-Efficient Fine-Tuning (PEFT) techniques:

1. Textual Inversion (TI): Learning class-specific tokens (e.g., \<mel\_lesion\>) to represent unique dermatological features.  
2. Low-Rank Adaptation (LoRA): Efficiently fine-tuning the diffusion model to capture clinical textures and patterns.  
3. Synthetic Generation: Generating high-quality images for minority classes through Text-to-Image (T2I) and Image-to-Image (I2I) pipelines.

## **Pipeline**

The workflow consists of the following five explicit steps:

1. Train Textual Inversion (TI): Learn class-specific embeddings for each lesion type.  
2. Merge Embeddings: Consolidate learned TI tokens into a single file.  
3. Train LoRA: Fine-tune the LDM using merged TI tokens to capture clinical details.  
4. Generate Images:  
   * I2I: Produce synthetic samples using the img2img\_generation.py script.  
   * T2I: Produce synthetic samples using the ti\_lora\_t2i\_img\_gen.ipynb notebook.  
5. Train Classifier: Train ResNet-18 or ViT-B/16 models using the augmented dataset.

## **Dataset**

* DermaMNIST: A 7-class skin lesion dataset (from MedMNIST) characterized by severe class imbalance.  
* SyntheticDerma: A custom augmented dataset class that combines original real images with synthetic samples generated via the diffusion pipeline to achieve a balanced distribution.

## **Installation**

conda create \-n ldm-lora python=3.10  
conda activate ldm-lora  
pip install \-r requirements.txt

## **Usage**

### **1\. Textual Inversion (TI) Training**

Train class-specific embeddings for a target lesion type (e.g., Melanoma):

accelerate launch image\_generation/ti\_training.py \\  
  \--pretrained\_model\_name\_or\_path Manojb/stable-diffusion-2-1-base \\  
  \--dataset\_name dermamnist \\  
  \--class\_name mel \\  
  \--placeholder\_token "\<mel\_lesion\>" \\  
  \--initializer\_token skin \\  
  \--output\_dir ./outputs/ti\_mel \\  
  \--resolution 512 \\  
  \--train\_batch\_size 8 \\  
  \--repeats 10 \\  
  \--learning\_rate 5e-4 \\  
  \--max\_train\_steps 3000 \\  
  \--gradient\_accumulation\_steps 1 \\  
  \--mixed\_precision fp16 \\  
  \--save\_steps 500 \\  
  \--checkpointing\_steps 500 \\  
  \--validation\_steps 500 \\  
  \--num\_validation\_images 6 \\  
  \--seed 42

### **2\. Merge Embeddings**

Consolidate individual class embeddings into a single file for LoRA training:

python image\_generation/merge\_embeddings.py

### **3\. LoRA Fine-tuning**

Fine-tune the model using the merged TI tokens to capture dermatological domain features:

accelerate launch image\_generation/ti\_lora\_train.py \\  
  \--output\_dir ./outputs/lora\_derma\_finetune \\  
  \--train\_batch\_size 8 \\  
  \--gradient\_accumulation\_steps 4 \\  
  \--learning\_rate 1e-4 \\  
  \--lr\_scheduler cosine \\  
  \--max\_train\_steps 4500 \\  
  \--rank 32 \\  
  \--num\_validation\_images 4 \\  
  \--validation\_epochs 1

### **4\. Image Generation**

**Image-to-Image (I2I) Generation:**

Generate synthetic data using the trained LoRA weights:

python image\_generation/img2img\_generation.py \\  
  \--lora\_path ./outputs/lora\_derma\_finetune/exp\_2026-03-15\_02-37-52\_lr0.0001\_bs8\_steps4500\_rank16/pytorch\_lora\_weights.safetensors \\  
  \--n\_per\_class 2000 \\  
  \--steps 30 \\  
  \--batch\_size 32 \\  
  \--exp\_name best\_dataset

**Text-to-Image (T2I) Generation:**

Use the provided Jupyter Notebook to create a synthetic dataset from text prompts using the learned TI tokens and LoRA:

* Refer to image\_generation/ti\_lora\_t2i\_img\_gen.ipynb.

### **5\. Classification Training**

Train the classifier (e.g., ResNet-18) on the augmented SyntheticDerma dataset:

python classification/Common\_Main.py \\  
  \--dataname SyntheticDerma \\  
  \--backbone resnet18 \\  
  \--backbone\_update ft \\  
  \--BatchSize 32 \\  
  \--lr 4e-4 \\  
  \--bb\_lr 2e-4 \\  
  \--lr\_decay 0.95 \\  
  \--lr\_decay\_iters 500 \\  
  \--total\_Iter\_Num 25000 \\  
  \--syn\_folder exp\_2025-11-05\_02-27-44\_lr0.0005\_bs16\_steps3000\_rank32\_i2i\_transform

