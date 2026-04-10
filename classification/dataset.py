# -*- coding:utf-8 -*-

import os
import json
import random
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import PIL
from PIL import Image

import medmnist
from medmnist import INFO, Evaluator, OrganAMNIST, OrganCMNIST, OrganSMNIST, PathMNIST, DermaMNIST, BreastMNIST, OCTMNIST, PneumoniaMNIST, ChestMNIST, RetinaMNIST, BloodMNIST, TissueMNIST
from Backbones.SAM.utils.transforms import ResizeLongestSide
import torch.nn.functional as F

info = {"pathmnist": {
        "python_class": "PathMNIST",
        "description": "The PathMNIST is based on a prior study for predicting survival from colorectal cancer histology slides, providing a dataset (NCT-CRC-HE-100K) of 100,000 non-overlapping image patches from hematoxylin & eosin stained histological images, and a test dataset (CRC-VAL-HE-7K) of 7,180 image patches from a different clinical center. The dataset is comprised of 9 types of tissues, resulting in a multi-class classification task. We resize the source images of 3×224×224 into 3×28×28, and split NCT-CRC-HE-100K into training and validation set with a ratio of 9:1. The CRC-VAL-HE-7K is treated as the test set.",
        "url": "https://zenodo.org/records/10519652/files/pathmnist.npz?download=1",
        "MD5": "a8b06965200029087d5bd730944a56c1",
        "url_64": "https://zenodo.org/records/10519652/files/pathmnist_64.npz?download=1",
        "MD5_64": "55aa9c1e0525abe5a6b9d8343a507616",
        "url_128": "https://zenodo.org/records/10519652/files/pathmnist_128.npz?download=1",
        "MD5_128": "ac42d08fb904d92c244187169d1fd1d9",
        "url_224": "https://zenodo.org/records/10519652/files/pathmnist_224.npz?download=1",
        "MD5_224": "2c51a510bcdc9cf8ddb2af93af1eadec",
        "task": "multi-class",
        "label": {
            "0": "adipose",
            "1": "background",
            "2": "debris",
            "3": "lymphocytes",
            "4": "mucus",
            "5": "smooth muscle",
            "6": "normal colon mucosa",
            "7": "cancer-associated stroma",
            "8": "colorectal adenocarcinoma epithelium",
        },
        "n_channels": 3,
        "n_samples": {"train": 89996, "val": 10004, "test": 7180},
        "license": "CC BY 4.0",
    },
    'url':'https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py'
}

MedMNIST_INFO = {'OrganAMNIST':11,'OrganCMNIST':11,'OrganSMNIST':11,'PathMNIST':9,'DermaMNIST':7,'BreastMNIST':2,'OCTMNIST':4,'PneumoniaMNIST':2,'ChestMNIST':14,'RetinaMNIST':5,'BloodMNIST':8,'TissueMNIST':8,'SyntheticDerma':7}

class MedMNIST_Embedding_Labels(Dataset):
    def __init__(self,configs,train,sample_num_per_cls=10000,seed=None):
        root_path = os.path.expanduser("~/.cache/medmnist")
        os.makedirs(root_path, exist_ok=True)
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        target_size = configs.img_size if hasattr(configs, 'img_size') else 224

        if train:
            self.pre_process = transforms.Compose([
                transforms.RandomResizedCrop(target_size, scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC), 
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.1),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            self.pre_process = transforms.Compose([
                transforms.Resize((target_size, target_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        
        OrganCMNIST_dataset = None
        if configs.dataname == 'OrganCMNIST':
            OrganCMNIST_dataset = OrganCMNIST(split=train, transform=self.pre_process, download=True, root=root_path, size=configs.img_size)
        elif configs.dataname == 'OrganSMNIST':
            OrganCMNIST_dataset = OrganSMNIST(split=train, transform=self.pre_process, download=True, root=root_path, size=configs.img_size)
        elif configs.dataname == 'PathMNIST':
            OrganCMNIST_dataset = PathMNIST(split=train, transform=self.pre_process, download=True, root=root_path, size=configs.img_size)
        elif configs.dataname == 'DermaMNIST':
            OrganCMNIST_dataset = DermaMNIST(split=train, transform=self.pre_process, download=True, root=root_path, size=configs.img_size)
        elif configs.dataname == 'BreastMNIST':
            OrganCMNIST_dataset = BreastMNIST(split=train, transform=self.pre_process, download=True, root=root_path, size=configs.img_size)
        elif configs.dataname == 'OrganAMNIST':
            OrganCMNIST_dataset = OrganAMNIST(split=train, transform=self.pre_process, download=True, root=root_path, size=configs.img_size)
        elif configs.dataname == 'OCTMNIST':
            OrganCMNIST_dataset = OCTMNIST(split=train, transform=self.pre_process, download=True, root=root_path, size=configs.img_size)
        elif configs.dataname == 'PneumoniaMNIST':
            OrganCMNIST_dataset = PneumoniaMNIST(split=train, transform=self.pre_process, download=True, root=root_path, size=configs.img_size)
        elif configs.dataname == 'ChestMNIST':
            OrganCMNIST_dataset = ChestMNIST(split=train, transform=self.pre_process, download=True, root=root_path, size=configs.img_size)
        elif configs.dataname == 'RetinaMNIST':
            OrganCMNIST_dataset = RetinaMNIST(split=train, transform=self.pre_process, download=True, root=root_path, size=configs.img_size)
        elif configs.dataname == 'BloodMNIST':
            OrganCMNIST_dataset = BloodMNIST(split=train, transform=self.pre_process, download=True, root=root_path, size=configs.img_size)
        elif configs.dataname == 'TissueMNIST':
            OrganCMNIST_dataset = TissueMNIST(split=train, transform=self.pre_process, download=True, root=root_path, size=configs.img_size)

        self.original_imgs = OrganCMNIST_dataset.imgs
        dims = len(self.original_imgs.shape)
        if dims == 3:
            self.original_imgs = np.expand_dims(self.original_imgs, axis=-1)
        self.original_targets = OrganCMNIST_dataset.labels

        save_path = '/gpfs3/well/papiez/users/cub991/PJ2022/EPLF/FoundEmbeddings/{}_{}/{}_{}/'.format(configs.dataname, configs.img_size, configs.backbone,configs.img_resize)
        embedding_file_name = save_path + '{}_img_embedding.npy'.format(train)
        gt_file_name = save_path + '{}_gt.npy'.format(train)

        self.imgs = np.load(embedding_file_name)
        self.targets = np.load(gt_file_name)
        self.num_cls = MedMNIST_INFO[configs.dataname]
        self.backbone = configs.backbone
        
        if 'SAM' in configs.backbone:
            N,C,W,H = self.imgs.shape
            self.embedding_size = C
            self.map_size = W
            self.patch_size = int(configs.img_resize/self.map_size)
            self.imgs = self.imgs.reshape(N,C,-1).transpose(0,2,1) 
        elif 'DINO' in configs.backbone or 'dinov2' in configs.backbone:
            B, N, C = self.imgs.shape
            self.embedding_size = C 
            self.map_size = int(math.sqrt(N-1))
            self.patch_size = int(configs.img_resize / self.map_size)

        if sample_num_per_cls < 1001:
            np.random.seed(seed)
            select_ids = []
            for label_id in range(self.num_cls):
                select_id = np.where(np.array(self.targets.squeeze(1)) == label_id)[0].tolist()
                select_ids = select_ids + select_id
            self.imgs = self.imgs[select_ids]
            self.targets = self.targets[select_ids]

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, index):
        data, label = self.imgs[index], self.targets[index].astype(int)
        data = torch.from_numpy(data)
        return data,torch.LongTensor(label)

# ======================================================================================
#   FULL SyntheticDataset 
# ======================================================================================

class SyntheticDataset(Dataset):
    def __init__(self,
                derm_split='train',
                synthetic_root=None,
                use_real=True,
                use_filtered=False,
                json_class_count=True, 
                filtered_json_path=None,
                export_json_path=None,
                dynamic_alpha=False, 
                synth_counts_dict=None,
                random_selection=True,
                seed=1234,
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
                img_size=224,
                img_resize=224,
                download=True):

        rng = random.Random(seed)
        root_path = os.path.expanduser("~/.cache/medmnist")
        os.makedirs(root_path, exist_ok=True)

        if synth_counts_dict is None:
            self.synth_counts_dict = {
                "AKIEC": 2500,
                "BCC": 2500,
                "BKL": 2500,
                "DF": 2500,
                "MEL": 2500,
                "NV": 0,    
                "VASC": 2500
            }
        else:
            self.synth_counts_dict = synth_counts_dict

        # --- Dataset meta ---
        data_flag = 'dermamnist'
        info = INFO[data_flag]
        self.class_names = list(info['label'].values())
        self.num_classes = len(self.class_names)
        self.dataset_name = "SyntheticDerma"
        self.use_real = use_real
        self.use_filtered = use_filtered
        self.json_class_count = json_class_count  # Sınıfa kaydet
        self.dynamic_alpha = dynamic_alpha 
        self.export_json_path = export_json_path 

        # ---------------------------
        # REAL DATA
        # ---------------------------
        if use_real:
            self.base_dataset = DermaMNIST(
                split=derm_split,
                transform=None,
                download=download,
                root=root_path,
                size=224
            )
            self.real_len = len(self.base_dataset)
        else:
            self.base_dataset = None
            self.real_len = 0

        self.img_size = img_size
        self.img_resize = img_resize

        # ---------------------------
        # Preprocessing
        # ---------------------------
        if img_resize == img_size:
            self.pre_process = transforms.Compose([
                transforms.RandomResizedCrop(
                    self.img_size, 
                    scale=(0.85, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2), 
                transforms.RandomRotation(degrees=20),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            self.sam_transform = ResizeLongestSide(self.img_resize)
            self.pre_process = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

        # ---------------------------
        # SYNTHETIC IMAGE POOL
        # ---------------------------
        all_paths = []
        if synthetic_root:
            all_paths += self._collect_image_paths(synthetic_root)

        pool_by_cls = {i: [] for i in range(self.num_classes)}
        
        # nv_idx = self._map_classname_to_label("NV")
        # print(nv_idx)
        # print(self._map_classname_to_label("NV"))
        
        for p, cname in all_paths:
            try:
                cid = self._map_classname_to_label(cname)
                
                # if cid == nv_idx:
                #     continue
                pool_by_cls[cid].append(p)
            except:
                continue
        nv_idx = self._map_classname_to_label("NV")
        pool_by_cls[nv_idx] = []        
        # =====================================================
        # SELECTION LOGIC
        # =====================================================
        
        class_to_idx = {
            "AKIEC": 0, "BCC": 1, "BKL": 2, "DF": 3, "MEL": 4, "NV": 5, "VASC": 6,
        }
        idx_to_class = {v: k for k, v in class_to_idx.items()}

        self.synthetic_paths = []
        self.exported_metadata = []

        if self.use_filtered and filtered_json_path:
            with open(filtered_json_path, "r") as f:
                raw_list = json.load(f)

            if self.json_class_count:
                print("[INFO] Applying strict Tier-Aware logic filtering (with quotas) to synthetic pools\n")
                
                json_candidates = {}
                for rec in raw_list:
                    fname = rec["filename"].strip().lower()
                    rec_class = rec.get("class", "").strip().upper()
                    
                    if rec_class in class_to_idx:
                        expected_idx = class_to_idx[rec_class]
                        if rec.get("selected") == True and rec.get("pred_class_idx") == expected_idx:
                            json_candidates[(rec_class, fname)] = rec

                for c in range(self.num_classes):
                    short_name = idx_to_class.get(c, "").upper()
                    
                    K = self.synth_counts_dict.get(short_name, 0)
                    if K <= 0: continue

                    original_list = pool_by_cls[c]
                    candidates = []
                    for p in original_list:
                        fname = os.path.basename(p).strip().lower()
                        key = (short_name, fname)
                        if key in json_candidates:
                            candidates.append({
                                "path": p, 
                                "confidence": float(json_candidates[key].get("confidence", 0.0)),
                                "record": json_candidates[key]
                            })

                    if not candidates:
                        print(f"[WARN] No valid candidates for {short_name} after JSON filtering.")
                        continue

                    candidates.sort(key=lambda x: x["confidence"], reverse=True)

                    high = [x for x in candidates if x["confidence"] >= 0.80]
                    medium = [x for x in candidates if 0.60 <= x["confidence"] < 0.80]
                    low = [x for x in candidates if x["confidence"] < 0.60]
                    
                    target_high = int(0.6 * K)
                    target_med = int(0.4 * K)
                    target_low = K - target_high - target_med 
                    
                    selected_high = high[:target_high]
                    selected_med = medium[:target_med]
                    selected_low = low[:target_low]
                    
                    deficit_high = target_high - len(selected_high)
                    deficit_med = target_med - len(selected_med)
                    deficit_low = target_low - len(selected_low)
                    
                    total_deficit = deficit_high + deficit_med + deficit_low
                    
                    if total_deficit > 0:
                        remaining_high = high[target_high:]
                        remaining_med = medium[target_med:]
                        remaining_low = low[target_low:]
                        
                        if len(remaining_high) > 0:
                            take_from_high = min(total_deficit, len(remaining_high))
                            selected_high += remaining_high[:take_from_high]
                            total_deficit -= take_from_high
                            remaining_high = remaining_high[take_from_high:] 
                            
                        if total_deficit > 0 and len(remaining_med) > 0:
                            take_from_med = min(total_deficit, len(remaining_med))
                            selected_med += remaining_med[:take_from_med]
                            total_deficit -= take_from_med
                            remaining_med = remaining_med[take_from_med:]
                            
                        if total_deficit > 0 and len(remaining_low) > 0:
                            take_from_low = min(total_deficit, len(remaining_low))
                            selected_low += remaining_low[:take_from_low]
                            total_deficit -= take_from_low
                            remaining_low = remaining_low[take_from_low:]
                    
                    selected_c = selected_high + selected_med + selected_low
                    
                    if len(selected_c) < K:
                        print(f"[WARN] {short_name}: Target {K}, but only found {len(selected_c)} valid images. (H:{len(selected_high)}, M:{len(selected_med)}, L:{len(selected_low)})")
                    else:
                        print(f"[INFO] {short_name}: Target {K} reached. (H:{len(selected_high)}, M:{len(selected_med)}, L:{len(selected_low)})")

                    for item in selected_c:
                        self.synthetic_paths.append((item["path"], c, item["confidence"]))
                        self.exported_metadata.append(item["record"])
                        
                print("\n--- Tier-Aware Filtering Complete ---\n")

            else:
                print("[INFO] Loading ALL valid synthetic images from JSON (Ignoring Quotas)...\n")
                
                physical_files_dict = {}
                for p, _ in all_paths:
                    fname = os.path.basename(p).strip().lower()
                    physical_files_dict[fname] = p

                matched_count = 0
                for rec in raw_list:
                    rec_class = rec.get("class", "").strip().upper()
                    
                    # if rec_class == "NV":
                    #     continue
                    
                    if rec_class in class_to_idx:
                        expected_idx = class_to_idx[rec_class]
                        
                        if rec.get("selected") == True and rec.get("pred_class_idx") == expected_idx:
                            fname = rec.get("filename", "").strip().lower()
                            
                            if fname in physical_files_dict:
                                full_path = physical_files_dict[fname]
                                conf = float(rec.get("confidence", 0.0))
                                
                                self.synthetic_paths.append((full_path, expected_idx, conf))
                                self.exported_metadata.append(rec)
                                matched_count += 1

                print(f"--- Loaded {matched_count} synthetic images from JSON (No Limits) ---\n")

            if self.export_json_path:
                with open(self.export_json_path, "w") as f:
                    json.dump(self.exported_metadata, f, indent=4)
                print(f"[INFO] Exported metadata of {len(self.exported_metadata)} selected images to {self.export_json_path}")

        else:
            print("[INFO] JSON Filtering OFF. Loading directly from physical pool based on synth_counts_dict...\n")
            for c in range(self.num_classes):
                short_name = idx_to_class.get(c, "").upper()
                K = self.synth_counts_dict.get(short_name, 0)
                
                if K <= 0: continue
                
                pool = pool_by_cls[c]
                
                if K > len(pool):
                    print(f"[WARN] Requested {K} for {short_name}, but pool only has {len(pool)}. Taking all.")
                    K = len(pool)

                picked = rng.sample(pool, K) if random_selection else pool[:K]
                for p in picked:
                    self.synthetic_paths.append((p, c, 1.0)) 


        # --------------------------------------------
        # SUMMARY & TARGETS
        # --------------------------------------------
        self.total_len = self.real_len + len(self.synthetic_paths)

        real_labels = []
        if self.use_real:
            for i in range(self.real_len):
                _, lab = self.base_dataset[i]
                if isinstance(lab, torch.Tensor):
                    lab = int(lab.item())
                elif isinstance(lab, np.ndarray):
                    lab = int(lab.squeeze())
                else:
                    lab = int(lab)
                real_labels.append(lab)

        real_labels = np.array(real_labels)
        real_counts = {i: int((real_labels == i).sum()) for i in range(self.num_classes)}
        synth_counts = {i: 0 for i in range(self.num_classes)}
        for item in self.synthetic_paths:
            c = item[1]
            synth_counts[c] += 1

        print("\n[Dataset Composition]")
        print(f"Total real samples: {self.real_len}")
        print(f"Total synthetic samples: {len(self.synthetic_paths)}")
        print(f"Total combined: {self.total_len}")
        print("Per-class distribution:")
        for i, cname in enumerate(self.class_names):
            short_n = idx_to_class.get(i, "")
            print(f"  {short_n:<5} - {cname:<45} | real: {real_counts.get(i, 0):4d} "
                f"| synthetic: {synth_counts.get(i, 0):4d} "
                f"| total: {real_counts.get(i, 0)+synth_counts.get(i, 0):4d}")
        print("-" * 85)
        
        all_targets = list(real_labels)
        for item in self.synthetic_paths:
            c = item[1]
            all_targets.append(c)
        self.targets = np.array(all_targets)

    # ======================================================================================
    # HELPERS
    # ======================================================================================
    def _collect_image_paths(self, root_dir):
        paths = []
        for class_name in sorted(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    paths.append((os.path.join(class_dir, fname), class_name))
        return paths

    def _map_classname_to_label(self, class_name):
        cn = class_name.lower()
        
        alias = {
            "akiec": "actinic keratoses and intraepithelial carcinoma",
            "bcc": "basal cell carcinoma",
            "bkl": "benign keratosis-like lesions",
            "df": "dermatofibroma",
            "mel": "melanoma",
            "nv": "melanocytic nevi",
            "vasc": "vascular lesions"
        }
        
        target_name = alias.get(cn, cn)
            
        for i, cname in enumerate(self.class_names):
            if cname.lower() == target_name:
                return i
                
        raise ValueError(f"Class eşleşmedi: '{class_name}'")

    # ======================================================================================
    # GET ITEM
    # ======================================================================================
    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if idx < self.real_len:
            img, lab = self.base_dataset[idx]

            if isinstance(img, torch.Tensor):
                img = Image.fromarray((img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
            else:
                img = Image.fromarray(np.array(img))

            img = img.convert("RGB")
            y = torch.tensor(int(lab), dtype=torch.long)
            y_soft = F.one_hot(y, num_classes=self.num_classes).float()

        else:
            synth_path, label, conf = self.synthetic_paths[idx - self.real_len]
            
            with Image.open(synth_path) as f:
                img = f.convert("RGB")

            y = torch.tensor(label, dtype=torch.long)
            one_hot = F.one_hot(y, num_classes=self.num_classes).float()

            if self.dynamic_alpha:
                if conf >= 0.80:
                    alpha = 0.05
                elif conf >= 0.60:
                    alpha = 0.10
                else:
                    alpha = 0.20
            else:
                alpha = 0.10
                
            y_soft = (1 - alpha) * one_hot + alpha / self.num_classes

        if self.img_resize == self.img_size:
            x = self.pre_process(img)
        else:
            sam_transform = ResizeLongestSide(self.img_resize)
            resized = sam_transform.apply_image(np.array(img))
            resized = Image.fromarray(resized).convert("RGB")
            x = self.pre_process(resized)

        return x, y_soft

class MedMNIST_Labels(Dataset):
    def __init__(self,dataset_name,train,img_size=224,img_resize=224,download=False,resize=False,sample_num_per_cls=10000,seed=None,mean = None, std=None):
        root_path = os.path.expanduser("~/.cache/medmnist")
        os.makedirs(root_path, exist_ok=True)
        
        if resize:
            self.pre_process = transforms.Compose(
                [transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST),
                 transforms.ToTensor(),
                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        else:
            self.pre_process =transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std),
                                        ])

        OrganCMNIST_dataset = None
        if dataset_name == 'OrganCMNIST':
            OrganCMNIST_dataset = OrganCMNIST(split=train, transform=self.pre_process, download=download, root=root_path, size=img_size)
        elif dataset_name == 'OrganSMNIST':
            OrganCMNIST_dataset = OrganSMNIST(split=train, transform=self.pre_process, download=download, root=root_path, size=img_size)
        elif dataset_name == 'PathMNIST':
            OrganCMNIST_dataset = PathMNIST(split=train, transform=self.pre_process, download=download, root=root_path, size=img_size)
        elif dataset_name == 'DermaMNIST':
            OrganCMNIST_dataset = DermaMNIST(split=train, transform=self.pre_process, download=download, root=root_path, size=img_size)
        elif dataset_name == 'BreastMNIST':
            OrganCMNIST_dataset = BreastMNIST(split=train, transform=self.pre_process, download=download, root=root_path, size=img_size)
        elif dataset_name == 'OrganAMNIST':
            OrganCMNIST_dataset = OrganAMNIST(split=train, transform=self.pre_process, download=download, root=root_path, size=img_size)
        elif dataset_name == 'OCTMNIST':
            OrganCMNIST_dataset = OCTMNIST(split=train, transform=self.pre_process, download=download, root=root_path, size=img_size)
        elif dataset_name == 'PneumoniaMNIST':
            OrganCMNIST_dataset = PneumoniaMNIST(split=train, transform=self.pre_process, download=download, root=root_path, size=img_size)
        elif dataset_name == 'ChestMNIST':
            OrganCMNIST_dataset = ChestMNIST(split=train, transform=self.pre_process, download=download, root=root_path, size=img_size)
        elif dataset_name == 'RetinaMNIST':
            OrganCMNIST_dataset = RetinaMNIST(split=train, transform=self.pre_process, download=download, root=root_path, size=img_size)
        elif dataset_name == 'BloodMNIST':
            OrganCMNIST_dataset = BloodMNIST(split=train, transform=self.pre_process, download=download, root=root_path, size=img_size)
        elif dataset_name == 'TissueMNIST':
            OrganCMNIST_dataset = TissueMNIST(split=train, transform=self.pre_process, download=download, root=root_path, size=img_size)

        self.dataset_name = dataset_name
        self.imgs = OrganCMNIST_dataset.imgs
        self.targets = OrganCMNIST_dataset.labels
        self.num_cls = MedMNIST_INFO[dataset_name]

        del OrganCMNIST_dataset

        if sample_num_per_cls < 1001:
            np.random.seed(seed)
            if dataset_name == 'ChestMNIST':
                indices = np.random.choice(self.imgs.shape[0], size=sample_num_per_cls*self.num_cls, replace=False)
                self.imgs = self.imgs[indices]
                self.targets = self.targets[indices]
            else:
                select_ids = []
                for label_id in range(self.num_cls):
                    select_id_all = np.where(np.array(self.targets.squeeze(1)) == label_id)[0].tolist()
                    select_id = select_id_all[:sample_num_per_cls]
                    select_ids = select_ids + select_id
                self.imgs = self.imgs[select_ids]
                self.targets = self.targets[select_ids]

        self.sam_pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(-1,1,1)
        self.sam_pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(-1,1,1)
        self.img_size = img_size
        self.img_resize = img_resize

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, index):
        data, label = self.imgs[index], self.targets[index].astype(int) 

        if self.img_resize == self.img_size:
            data = Image.fromarray(data)
            data = data.convert("RGB")
            resize_img_tensor = self.pre_process(data)
        else:
            sam_transform = ResizeLongestSide(self.img_resize)
            resize_img = sam_transform.apply_image(data)
            resize_img = Image.fromarray(resize_img)
            resize_img = resize_img.convert("RGB")
            resize_img_tensor = self.pre_process(resize_img)

        return resize_img_tensor,label