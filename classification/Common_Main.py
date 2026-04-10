# -*- coding:utf-8 -*-

import os
# # Set the custom path where you want to save the pretrained models
# custom_cache_path = "/gpfs3/well/papiez/users/cub991/PJ2022/EPLF/FoundCheckpoints/TIMM"
#
# # Set the TORCH_HOME environment variable to the custom path
# os.environ['HF_HOME'] = "/gpfs3/well/papiez/users/cub991/PJ2022/EPLF/FoundCheckpoints/TIMM"
import sys
import argparse
import torch
# torch.hub.set_dir("/gpfs3/well/papiez/users/cub991/PJ2022/EPLF/FoundCheckpoints/TIMM")
from torch.backends import cudnn
# from Network import EffFTFoundClsNet
from Common_Trainer import Trainer
from dataset import *
import numpy as np
import datetime
from utils import *
# from Backbones.DINO.utils import neq_load_external_vit
import timm
from torch.utils.data import WeightedRandomSampler
torch.multiprocessing.set_sharing_strategy('file_system')

print(os.environ.get('HF_HOME'))

""" Path Config """

embedding_dim_info = {'SAM_vit_b':256, 'SAM_vit_l':256, 'SAM_vit_h':256, 'MedSAM_vit_b':256,'DINO_vit_small':384, 'DINO_vit_base':768, 'DINO_vit_large':1024,'dinov2_vits14':384, 'dinov2_vitb14':768, 'dinov2_vitl14':1024, 'dinov2_vitg14':1536}
embedding_patchsize_info = {'SAM_vit_b':16, 'SAM_vit_l':16, 'SAM_vit_h':16, 'MedSAM_vit_b':16,'DINO_vit_small':16, 'DINO_vit_base':16, 'DINO_vit_large':16,'dinov2_vits14':14, 'dinov2_vitb14':14, 'dinov2_vitl14':14, 'dinov2_vitg14':14}
Dataset_Name = ['DermaMNIST','PneumoniaMNIST','OrganAMNIST','OrganCMNIST','OrganSMNIST','PathMNIST','BreastMNIST','OCTMNIST','ChestMNIST','RetinaMNIST','BloodMNIST','TissueMNIST']
LR_List1 = [1e-6]
LR_List2 = [1e-7]

Backbon_Dict = {'vgg16':'vgg16.tv_in1k',
                'resnet18':'resnet18.a1_in1k',
                'densenet121':'densenet121.ra_in1k',
                'effi_b4':'efficientnet_b4.ra2_in1k',

                'Incept_v3':'inception_v3.tv_in1k', # classification

                'InceptResnet_v2':'inception_resnet_v2.tf_in1k', # classification
                #'Mobile_v3':'mobilenetv3_small_100.lamb_in1k',# classification


                'vit_b16':'vit_base_patch16_224.augreg2_in21k_ft_in1k',
                'clip_b16':'vit_base_patch16_clip_224.laion2b_ft_in12k_in1k',
                'eva_clip':  'eva02_base_patch14_224.mim_in22k', #'eva02_base_patch16_clip_224.merged2b_s8b_b131k',
                'clip_opai':'vit_base_patch16_clip_224.openai_ft_in12k_in1k',

                'clip_opaif':'vit_base_patch16_clip_224.openai',
                'dino_base':'vit_base_patch16_224.dino',
                'dino_small':'vit_small_patch16_224.dino',
                'dino2_small':'vit_small_patch14_dinov2.lvd142m',
                'dino2_base':'vit_base_patch14_dinov2.lvd142m',
                'sam_base':'samvit_base_patch16.sa1b',
                'sam_cls':'vit_base_patch16_224.sam_in1k'}


# vgg16.tv_in1k, resnet18.a1_in1k, densenet121.ra_in1k, (for classification)
# efficientnet_b4.ra2_in1k, (for classification)
# vit_base_patch16_224.augreg2_in21k_ft_in1k,  (for classification)
# vit_base_patch16_clip_224.laion2b_ft_in12k_in1k, (for classification)
# vit_base_patch16_clip_224.openai_ft_in12k_in1k  (for classification) # new
# vit_base_patch16_clip_224.openai (for feature extraction) # new
# eva02_base_patch16_clip_224.merged2b_s8b_b131k, (for classification)
# vit_base_patch16_224.dino, vit_small_patch16_224.dino, (for feature extraction)
# vit_small_patch14_dinov2.lvd142m, vit_base_patch14_dinov2.lvd142m (for feature extraction) # new
# samvit_base_patch16.sa1b, (for feature extraction)
# vit_base_patch16_224.sam_in1k (for classification) # new

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


#add_path('/home/wfp/2020-Semi-PU/TPAMI_Rebuttal_R2/REFUGE/CPS/Uniform_Framework/furnace')

def training_validation(configs):
    # pretrain
    cur_path = os.path.abspath(os.curdir)
    #configs.pretrained_model = cur_path + '/' + 'DATA/pytorch-weight/resnet50_v1c.pth'
    # arguments: dataname,img_size,img_resize,train_num_per_cls, // backbone,backbone_update,bb_lr,feat_adaptor_depth,linear_clsfier_depth,num_heads,//  prob_type,global_w,total_Iter_Num,BatchSize,Optim,lr
    SAVE_DIR_Prefix = cur_path + '/' + '{}_{}_{}/'.format(configs.dataname,configs.img_size,configs.img_resize) \
                + '{}_{}/'.format(configs.backbone_update,configs.bb_lr)+ '{}_Iter{}/'.format(configs.backbone,configs.total_Iter_Num)+ '{}_lr{}_{}_bs{}/'.format(configs.Optim, configs.lr,configs.lr_decay,configs.BatchSize)

    if hasattr(configs, "target_ir") and configs.target_ir > 1:
        SAVE_DIR_Prefix = SAVE_DIR_Prefix.rstrip('/') + f'_IR{configs.target_ir}/'
        
    if configs.dataname.lower() == 'syntheticderma' and configs.syn_folder:
        SAVE_DIR_Prefix = SAVE_DIR_Prefix.rstrip('/') + f"_synthetic_{configs.syn_folder}/"

    
    if not os.path.exists(SAVE_DIR_Prefix):
        # os.mkdir(SAVE_DIR)
        os.makedirs(SAVE_DIR_Prefix)



    for sample_random_seed in configs.random_seeds:
        save_result_file = SAVE_DIR_Prefix + 'seed{}/test_ndarray.npy'.format(sample_random_seed)
        if os.path.exists(save_result_file):
            continue

        if sample_random_seed < 100:

            control_seed(sample_random_seed)

            train_starttime = datetime.datetime.now()


            SAVE_DIR_Seed = SAVE_DIR_Prefix + 'seed{}/'.format(sample_random_seed)
            if not os.path.exists(SAVE_DIR_Seed):
                # os.mkdir(SAVE_DIR)
                os.makedirs(SAVE_DIR_Seed)

            ## model
            configs.num_cls = MedMNIST_INFO[configs.dataname]
            out_ch = configs.num_cls
            if configs.num_cls == 2:
                out_ch = 1


            Model_Main = timm.create_model(Backbon_Dict[configs.backbone], pretrained=True, num_classes=out_ch, drop_rate=0.4, drop_path_rate=0.2)
 
            ## dataloader
            if 'dino2' in configs.backbone:
                configs.img_resize = 518
            mean, std = Model_Main.default_cfg['mean'], Model_Main.default_cfg['std']

            if configs.dataname.lower() == 'syntheticderma':
                print(">>> Using custom SyntheticDerma dataset (real + synthetic)")
                
                synth_counts = {"AKIEC": 200, 
                                "BCC": 200, 
                                "BKL": 200, 
                                "DF": 480, 
                                "MEL": 320, 
                                "NV": 0, 
                                "VASC": 200}
                
                syn_root = '/home/edogan/Downloads/ertugrul/myenv/ti_lora_image_generation/outputs/lora_derma_finetune/exp_2026-03-15_02-37-52_lr0.0001_bs8_steps4500_rank16/generated_images_best_dataset'
                # input_json = "/home/edogan/Downloads/ertugrul/myenv/FMMIC-supcon/cleanlab_results/cleanlab_exp_2026-03-15_02-37-52_lr0.0001_bs8_steps4500_rank16_generated_images_best_dataset_selection_results.json"
                # output_json = os.path.join(SAVE_DIR_Seed, f"selected_synt_images.json")
                
                train_dataset = SyntheticDataset(
                    derm_split='train',
                    synthetic_root=syn_root,        
                    use_real=True,
                    use_filtered=False,
                    json_class_count=False,              
                    # filtered_json_path=input_json,  
                    # export_json_path=output_json,
                    dynamic_alpha=False,
                    synth_counts_dict=None, 
                    seed=configs.syn_seed,
                    mean=mean,
                    std=std,
                    img_size=configs.img_resize,
                    img_resize=configs.img_resize  
                )
                
                val_dataset = MedMNIST_Labels(dataset_name='DermaMNIST', train='val',
                                            img_size=configs.img_size, img_resize=configs.img_resize,
                                            download=True, mean=mean, std=std)
                test_dataset = MedMNIST_Labels(dataset_name='DermaMNIST', train='test',
                                            img_size=configs.img_size, img_resize=configs.img_resize,
                                            download=True, mean=mean, std=std)
            else:
                train_dataset = MedMNIST_Labels(dataset_name=configs.dataname, train='train', img_size=configs.img_size,
                                                img_resize=configs.img_resize, download=True,
                                                sample_num_per_cls=configs.train_num_per_cls, seed=sample_random_seed,
                                                mean=mean, std=std)
                val_dataset = MedMNIST_Labels(dataset_name=configs.dataname, train='val', img_size=configs.img_size,
                                            img_resize=configs.img_resize, download=True, mean=mean, std=std)
                test_dataset = MedMNIST_Labels(dataset_name=configs.dataname, train='test', img_size=configs.img_size,
                                            img_resize=configs.img_resize, download=True, mean=mean, std=std)
                        
            if len(train_dataset) < configs.BatchSize :
                            configs.BatchSize =len(train_dataset)
            
            y_train_indices = train_dataset.targets
            
            if isinstance(y_train_indices, torch.Tensor):
                y_train_indices = y_train_indices.cpu().numpy()
            if len(y_train_indices.shape) > 1:
                y_train_indices = y_train_indices.squeeze()
            
            class_sample_count = np.array(
                [len(np.where(y_train_indices == t)[0]) for t in np.unique(y_train_indices)]
            )
            weight = 1. / (np.power(class_sample_count, 0.5))
            print("Class sample counts:", class_sample_count)
            print("Class weights:", weight)
            
            samples_weight = np.array([weight[t] for t in y_train_indices])
            samples_weight = torch.from_numpy(samples_weight)
            sampler = WeightedRandomSampler(
                weights=samples_weight.double(),
                num_samples=len(samples_weight),
                replacement=True
            )
            
            train_loaders = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.BatchSize,
                                                        # shuffle=True,
                                                        shuffle=False,
                                                        sampler=sampler,
                                                        num_workers=4, pin_memory=True, drop_last=True)
            val_loaders = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=64, num_workers=4, pin_memory=True)
            test_loaders = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, num_workers=4, pin_memory=True)

            configs.iter_per_epoch = len(train_loaders)
            configs.total_epoch = configs.total_Iter_Num // configs.iter_per_epoch

            # training
            checkpoint_file = SAVE_DIR_Seed + 'checkpoint.pth.tar'
            checkpoint = None
            if os.path.exists(checkpoint_file):
                checkpoint = torch.load(checkpoint_file)

            trainer = Trainer(Model_Main,checkpoint,configs,train_loaders,val_loaders, test_loaders,SAVE_DIR_Seed)
            trainer.train()

            train_endtime = datetime.datetime.now()
            print('train time:', (train_endtime - train_starttime).seconds)
    
    all_seeds_metrics = []
    
    for cv in range(len(configs.random_seeds)): 
        SAVE_DIR_Seed = SAVE_DIR_Prefix + 'seed{}/'.format(cv)
        metric_path = SAVE_DIR_Seed + 'test_ndarray.npy'
        
        if os.path.exists(metric_path):
            metric_ndarray = np.load(metric_path, allow_pickle=True)
            all_seeds_metrics.append(metric_ndarray[0]) 

    if len(all_seeds_metrics) > 0:
        all_seeds_metrics = np.array(all_seeds_metrics) 

        avg_metrics = np.mean(all_seeds_metrics, axis=0)
        std_metrics = np.std(all_seeds_metrics, axis=0)

        avg_acc, avg_auc, avg_f1, avg_acsa = avg_metrics[:4]
        std_acc, std_auc, std_f1, std_acsa = std_metrics[:4]

        avg_cls_accs = avg_metrics[4:]
        std_cls_accs = std_metrics[4:]

        with open("%s/fivef_testout_index.txt" % SAVE_DIR_Prefix, "a") as f:
            f.write("\n=======================================================\n")
            f.write(f"AGGREGATED RESULTS ({len(configs.random_seeds)} Seeds)\n")
            f.write("=======================================================\n")
            
            f.write(f"Overall Acc : {avg_acc:.4f} ± {std_acc:.4f}\n")
            f.write(f"Overall AUC : {avg_auc:.4f} ± {std_auc:.4f}\n")
            f.write(f"Macro F1    : {avg_f1:.4f} ± {std_f1:.4f}\n")
            f.write(f"Avg Cls Acc : {avg_acsa:.4f} ± {std_acsa:.4f}\n\n")
            
            f.write("--- Class Specific Mean ± Std ---\n")
            for i, (m, s) in enumerate(zip(avg_cls_accs, std_cls_accs)):
                f.write(f"Class {i}: {m:.4f} ± {s:.4f}\n")
            f.write("=======================================================\n")

def main(configs):


    training_validation(configs)


if __name__ == '__main__':


    ## set parameters
    parser = argparse.ArgumentParser()

    # network param
    parser.add_argument('--backbone', type=str, default='vgg16')
    # vgg16.tv_in1k, resnet18.a1_in1k, densenet121.ra_in1k, (for classification)
    # efficientnet_b4.ra2_in1k, (for classification)
    # vit_base_patch16_224.augreg2_in21k_ft_in1k,  (for classification)
    # vit_base_patch16_clip_224.laion2b_ft_in12k_in1k, (for classification)
    # vit_base_patch16_clip_224.openai_ft_in12k_in1k  (for classification) # new
    # vit_base_patch16_clip_224.openai (for feature extraction) # new
    # eva02_base_patch16_clip_224.merged2b_s8b_b131k, (for classification)
    # vit_base_patch16_224.dino, vit_small_patch16_224.dino, (for feature extraction)
    # vit_small_patch14_dinov2.lvd142m, vit_base_patch14_dinov2.lvd142m (for feature extraction) # new
    # samvit_base_patch16.sa1b, (for feature extraction)
    # vit_base_patch16_224.sam_in1k (for classification) # new


    # 'SAM_vit_b', 'SAM_vit_l', 'SAM_vit_h', 'MedSAM_vit_b','DINO_vit_small', 'DINO_vit_base', 'DINO_vit_large','dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'
    # parser.add_argument('--patch_size', type=int, default=16)
    # parser.add_argument('--feat_adaptor_depth', type=int, default=3)
    # parser.add_argument('--num_heads', type=int, default=3)
    # parser.add_argument('--linear_clsfier_depth', type=int, default=1) #linear_clsfier_depth

    # parser.add_argument('--pretrained_decoder_path', type=str,
    #                     default='/gpfs3/well/papiez/users/cub991/PJ2022/EPLF/MedMNIST/FTFound')
    # parser.add_argument('--pt_decoder_cp_file', type=str,
    #                     default='/gpfs3/well/papiez/users/cub991/PJ2022/EPLF/MedMNIST/FTFound')





    # dataset param
    parser.add_argument('--job_array_id', type=int, default=-10)
    parser.add_argument('--dataname', type=str, default='PathMNIST')  # 'OrganCMNIST', 'OrganSMNIST', 'PathMNIST'
    parser.add_argument('--img_size', type=int, default=224) # 28, 62, 128, 224
    parser.add_argument('--img_resize', type=int, default=224)  # DINO: 224 ,  SAM:1024
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--num_cls', type=int, default=3)

    parser.add_argument('--train_num_per_cls',type=int, default=10000) #

    parser.add_argument('--random_seeds', nargs='+', type=int, default=[0, 1, 2])



    # train param
    parser.add_argument('--BatchSize', type=int, default=128)
    parser.add_argument('--iter_per_epoch', type=int, default=100)
    parser.add_argument('--GPU_ID', type=str, default='0')  # 10  ****************** tunable ******************
    # parser.add_argument('--temperature', type=float, default=1.0)
    #parser.add_argument('--task2_iter_per_epoch', type=int, default=100)momentum

    parser.add_argument('--momentum', type=float, default=0.9)

    parser.add_argument('--Optim', type=str, default='AdamW')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay', type=float, default=0.90)
    parser.add_argument('--lr_decay_iters', type=int, default=500)

    parser.add_argument('--backbone_update', type=str, default='freeze') # 'freeze', 'finetune'
    parser.add_argument('--bb_lr', type=float, default=1e-6)




    parser.add_argument('--total_Iter_Num', type=int, default=15000) #150
    parser.add_argument('--total_epoch', type=int, default=100)  # 200 LR_IRER


    parser.add_argument('--TRAIN_USE_CHECKPOINT', type=bool, default=False)

    parser.add_argument('--syn_seed', type=int, default=42)
    parser.add_argument('--syn_folder', type=str, default='', help='Optional folder name for synthetic dataset')


    ## loss param
    # parser.add_argument('--edge_loss_type', type=str, default='BCE')  # 'L1Loss', balanced
    # parser.add_argument('--edge_loss_pos_w', type=float, default=-1) # 0.95, None
    # parser.add_argument('--threshold_list', nargs='+', type=float,
    #                     default=[0.5,])





    CONFIGs = parser.parse_args()

    if CONFIGs.job_array_id > -1:
        CONFIGs.dataname = Dataset_Name[CONFIGs.job_array_id]

        if CONFIGs.backbone in ['vgg16', 'resnet18', 'densenet121', 'effi_b4', 'Incept_v3', 'InceptResnet_v2']:
            CONFIGs.bb_lr = LR_List1[0]
        else:
            CONFIGs.bb_lr = LR_List2[0]

    #os.environ["CUDA_VISIBLE_DEVICES"] = CONFIGs.GPU_ID

    if CONFIGs.backbone == 'dino2_base' and CONFIGs.backbone_update == 'ft':
        CONFIGs.BatchSize = 32
    else:
        CONFIGs.BatchSize = 64

    cudnn.benchmark = True



    main(CONFIGs)