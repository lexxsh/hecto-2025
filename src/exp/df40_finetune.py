from .. import config as C
from ..config import Config
from src.config import Config, Augmentations

experiments = {   
    "df40-openfake_final": [
        Config(
            # Model (Large 336 - 256 image will be upscaled but checkpoint compatible)
            backbone=C.Backbone.PerceptionEncoder_L_p14_336,
            backbone_args=C.BackboneArgs(img_size=224),
            checkpoint="../../model/GenD_PE_finetune",
            
            head=C.Head.Linear,
            unfreeze_layers=["norm_pre", "norm1", "norm2", "norm"],
            loss=C.Loss(
                ce_labels=1.0,
                uniformity=0.1,
                alignment_labels=0.5,
                class_weights=[1.0, 1.0]  # [real_weight, fake_weight]
            ),
            
            # Training hyperparams
            lr=2e-4,
            min_lr=1e-5,
            warmup_epochs=1,  # Warmup for stable training with unfrozen LayerNorm
            weight_decay=0.01,
            optimizer="AdamW",
            lr_scheduler="cosine",
            num_epochs_in_cycle=1,
            
            trn_files=[
                # FS (Face-Swapping) - 11 files
                "config/datasets/DF40/FS/fsgan_ff_data_deepfake.txt",
                "config/datasets/DF40/FS/faceswap_ff_data_deepfake.txt",
                "config/datasets/DF40/FS/simswap_FF_data_deepfake.txt",
                "config/datasets/DF40/FS/inswap_FF_data_deepfake.txt",
                "config/datasets/DF40/FS/blendface_ff_data_deepfake.txt",
                "config/datasets/DF40/FS/uniface_FF_data_deepfake.txt",
                "config/datasets/DF40/FS/mobileswap_FF_data_deepfake.txt",
                "config/datasets/DF40/FS/e4s_ff_data_deepfake.txt",
                "config/datasets/DF40/FS/facedancer_FF_data_deepfake.txt",
                "config/datasets/DF40/FS/deepfacelab_fake_data_deepfake.txt",
                "config/datasets/DF40/FS/deepfacelab_real_data_deepfake.txt",
                
                # FR (Face-Reenactment) - 14 files
                "config/datasets/DF40/FR/fomm_FF_data_deepfake.txt",
                "config/datasets/DF40/FR/facevivid_ff_data_deepfake.txt",
                "config/datasets/DF40/FR/wav2lip_FF_data_deepfake.txt",
                "config/datasets/DF40/FR/mraa_ff_data_deepfake.txt",
                "config/datasets/DF40/FR/oneshot_ff_data_deepfake.txt",
                "config/datasets/DF40/FR/pirender_FF_data_deepfake.txt",
                "config/datasets/DF40/FR/tpsm_ff_data_deepfake.txt",
                "config/datasets/DF40/FR/lia_ff_data_deepfake.txt",
                "config/datasets/DF40/FR/danet_ff_data_deepfake.txt",
                "config/datasets/DF40/FR/sadtalker_FF_data_deepfake.txt",
                "config/datasets/DF40/FR/mcnet_ff_data_deepfake.txt",
                "config/datasets/DF40/FR/hyperreenact_ff_data_deepfake.txt",
                "config/datasets/DF40/FR/heygen_FF_data_deepfake.txt",
                "config/datasets/DF40/FR/heygen_real_data_deepfake.txt",
                
                # EFS (Entire Face Synthesis) - 14 files
                "config/datasets/DF40/EFS/vqgan_ff_data_deepfake.txt",
                "config/datasets/DF40/EFS/StyleGAN2_FF_data_deepfake.txt",
                "config/datasets/DF40/EFS/stylegan3_ff_data_deepfake.txt",
                "config/datasets/DF40/EFS/StyleGANXL_FF_data_deepfake.txt",
                "config/datasets/DF40/EFS/sd2.1_FF_data_deepfake.txt",
                "config/datasets/DF40/EFS/ddim_ff_data_deepfake.txt",
                "config/datasets/DF40/EFS/rddm_ff_data_deepfake.txt",
                "config/datasets/DF40/EFS/pixart_FF_data_deepfake.txt",
                "config/datasets/DF40/EFS/DiT_FF_data_deepfake.txt",
                "config/datasets/DF40/EFS/SiT_FF_data_deepfake.txt",
                "config/datasets/DF40/EFS/MidJourney_fake_data_deepfake.txt",
                "config/datasets/DF40/EFS/MidJourney_real_data_deepfake.txt",
                "config/datasets/DF40/EFS/whichisreal_fake_data_deepfake.txt",
                "config/datasets/DF40/EFS/whichisreal_real_data_deepfake.txt",
                
                # FE (Face Edit) - 9 files
                "config/datasets/DF40/FE/collabdiff_fake_data_deepfake.txt",
                "config/datasets/DF40/FE/collabdiff_real_data_deepfake.txt",
                "config/datasets/DF40/FE/e4e_ff_data_deepfake.txt",
                "config/datasets/DF40/FE/stargan_fake_data_deepfake.txt",
                "config/datasets/DF40/FE/stargan_real_data_deepfake.txt",
                "config/datasets/DF40/FE/starganv2_fake_data_deepfake.txt",
                "config/datasets/DF40/FE/starganv2_real_data_deepfake.txt",
                "config/datasets/DF40/FE/styleclip_fake_data_deepfake.txt",
                "config/datasets/DF40/FE/styleclip_real_data_deepfake.txt",

                "config/datasets/DF40/real_FF_data_deepfake.txt",
              
                "config/datasets/Openfake/OpenFake_336/midjourney-7_data_deepfake.txt",
                "config/datasets/Openfake/OpenFake_336/real_sampled_v4_data_deepfake.txt",
                "config/datasets/Openfake/OpenFake_336/sd3-5_sampled_data_deepfake.txt",
                "config/datasets/Openfake/OpenFake_336/sdxl_sampled_data_deepfake.txt",
                "config/datasets/Openfake/OpenFake_336/imagen-4.0_sampled_data_deepfake.txt"
            ],
            val_files=[
                "config/datasets/DF40/DF40_vali_fake_data_deepfake.txt", 
                "config/datasets/DF40/DF40_vali_real_data_deepfake.txt", 
                "config/datasets/Deepfake_eval/fake_data_deepfake.txt",
                "config/datasets/Deepfake_eval/real_data_deepfake.txt",
            ],  
            tst_files=["config/datasets/Deepfake_eval/fake_data_deepfake.txt",
                       "config/datasets/Deepfake_eval/real_data_deepfake.txt"],

            batch_size=64,
            mini_batch_size=32,
            max_epochs=5,
            
            # Hardware
            run_dir="runs",
            wandb=True,
            devices=[0],
            num_workers=8,
            #use_balanced_sampling=True,
            #sampling_strategy="weighted",
            # Checkpoint
            monitor_metric="train/mAP_frame",
            monitor_metric_mode="max",
        )
    ],
    
}   