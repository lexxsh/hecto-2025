import os
import random
from pathlib import Path

# 랜덤 시드 고정
random.seed(42)

base_dir = Path("/home/gcl/lexxsh/deepfake/GenD/config/datasets/DF40")

# 방법론별 모델 매핑 (txt 파일명 기준)
categories = {
    "FS": [  # Face-Swapping - 10개
        "fsgan_ff.txt",
        "faceswap_ff.txt", 
        "simswap_FF.txt",
        "inswap_FF.txt",
        "blendface_ff.txt",
        "uniface_FF.txt",
        "mobileswap_FF.txt",
        "e4s_ff.txt",
        "facedancer_FF.txt",
        "deepfacelab_fake.txt",
        "deepfacelab_real.txt",
    ],
    "FR": [  # Face-Reenactment - 14개
        "fomm_FF.txt",
        "facevivid_ff.txt",
        "wav2lip_FF.txt",
        "mraa_ff.txt",
        "oneshot_ff.txt",
        "pirender_FF.txt",
        "tpsm_ff.txt",
        "lia_ff.txt",
        "danet_ff.txt",
        "sadtalker_FF.txt",
        "mcnet_ff.txt",
        "hyperreenact_ff.txt",
        "heygen_FF.txt",
        "heygen_real.txt",
    ],
    "EFS": [  # Entire Face Synthesis - 16개
        "vqgan_ff.txt",
        "StyleGAN2_FF.txt",
        "stylegan3_ff.txt",
        "StyleGANXL_FF.txt",
        "sd2.1_FF.txt",
        "ddim_ff.txt",
        "rddm_ff.txt",
        "pixart_FF.txt",
        "DiT_FF.txt",
        "SiT_FF.txt",
        "MidJourney_fake.txt",
        "MidJourney_real.txt",
        "whichisreal_fake.txt",
        "whichisreal_real.txt",
    ],
    "FE": [  # Face Edit - 9개
        "collabdiff_fake.txt",
        "collabdiff_real.txt",
        "e4e_ff.txt",
        "stargan_fake.txt",
        "stargan_real.txt",
        "starganv2_fake.txt",
        "starganv2_real.txt",
        "styleclip_fake.txt",
        "styleclip_real.txt",
    ]
}

# 각 카테고리별 폴더 생성 및 샘플링
for category, file_list in categories.items():
    category_dir = base_dir / category
    category_dir.mkdir(exist_ok=True)
    
    print(f"\n=== {category} ===")
    
    for filename in file_list:
        source_file = base_dir / filename
        
        if not source_file.exists():
            print(f"  ⚠️  {filename}: File not found, skipping")
            continue
        
        # 파일 읽기
        with open(source_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        total_count = len(lines)
        
        # 샘플링 개수 결정 (최대 1000개)
        sample_count = min(3000, total_count)
        
        if sample_count == 0:
            print(f"  ⚠️  {filename}: Empty file, skipping")
            continue
        
        # 랜덤 샘플링
        sampled_lines = random.sample(lines, sample_count)
        
        # 새 파일에 저장
        target_file = category_dir / filename
        with open(target_file, 'w') as f:
            f.write('\n'.join(sampled_lines))
            if sampled_lines:
                f.write('\n')
        
        print(f"  ✓ {filename}: Sampled {sample_count}/{total_count} → {category}/{filename}")

print("\n=== Summary ===")
for category in categories.keys():
    category_dir = base_dir / category
    if category_dir.exists():
        file_count = len(list(category_dir.glob("*.txt")))
        print(f"{category}: {file_count} files created in {category_dir}")
