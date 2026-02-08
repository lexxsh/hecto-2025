import os
import random
from pathlib import Path

# 랜덤 시드 고정 (재현성)
random.seed(42)

# 현재 디렉토리
base_dir = Path("/home/gcl/lexxsh/deepfake/GenD/config/datasets/DF40")
val_per_model = 100

# Validation 파일 초기화
val_fake_file = base_dir / "validation_fake.txt"
val_real_file = base_dir / "validation_real.txt"

val_fake_file.write_text("")
val_real_file.write_text("")

val_fake_lines = []
val_real_lines = []

# 모든 txt 파일 처리
txt_files = sorted(base_dir.glob("*.txt"))

for txt_file in txt_files:
    filename = txt_file.name
    
    # validation, real_FF 파일은 스킵
    if filename.startswith("validation_") or filename == "real_FF.txt":
        continue
    
    print(f"Processing {filename}...")
    
    # 파일 읽기
    with open(txt_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    total_lines = len(lines)
    
    if total_lines == 0:
        print(f"  - {filename}: No data, skipping")
        continue
    
    # 추출할 개수 결정 (최대 100개)
    num_to_extract = min(val_per_model, total_lines)
    
    # 랜덤 샘플링
    sampled_lines = random.sample(lines, num_to_extract)
    remaining_lines = [line for line in lines if line not in sampled_lines]
    
    # real/fake 분류
    is_real = ("real" in filename.lower() and "whichisreal" not in filename.lower()) or filename == "heygen_real.txt"
    is_fake = "fake" in filename.lower() or "ff" in filename.lower() or "FF" in filename
    
    # whichisreal은 별도 처리
    if "whichisreal" in filename.lower():
        if "real" in filename.lower():
            is_real = True
            is_fake = False
        else:
            is_real = False
            is_fake = True
    
    # validation에 추가
    if is_real:
        val_real_lines.extend(sampled_lines)
        target = "real"
    elif is_fake:
        val_fake_lines.extend(sampled_lines)
        target = "fake"
    else:
        print(f"  - {filename}: Could not determine real/fake, skipping")
        continue
    
    print(f"  - Extracted {num_to_extract}/{total_lines} to validation_{target}.txt")
    
    # 원본 파일 업데이트 (남은 라인만 저장)
    with open(txt_file, 'w') as f:
        f.write('\n'.join(remaining_lines))
        if remaining_lines:
            f.write('\n')
    
    print(f"  - Updated {filename} with {len(remaining_lines)} remaining lines")

# real_FF.txt에서 fake와 동일한 개수만큼 추출
real_ff_file = base_dir / "real_FF.txt"
if real_ff_file.exists():
    print(f"\nProcessing real_FF.txt to match fake count...")
    
    with open(real_ff_file, 'r') as f:
        real_ff_lines = [line.strip() for line in f if line.strip()]
    
    # fake 개수와 동일하게 맞추기
    num_fake = len(val_fake_lines)
    num_real_current = len(val_real_lines)
    num_needed = num_fake - num_real_current
    
    if num_needed > 0:
        num_to_extract = min(num_needed, len(real_ff_lines))
        sampled_real = random.sample(real_ff_lines, num_to_extract)
        val_real_lines.extend(sampled_real)
        
        remaining_real = [line for line in real_ff_lines if line not in sampled_real]
        
        print(f"  - Extracted {num_to_extract} from real_FF.txt")
        
        # real_FF.txt 업데이트
        with open(real_ff_file, 'w') as f:
            f.write('\n'.join(remaining_real))
            if remaining_real:
                f.write('\n')
        
        print(f"  - Updated real_FF.txt with {len(remaining_real)} remaining lines")

# Validation 파일 저장
with open(val_fake_file, 'w') as f:
    f.write('\n'.join(val_fake_lines))
    if val_fake_lines:
        f.write('\n')

with open(val_real_file, 'w') as f:
    f.write('\n'.join(val_real_lines))
    if val_real_lines:
        f.write('\n')

print(f"\n=== Summary ===")
print(f"Validation Fake: {len(val_fake_lines)} images")
print(f"Validation Real: {len(val_real_lines)} images")
print(f"Total Validation: {len(val_fake_lines) + len(val_real_lines)} images")
print(f"\nValidation files created:")
print(f"  - {val_fake_file}")
print(f"  - {val_real_file}")
