import random
from pathlib import Path

# 랜덤 시드 고정
random.seed(42)

base_dir = Path("/home/gcl/lexxsh/deepfake/GenD/config/datasets/DF40")

# 각 폴더별 fake 개수
fake_counts = {
    "FS": 9000,
    "FR": 13000,
    "EFS": 9530,
    "FE": 4900
}

# 전체 fake 개수
total_fake = sum(fake_counts.values())
print(f"Total fake count: {total_fake}")

# real_FF.txt 읽기
real_ff_file = base_dir / "real_FF.txt"
with open(real_ff_file, 'r') as f:
    real_lines = [line.strip() for line in f if line.strip()]

total_real = len(real_lines)
print(f"Total real count: {total_real}")

# 각 폴더별로 비례 분배
print("\n=== Proportional Distribution ===")

# 랜덤 셔플
random.shuffle(real_lines)

start_idx = 0
for category, fake_count in fake_counts.items():
    # 비례 계산
    proportion = fake_count / total_fake
    sample_count = int(total_real * proportion)
    
    # 실제 샘플링
    sampled_real = real_lines[start_idx:start_idx + sample_count]
    start_idx += sample_count
    
    # 폴더에 저장
    category_dir = base_dir / category
    category_dir.mkdir(exist_ok=True)
    
    output_file = category_dir / "real_FF.txt"
    with open(output_file, 'w') as f:
        f.write('\n'.join(sampled_real))
        if sampled_real:
            f.write('\n')
    
    print(f"{category}: {fake_count} fake → {sample_count} real (ratio: {proportion:.3f})")
    print(f"  Saved to {category}/real_FF.txt")

print(f"\nTotal distributed: {start_idx}/{total_real}")
print(f"Remaining: {total_real - start_idx}")
