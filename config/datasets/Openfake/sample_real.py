import random
from pathlib import Path

# 랜덤 시드 고정
random.seed(42)

# 파일 경로
source_file = Path("/home/gcl/lexxsh/deepfake/GenD/config/datasets/Openfake/dalle-3.txt")
output_file = Path("/home/gcl/lexxsh/deepfake/GenD/config/datasets/Openfake/dalle-3_sampled.txt")

# 샘플링 개수 설정
SAMPLE_COUNT = 3000  # 원하는 샘플 개수로 변경 가능

print(f"Reading from: {source_file}")

# 파일 읽기
with open(source_file, 'r') as f:
    lines = [line.strip() for line in f if line.strip()]

total_count = len(lines)
print(f"Total lines: {total_count}")

# 샘플링 개수 결정
sample_count = min(SAMPLE_COUNT, total_count)

# 랜덤 샘플링
sampled_lines = random.sample(lines, sample_count)

# 새 파일에 저장
with open(output_file, 'w') as f:
    f.write('\n'.join(sampled_lines))
    if sampled_lines:
        f.write('\n')

print(f"✓ Sampled {sample_count}/{total_count} lines")
print(f"✓ Saved to: {output_file}")
