# 제출 패키지 (Docker 이미지 포함)

이 폴더는 제출 요구사항 구조를 따릅니다. 또한, **빌드된 Docker 이미지 파일**(`gend-submission.tar`)을 포함합니다.

## 구조

- `model/model.pt`: 최종 추론용 모델 가중치
- `src/`: GenD 소스 코드
- `config/config.yaml`: 학습/추론 설정
- `config/datasets/`: 데이터셋 목록(txt)
- `env/`: Docker 및 requirements
- `train_data/`: 학습 데이터 (현재 포함됨)
- `test_data/`: 평가 데이터 (필요 시 별도 복사)
- `train.py`: 학습 엔트리포인트
- `inference.py`: 추론 엔트리포인트
- `gend-submission.tar`: **빌드된 Docker 이미지 파일 (제출용)**

## 데이터 경로

`config/datasets/*_data_deepfake.txt` 파일의 이미지 경로는 **상대경로**입니다.
즉, txt 파일 위치 기준으로 `train_data/df40-openfake_final/...` 을 찾도록 되어 있으며, 절대경로에 의존하지 않습니다.

## Docker 이미지 제출

이미지는 아래 파일로 제출됩니다:

- `gend-submission.tar`

생성 명령:

```bash
docker save gend-submission -o gend-submission.tar
```

## Docker 이미지 로드 및 실행

### 로드

```bash
docker load -i gend-submission.tar
```

## Docker 없이 직접 실행 (로컬)

아래 명령은 Docker 없이 로컬 환경에서 실행하는 방법입니다.
Python 및 라이브러리는 `env/requirements.txt` 또는 `env/environment.yml`로 설치하세요.

### 학습 실행 예시 (로컬)

```bash
python train.py df40-openfake_final
```

### 추론 실행 예시 (로컬)

```bash
python inference.py --input_folder ./test_data --output_csv ./output.csv
```

### 학습 실행 예시

```bash
docker run --rm --gpus all \
  -v $(pwd)/train_data:/workspace/train_data \
  -v $(pwd)/runs:/workspace/runs \
  gend-submission \
  python train.py df40-openfake_final
```

### 추론 실행 예시

```bash
docker run --rm --gpus all \
  -v $(pwd)/test_data:/workspace/test_data \
  -v $(pwd)/output:/workspace/output \
  gend-submission \
  --input_folder /workspace/test_data \
  --output_csv /workspace/output/output.csv
```

## Dockerfile / Requirements

제출 필수 파일은 `env/`에 포함되어 있습니다.

- `env/Dockerfile`
- `env/requirements.txt`
- `env/environment.yml`

## 데이터 다운로드

용량 문제로 아래 파일은 GitHub에 포함하지 않았습니다.
Google Drive에서 다운로드 후 제출 폴더에 배치하세요.

- `train_data.zip` (학습 데이터)
- `gend-submission.tar` (빌드된 Docker 이미지)
- `model.zip` (학습완료된 모델파일, 사전학습 가중치)

다운로드 링크 (Google Drive):

```
https://drive.google.com/drive/folders/1OrYkeM9H293qAutUIiA-9uHEuo5c-w1X?usp=sharing
```

배치 위치:
- `train_data.zip` → 압축 해제 후 `train_data/`에 위치
- `gend-submission.tar` → 루트 디렉터리(`your_submission/`)에 위치
- `model.zip` → 압축 해제 후(`model/`)에 위치
