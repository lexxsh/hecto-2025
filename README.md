# 제출 패키지 (Docker 이미지 포함)

이 폴더는 제출 요구사항 구조를 따릅니다. 또한, **빌드된 Docker 이미지 파일**(`gend-submission.tar`)을 포함합니다.

## 구조

- `model/model.pt`: 최종 추론용 모델 가중치
- `src/`: 소스 코드
- `config/config.yaml`: 학습/추론 설정
- `config/datasets/`: 데이터셋 목록(txt)
- `env/`: Docker 및 requirements
- `train_data/`: 학습 데이터
- `test_data/`: 평가 데이터
- `train.py`: 학습 엔트리포인트
- `inference.py`: 추론 엔트리포인트
- `gend-submission.tar`: **빌드된 Docker 이미지 파일 (제출용)**

## 데이터 경로

`config/datasets/*_data_deepfake.txt` 파일의 이미지 경로는 **상대경로**입니다. <br/>
즉, txt 파일 위치 기준으로 `train_data/df40-openfake_final/...` 을 찾도록 되어 있으며, 절대경로에 의존하지 않습니다.<br/>
학습에 사용한 데이터셋은 총 3종류로 [Deepfake-eval 2024](https://huggingface.co/datasets/nuriachandra/Deepfake-Eval-2024),[DF40](https://github.com/YZY-stack/DF40), 
[Openfake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) 에서 Sampling을 진행하였습니다. <br/>
주 train 데이터셋은 DF40으로 데이터 불균형을 해소하기 위해 생성방법별 약 3,000장을 선별하였으며, Validation set으로는 deepfake-eval의 일부, DF40의 생성방법별 일부를 사용하였습니다.

## 데이터 다운로드

용량 문제로 아래 파일은 GitHub에 포함하지 않았습니다. <br/>
Google Drive에서 다운로드 후 제출 폴더에 배치하세요.

- `train_data.zip` (학습 데이터) 설치 명령어 - `gdown 1zB_pUCPo5JeOhoDDoSWQjIjzuRYD6CF-`
- `gend-submission.tar` (빌드된 Docker 이미지) 설치 명령어 - `gdown 12XN1Ssmdte8rVCWORG5ngkxpkUfH95gh`
- `model.zip` (학습완료된 모델파일, 사전학습 가중치) 설치 명령어 - `gdown 1BwaYuwmkjFbOYLYpDF_rDuOS7YbBpmT8`
- `open.zip` (테스트 데이터) 설치 명령어 - `wget https://cfiles.dacon.co.kr/competitions/236628/open.zip`

gdown을 사용하면 빠르게 다운로드 받을 수 있으나, `Too many users....`와 같은 시도 관련 에러가 발생할 수 있으니 참고바랍니다.

다운로드 링크 (Google Drive - 전체):

```
https://drive.google.com/drive/folders/1OrYkeM9H293qAutUIiA-9uHEuo5c-w1X?usp=sharing
```

배치 위치:
- `train_data.zip` → 압축 해제 후 (`train_data/`)에 위치
  - 압축해제시 `train_data/df40-openfake_final` 형식의 구조로 되어야 합니다.
- `gend-submission.tar` → 프로젝트 루트 디렉터리에 위치
- `model.zip` → 압축 해제 후(`model/`)에 위치
- `open.zip` → 압축 해제 후(`test_data/`)에 위치

## Docker 이미지 제출

이미지는 아래 파일로 제출됩니다:

- `gend-submission.tar`

생성 명령:

```bash
docker save gend-submission -o gend-submission.tar
```

## Docker 이미지 로드 및 실행

> Docker 구축이 완전하지 않습니다. Docker가 오류가 뜨는 경우에는 가상환경 설치 후, 라이브러리 설치 후 로컬로 실행해주세요. (아래 [Docker 없이 직접 실행 (로컬)](#docker-없이-직접-실행-로컬) 참고)

### 로드

```bash
docker load -i gend-submission.tar
```

## Docker 없이 직접 실행 (로컬)

아래 명령은 Docker 없이 로컬 환경에서 실행하는 방법입니다. <br/>
Python 및 라이브러리는 가상환경 생성 후 `pip install -r requirements.txt`로 설치하세요. <br/>conda 환경을 사용하는 경우 `conda env create -f env/environment.yml`로 설치할 수 있습니다.

### 모델 구조

모델구조는 다음과 같습니다.

Perception Encoder(`vit_pe_core_large_patch14_336`, ViT)를 backbone으로 사용합니다. 초기에는 backbone을 freeze하고 Linear Probe head만 학습하는 방식을 시도하였으나 성능 한계(AUC 91)가 있었습니다. 이를 개선하기 위해 **LN-tuning** 방식을 적용하였습니다. [위 코드](https://github.com/yermandy/GenD)를 베이스로 수정하였으며, 학습블록, 레이어를 변경하며 최적의 학습 레이어로 업데이트하였습니다..

**LN-tuning**은 backbone의 Layer Normalization 블록의 affine 파라미터(scale, shift)만 학습하는 PEFT 기법으로, 전체 파라미터의 약 0.03%만 업데이트합니다. FFT나 LoRA는 소규모 데이터에서 빠르게 overfitting 되어, 점수의 한계가 나타났으나 이 방법론을 사용한 뒤 94+를 달성할 수 있었습니다.

이에 특성 벡터를 L2 정규화하여 unit hypersphere 위에 매핑한 뒤, Alignment Loss(같은 클래스 간 거리 최소화)와 Uniformity Loss(특성의 균등 분포)를 Cross-Entropy Loss와 함께 사용하여 일반화 성능을 강화합니다. 추론 시에는 softmax를 적용하여 Fake 확률을 출력합니다.

### 학습 실행 예시 (로컬)

```bash
python train.py df40-openfake_final
```

### 추론 실행 예시 (로컬)

```bash
python inference.py --input_folder ./test_data/test_data --output_csv ./output.csv
```

### 학습 실행 예시 (도커 활용)

도커가 오류가 뜨는 경우에는 가상환경설치 후 로컬로 실행해주세요

```bash
docker run --rm --gpus all \
    --entrypoint python \
    -v $(pwd)/src:/workspace/src \
    -v $(pwd)/model:/workspace/model \
    -v $(pwd)/train_data:/workspace/train_data \
    -v $(pwd)/runs:/workspace/runs \
    gend-submission \
    train.py df40-openfake_final
```

### 추론 실행 예시 (도커 활용)

RetinaFace 모델이 없는 경우 아래 명령어로 미리 다운로드하세요:

```bash
mkdir -p weights/models/buffalo_l
wget https://huggingface.co/datasets/theanhntp/Liblib/resolve/ae4357741af379482690fe3e0f2fa6fd32ba33b4/insightface/models/buffalo_l/det_10g.onnx -O weights/models/buffalo_l/det_10g.onnx
```

```bash
docker run --rm --gpus all \
    -v $(pwd)/src:/workspace/src \
    -v $(pwd)/model:/workspace/model \
    -v $(pwd)/weights:/workspace/weights \
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

