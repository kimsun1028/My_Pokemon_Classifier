# 포켓몬 이미지 분류기 (Pokemon Image Classifier)

딥러닝을 활용한 포켓몬 이미지 자동 분류 시스템입니다. CNN 기반의 여러 모델을 비교하고, 최고 성능 모델을 선택하여 웹 인터페이스로 제공합니다.

## 목표
주어진 포켓몬 이미지에서 포켓몬의 이름을 정확하게 맞추는 다중 분류 모델 구현 및 비교

## 데이터셋
- **[7,000 Labeled Pokemon](https://www.kaggle.com/datasets/ameeshs/pokemon-image-dataset)** (Kaggle)
- 클래스 수: 150개 포켓몬
- 이미지 형식: PNG, JPG
- 이미지 크기: 대부분 120x120 ~ 200x200 pixels
- 데이터 분할: 학습(70%), 검증(15%), 테스트(15%)

## 프로젝트 구조
```
My_Pokemon_Classifier/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/              # 원본 데이터셋
│   └── processed/        # 전처리된 데이터
├── src/
│   ├── data_loader.py    # 데이터 로드 및 전처리
│   ├── models/           # 모델 구현체
│   │   ├── resnet_classifier.py
│   │   ├── efficientnet_classifier.py
│   │   ├── custom_cnn.py
│   │   └── vit_classifier.py
│   ├── train.py          # 학습 스크립트
│   └── evaluate.py       # 평가 스크립트
├── notebooks/
│   └── experiment_results.ipynb  # 실험 결과 분석
├── app/
│   └── streamlit_demo.py # 웹 인터페이스 데모
└── results/              # 학습 결과 및 모델
```


## 설치 및 실행

### 1. 환경 설정
```bash
# 패키지 설치
pip install -r requirements.txt

# 또는 conda 사용 (권장)
conda create -n pokemon_classifier python=3.10
conda activate pokemon_classifier
pip install -r requirements.txt
```

### 2. 데이터 준비
```bash
# Kaggle에서 데이터셋 다운로드
# data/raw/ 폴더에 이미지 저장
```

### 3. 모델 학습
```bash
python src/train.py --model alexnet --epochs 1 --batch_size 32
python src/train.py --model vggnet --epochs 1 --batch_size 32
python src/train.py --model googlenet --epochs 1 --batch_size 32
python src/train.py --model resnet --epochs 1 --batch_size 32
```

### 4. 모델 평가
```bash
python src/evaluate.py --model alexnet
python src/evaluate.py --model vggnet
python src/evaluate.py --model googlenet
python src/evaluate.py --model resnet
```

### 5. Streamlit 데모 실행
```bash
streamlit run app/streamlit_demo.py
```

## 모델 성능 비교

### 학습된 모델들
- **AlexNet**: 가벼운 CNN 아키텍처, 빠른 학습
- **VGGNet**: 깊은 신경망, 높은 정확도 but 느린 학습
- **GoogleNet (Inception)**: 다중 스케일 특징 추출
- **ResNet50**: 잔차 연결을 이용한 깊은 신경망

### 성능 지표

| 모델 | 학습 정확도 | 검증 정확도 | 테스트 정확도 | 학습시간 |
|------|----------|----------|----------|---------|
| AlexNet | TBD | TBD | TBD | ~5분 |
| VGGNet | TBD | TBD | TBD | ~15분 |
| GoogleNet | TBD | TBD | TBD | ~10분 |
| ResNet50 | TBD | TBD | TBD | ~8분 |

*성능 지표는 학습 후 업데이트됩니다*

## 학습 곡선 및 분석

학습 과정에서 나타나는 정확도와 손실값 변화를 시각화한 곡선들을 확인할 수 있습니다:

- **정확도 곡선**: 에포크별 학습 및 검증 정확도 추이
- **손실값 곡선**: 에포크별 학습 및 검증 손실값 추이

이들 자료는 `notebooks/experiment_results.ipynb`에서 확인할 수 있습니다. 각 모델의 학습 히스토리는 `results/` 폴더의 JSON 파일에서 로드됩니다.

## 주의사항

- ⚠️ **데이터셋 다운로드**: Kaggle에서 별도로 다운로드하여 `data/raw/` 폴더에 저장해야 합니다
- ⚠️ **GPU 권장**: GPU 사용 시 학습 시간이 대폭 단축됩니다 (CUDA 11.8 이상)
- ⚠️ **메모리 요구량**: VGGNet 모델은 GPU 메모리 8GB 이상 권장
- ⚠️ **첫 실행**: 첫 모델 학습 시 모델 가중치 다운로드로 시간이 소요될 수 있습니다

## 참고 자료

- [PyTorch Documentation](https://pytorch.org/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Timm Models](https://github.com/rwightman/pytorch-image-models)
- [Streamlit Documentation](https://docs.streamlit.io/)

