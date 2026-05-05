# 포켓몬 이미지 분류 프로젝트 (Pokemon Image Classifier)

## 목표
주어진 포켓몬 이미지에서 포켓몬의 이름을 맞추는 다중 분류 모델 구현

## 데이터셋
- **[7,000 Labeled Pokemon](https://www.kaggle.com/datasets/ameeshs/pokemon-image-dataset)** (Kaggle)
- 클래스 수: 150개 포켓몬
- 이미지 형식: PNG, JPG

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

| 모델 | Accuracy | Precision | Recall | F1-Score | 학습시간 |
|------|----------|-----------|--------|----------|---------|
| AlexNet | - | - | - | - | - |
| VGGNet | - | - | - | - | - |
| GoogleNet | - | - | - | - | - |
| ResNet50 | - | - | - | - | - |

*실험 결과는 학습 후 업데이트 예정*

## 학습 곡선

실험 결과는 `notebooks/experiment_results.ipynb`에서 확인하실 수 있습니다.

## 주의사항

- ⚠️ 데이터셋은 별도로 Kaggle에서 다운로드 필요
- ⚠️ GPU 사용 권장 (CUDA 11.8 이상)
- ⚠️ VGGNet 모델은 메모리 요구량이 많음

## 참고 자료

- [PyTorch Documentation](https://pytorch.org/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Timm Models](https://github.com/rwightman/pytorch-image-models)
- [Streamlit Documentation](https://docs.streamlit.io/)

