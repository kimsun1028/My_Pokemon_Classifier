# 포켓몬 이미지 분류기 (Pokemon Image Classifier)

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

## 필수 요구사항

### 1. 필수 기능 (20점)
- ✅ 4개 이상의 서로 다른 분류기 구현
  - AlexNet (Pretrained weights 사용)
  - VGGNet (Fine-tuning)
  - GoogleNet (Pretrained 활용)
  - ResNet50 (Pretrained weights 사용)
- ✅ 각 모델별 성능 비교 (Test Recall, Test Precision, F1-Score, Accuracy 등)
- ✅ 성능 지표 및 설정 기록

### 2. 실험 결과 문서화 (5점)
- ✅ README.md에 성능 결과 기록
- ✅ 학습 곡선 그래프 포함
- ✅ 각 모델의 예제 결과

### 3. 데모 GUI 추가 (5점)
- ✅ Streamlit 사용하여 웹 인터페이스 구축
- ✅ 테스트 이미지 업로드 기능
- ✅ 실시간 예측 결과 표시

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
python src/train.py --model alexnet --epochs 50 --batch_size 32
python src/train.py --model vggnet --epochs 50 --batch_size 32
python src/train.py --model googlenet --epochs 50 --batch_size 32
python src/train.py --model resnet --epochs 50 --batch_size 32
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

## 라이선스

이 프로젝트는 교육 목적으로 작성되었습니다.
