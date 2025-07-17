# 🧠 딥러닝 (Deep Learning) 실습 저장소

충북대학교 OOOO년 O학기 '딥러닝' 과목을 수강하며 진행한 모든 실습 코드와 결과물을 아카이빙하는 저장소입니다. 딥러닝의 기초적인 신경망부터 CNN, RNN 등 주요 모델의 구조를 이해하고 직접 구현하는 것을 목표로 합니다.

## 📚 과목 정보 (Course Information)

- **과목명**: 딥러닝 (Deep Learning)
- **수강 학기**: 2026년 1학기
- **주요 사용 언어**: `Python`
- **핵심 라이브러리**: `PyTorch`, `NumPy`, `Matplotlib`, `Scikit-learn`

---

## 📂 실습 내용 (Lab Contents)

각 디렉터리는 딥러닝의 핵심적인 모델 또는 개념을 다루고 있습니다. 제목을 클릭하면 해당 실습 폴더로 이동합니다.

### 1. [Week01_Neural-Network-Basics](<./Week01_Neural-Network-Basics>)
- **설명**: 딥러닝의 가장 기본 단위인 인공 신경망(ANN)의 구조를 이해하고, 다층 퍼셉트론(MLP)을 직접 구현합니다. 활성화 함수와 손실 함수의 역할을 학습했습니다.
- **주요 개념**: `Perceptron`, `Multi-Layer Perceptron (MLP)`, `Activation Function (Sigmoid, ReLU)`, `Loss Function`, `Gradient Descent`

### 2. [Week02_Optimization-and-Regularization](<./Week02_Optimization-and-Regularization>)
- **설명**: 딥러닝 모델의 학습을 안정시키고 성능을 향상시키는 다양한 최적화 및 정규화 기법들을 학습합니다. 경사 하강법의 한계를 개선하는 옵티마이저들과 과적합을 방지하는 방법들을 실습했습니다.
- **주요 개념**: `Optimizers (SGD, Adam, RMSprop)`, `Batch Normalization`, `Dropout`, `Weight Decay`

### 3. [Week03_Convolutional-Neural-Networks](<./Week03_Convolutional-Neural-Networks>)
- **설명**: 이미지 데이터 처리에 뛰어난 성능을 보이는 합성곱 신경망(CNN)을 구현합니다. 컨볼루션과 풀링 연산의 원리를 이해하고, CIFAR-10 데이터셋으로 이미지 분류 모델을 학습시켰습니다.
- **주요 개념**: `Convolution`, `Pooling`, `Padding`, `Stride`, `LeNet`, `VGG`, `ResNet`

### 4. [Week04_Recurrent-Neural-Networks](<./Week04_Recurrent-Neural-Networks>)
- **설명**: 순차(Sequential) 데이터 처리에 적합한 순환 신경망(RNN)의 구조를 학습합니다. RNN의 장기 의존성 문제를 해결하기 위한 LSTM과 GRU 모델을 구현하여 텍스트 생성 실습을 진행했습니다.
- **주요 개념**: `Recurrent Neural Network (RNN)`, `Vanishing/Exploding Gradients`, `LSTM (Long Short-Term Memory)`, `GRU (Gated Recurrent Unit)`

### 5. [Week05_Generative-Models](<./Week05_Generative-Models>)
- **설명**: 기존 데이터를 학습하여 새로운 데이터를 생성하는 생성 모델의 기초를 학습합니다. Autoencoder를 이용한 차원 축소와 GAN을 이용한 간단한 이미지 생성 실습을 진행했습니다.
- **주요 개념**: `Autoencoder (AE)`, `Variational Autoencoder (VAE)`, `Generative Adversarial Network (GAN)`

### 6. [Week06_Attention-and-Transformer](<./Week06_Attention-and-Transformer>)
- **설명**: 현대 자연어 처리의 핵심인 어텐션 메커니즘과 트랜스포머 모델의 구조를 학습합니다. Self-Attention의 원리를 이해하고, 간단한 시퀀스 변환 모델을 구현했습니다.
- **주요 개념**: `Seq2Seq`, `Attention Mechanism`, `Self-Attention`, `Transformer`

---

## 🚀 실행 방법 (How to Run)

각 실습은 독립적인 스크립트로 구성되어 있습니다.

1.  프로젝트 실행에 필요한 라이브러리를 설치합니다. 가상 환경 구성을 권장합니다.
    ```bash
    pip install -r requirements.txt
    ```
    > **`requirements.txt` 예시:**
    > ```txt
    > torch
    > torchvision
    > numpy
    > matplotlib
    > scikit-learn
    > ```

2.  원하는 실습의 디렉터리로 이동합니다.
    ```bash
    cd <실습_폴더명>  # 예: cd Week03_Convolutional-Neural-Networks
    ```

3.  폴더 내의 파이썬 스크립트(`*.py` 또는 `*.ipynb`)를 실행하여 모델 학습 및 평가를 진행합니다.

## 📝 정리 및 후기

'딥러닝' 과목을 통해 현대 인공지능의 근간이 되는 다양한 신경망 모델들의 구조와 원리를 깊이 있게 탐구할 수 있었습니다. 특히 복잡한 이론을 파이토치(PyTorch)를 이용해 직접 코드로 구현하고, 데이터가 모델을 통과하며 학습되는 과정을 눈으로 확인하는 경험은 매우 가치 있었습니다.

---
*Made by **전양호***
