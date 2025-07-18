# 딥 러닝 실체 (2025학년도 1학기)

![header](https://capsule-render.vercel.app/api?type=waving&color=auto&height=200&section=header&text=Deep%20Learning%20from%20Scratch&fontSize=50)

## 📖 리포지토리 소개

이 리포지토리는 **2025학년도 1학기 '딥 러닝 실체'** 수업의 학습 내용을 아카이빙하기 위해 만들어졌습니다.

본 과정의 목표는 TensorFlow나 PyTorch 같은 프레임워크의 편리한 기능 뒤에 숨겨진 딥러닝의 실제 동작 원리를 파헤치는 것입니다. 이를 위해 주 교재인 『밑바닥부터 시작하는 딥러닝 1 (리마스터판)』의 내용을 기반으로, 순수 Python과 NumPy를 사용하여 딥러닝의 핵심 알고리즘을 한 줄 한 줄 직접 구현하며 학습합니다.

---

## 🎯 학습 목표

- **딥러닝의 기본 개념 이해:** 퍼셉트론부터 신경망, CNN에 이르기까지 딥러닝의 핵심 구성 요소를 학습합니다.
- **내부 동작 원리 파악:** 손실 함수, 경사 하강법, 그리고 가장 중요한 **오차역전파법**의 원리를 '계산 그래프'를 통해 완벽하게 이해하고 직접 구현합니다.
- **Python 기반 구현 능력 향상:** 외부 라이브러리 의존성을 최소화하고 NumPy를 활용하여 딥러닝 모델을 처음부터 끝까지 구축하는 프로그래밍 역량을 강화합니다.
- **견고한 기초 확립:** 향후 복잡한 딥러닝 모델과 프레임워크를 더욱 깊이 있게 이해할 수 있는 단단한 수학적, 공학적 기반을 다집니다.

---

## 📚 주 교재 정보

| | |
| :--- | :--- |
| **도서명** | 밑바닥부터 시작하는 딥러닝 1 (리마스터판) |
| **저자** | 사이토 고키 |
| **출판사** | 한빛미디어 |
| **소개** | 딥러닝의 핵심을 이론과 함께 직접 구현하며 배우는 최고의 입문서 |

<p align="center">
  <img src="https://www.hanbit.co.kr/data/books/B8975299569_l.jpg" width="250">
</p>

---

## 🗂️ 디렉토리 구조 및 학습 내용

각 챕터별 디렉토리에는 핵심 이론 정리 노트와 실습 코드가 포함되어 있습니다.

- **`ch01_python_basics/`**: 파이썬 기초 및 NumPy, Matplotlib 사용법
- **`ch02_perceptron/`**: 퍼셉트론의 개념과 AND, OR, NAND 게이트 구현
- **`ch03_neural_network/`**: 신경망의 구조, 활성화 함수(Sigmoid, ReLU) 및 순전파 구현
- **`ch04_neural_network_learning/`**: 신경망 학습, 손실 함수, 수치 미분과 기울기
- **`ch05_backpropagation/`**: **(핵심)** 계산 그래프를 이용한 오차역전파법 원리 이해 및 구현
- **`ch06_learning_techniques/`**: 매개변수 최적화(Adam, SGD), 가중치 초기값, 과적합 방지(가중치 감소, 드롭아웃)
- **`ch07_cnn/`**: 합성곱 신경망(CNN), 합성곱 및 풀링 계층 구현
- **`ch08_deep_learning_summary/`**: 딥러닝의 역사, 응용 및 미래

---

## 💻 환경 설정 및 실행 방법

본 리포지토리의 코드는 **Google Colab** 또는 **로컬 환경**에서 실행할 수 있습니다.

### 1. Google Colab (권장)

리마스터판에서 제공하는 Colab 노트북을 활용하여 별도의 설정 없이 바로 실습을 진행할 수 있습니다. 각 챕터 폴더의 `.ipynb` 파일을 Colab에서 열어주세요.

### 2. 로컬 환경

```bash
# 1. 리포지토리 클론
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git)
cd YOUR_REPOSITORY

# 2. 필요한 라이브러리 설치 (가상환경 사용을 권장합니다)
pip install numpy matplotlib

# 3. 각 챕터의 파이썬 스크립트 실행
python ch03_neural_network/neuralnet_mnist.py
