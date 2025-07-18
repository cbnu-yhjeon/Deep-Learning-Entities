## 파일 설명
| 파일명 | 파일 용도 | 관련 절 | 페이지 |
|:--   |:--      |:--    |:--      |
| mnist_show.py | MNIST 데이터셋을 읽어와 훈련 데이터 중 0번째 이미지를 화면에 출력합니다. | 3.6.1 손글씨 데이터셋 | 99 |
| neuralnet_mnist.py | 신경망으로 손글씨 숫자 그림을 추론합니다. 입력층, 은닉층1, 은닉층2, 출력층의 뉴런 수는 각각 784, 50, 100, 10입니다. | 3.6.2 신경망의 추론 처리 | 100 |
| neuralnet_mnist_batch.py | neuralnet_mnist.py에 배치 처리 기능을 더했습니다. | 3.6.3 배치 처리 | 104 |
| relu.py | ReLU 함수를 구현한 코드입니다. | 3.2.7 ReLU 함수 | 76 |
| sample_weight.pkl | 미리 학습해둔 가중치 매개변수의 값들입니다. | 3.6.2 신경망의 추론 처리 | 100 |
| sig_step_compare.py | 시그모이드 함수와 계단 함수의 그래프 모양을 비교해봅니다. | 3.2.5 시그모이드 함수와 계단 함수 비교 | 74 |
| sigmoid.py | 시그모이드 함수를 구현한 코드입니다. | 3.2.4 시그모이드 함수 구현하기 | 72 |
| step_function.py | 계단 함수를 구현한 코드입니다. | 3.2.3 계단 함수의 그래프 | 70 |

## 3장 신경망
앞 장 배운 퍼셉트론 관련해서는 좋은 소식과 나쁜 소식이 있었습니다. 좋은 소식은 퍼셉트론으로 복잡한 함수도 표현할 수 있다는 것입니다. 그 예로 컴퓨터가 수행하는 복잡한 처리도 퍼셉트론으로 (이론상) 표현할 수 있음을 앞 장에서 설명했습니다. 나쁜 소식은 가중치를 설정하는 작업(원하는 결과를 출력하도록 가중치 값을 적절히 정하는 작업)은 여전히 사람이 수동으로 한다는 것입니다. 앞 장에서는 AND, OR 게이트의 진리표를 보면서 우리 인간이 적절한 가중치 값을 정했습니다.

신경망은 이 나쁜 소식을 해결해줍니다. 무슨 말인고 하니, 가중치 매개변수의 적절한 값을 데이터로부터 자동으로 학습하는 능력이 이제부터 살펴볼 신경망의 중요한 성질입니다. 이번 장에서는 신경망의 개요를 설명하고, 신경망이 입력 데이터가 무엇인지 식별하는 처리 과정을 자세히 알아봅니다. 아쉽지만 데이터에서 가중치 매개변수 값을 학습하는 방법은 다음 장까지 기다리셔야 합니다.

## 목차
```
3.1 퍼셉트론에서 신경망으로 
__3.1.1 신경망의 예 
__3.1.2 퍼셉트론 복습 
__3.1.3 활성화 함수의 등장 
3.2 활성화 함수 
__3.2.1 시그모이드 함수 
__3.2.2 계단 함수 구현하기 
__3.2.3 계단 함수의 그래프 
__3.2.4 시그모이드 함수 구현하기 
__3.2.5 시그모이드 함수와 계단 함수 비교 
__3.2.6 비선형 함수 
__3.2.7 ReLU 함수 
3.3 다차원 배열의 계산 
__3.3.1 다차원 배열 
__3.3.2 행렬의 내적 
__3.3.3 신경망의 내적 
3.4 3층 신경망 구현하기 
__3.4.1 표기법 설명 
__3.4.2 각 층의 신호 전달 구현하기 
__3.4.3 구현 정리 
3.5 출력층 설계하기 
__3.5.1 항등 함수와 소프트맥스 함수 구현하기 
__3.5.2 소프트맥스 함수 구현 시 주의점 
__3.5.3 소프트맥스 함수의 특징 
__3.5.4 출력층의 뉴런 수 정하기
3.6 손글씨 숫자 인식 
__3.6.1 MNIST 데이터셋 
__3.6.2 신경망의 추론 처리 
__3.6.3 배치 처리 
```

## 이번 장에서 배운 내용
* 신경망에서는 활성화 함수로 시그모이드 함수와 ReLU 함수 같은 매끄럽게 변화하는 함수를 이용한다.
* 넘파이의 다차원 배열을 잘 사용하면 신경망을 효율적으로 구현할 수 있다.
* 기계학습 문제는 크게 회귀와 분류로 나눌 수 있다.
* 출력층의 활성화 함수로는 회귀에서는 주로 항등 함수를, 분류에서는 주로 소프트맥스 함수를 이용한다.
* 분류에서는 출력층의 뉴런 수를 분류하려는 클래스 수와 같게 설정한다.
* 입력 데이터를 묶은 것을 배치라 하며, 추론 처리를 이 배치 단위로 진행하면 결과를 훨씬 빠르게 얻을 수 있다.
