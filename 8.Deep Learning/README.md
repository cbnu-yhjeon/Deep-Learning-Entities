## 파일 설명
| 파일명 | 파일 용도 | 관련 절 | 페이지 |
|:--   |:--      |:--    |:--      |
| awesome_net.py | 빈 파일입니다. 여기에 여러분만의 멋진 신경망을 구현해보세요! |  |  |
| deep_convnet.py | [그림 8-1]의 깊은 신경망을 구현한 소스입니다. | 8.1.1 더 깊은 신경망으로 | 262 |
| deep_convnet_params.pkl | deep_convnet.py용 학습된 가중치입니다. |  |  |
| half_float_network.py | 수치 정밀도를 반정밀도(16비트)로 낮춰 계산하여 배정밀도(64비트)일 때와 정확도를 비교해본다. | 8.3.4 연산 정밀도와 비트 줄이기 | 278 |
| misclassified_mnist.py | 이번 장에서 구현한 신경망이 인식에 실패한 손글씨 이미지들을 화면에 보여줍니다. | 8.1.1 더 깊은 신경망으로 | 263 |
| train_deepnet.py | deep_convnet.py의 신경망을 학습시킵니다. 몇 시간은 걸리기 때문에 다른 코드에서는 미리 학습된 가중치인 deep_convnet_params.pkl을 읽어서 사용합니다. | 8.1.1 더 깊은 신경망으로 | 262 |

## 8장 딥러닝
딥러닝은 층을 깊게 한 심층 신경망입니다. 심층 신경망은 지금까지 설명한 신경망을 바탕으로 뒷단에 층을 추가하기만 하면 만들 수 있지만, 커다란 문제가 몇 개 있습니다. 이번 장에서는 딥러닝의 특징과 과제, 그리고 가능성을 살펴봅니다. 또 오늘날의 첨단 딥러닝에 대한 설명도 준비했습니다.

## 목차
```
8.1 더 깊게 
__8.1.1 더 깊은 네트워크로 
__8.1.2 정확도를 더 높이려면 
__8.1.3 깊게 하는 이유 
8.2 딥러닝의 초기 역사 
__8.2.1 이미지넷 
__8.2.2 VGG 
__8.2.3 GoogLeNet 
__8.2.4 ResNet 
8.3 더 빠르게(딥러닝 고속화) 
__8.3.1 풀어야 할 숙제 
__8.3.2 GPU를 활용한 고속화 
__8.3.3 분산 학습 
__8.3.4 연산 정밀도와 비트 줄이기 
8.4 딥러닝의 활용 
__8.4.1 사물 검출 
__8.4.2 분할 
__8.4.3 사진 캡션 생성 
8.5 딥러닝의 미래 
__8.5.1 이미지 스타일(화풍) 변환 
__8.5.2 이미지 생성 
__8.5.3 자율 주행 
__8.5.4 Deep Q-Network(강화학습) 
```

## 이번 장에서 배운 내용
* 수많은 문제에서 신경망을 더 깊게 하여 성능을 개선할 수 있다.
* 이미지 인식 기술 대회인 ILSVRC에서는 최근 딥러닝 기반 기법이 상위권을 독점하고 있으며, 그 깊이도 더 깊어지는 추세다.
* 유명한 신경망으로는 VGG, GoogLeNet, ResNet이 있다.
* GPU와 분산 학습, 비트 정밀도 감소 등으로 딥러닝을 고속화할 수 있다.
* 딥러닝(신경망)은 사물 인식뿐 아니라 사물 검출과 분할에도 이용할 수 있다.
* 딥러닝의 응용 분야로는 사진의 캡션 생성, 이미지 생성, 강화학습 등이 있다. 최근에는 자율 주행에도 딥러닝을 접목하고 있어 기대된다.
