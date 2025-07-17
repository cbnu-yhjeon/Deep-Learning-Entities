import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 데이터 준비
# make_moons 함수를 사용하여 비선형 데이터 생성 (과적합 유도에 용이)
X, y = make_moons(n_samples=500, noise=0.25, random_state=42)

# 데이터 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 훈련 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 2. 모델 정의

# (1) 과적합 모델 (드롭아웃 없음)
def create_overfit_model():
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid') # 이진 분류를 위한 sigmoid 활성화 함수
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# (2) 드롭아웃 적용 모델
def create_dropout_model(dropout_rate=0.5):
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dropout(dropout_rate), # 드롭아웃 레이어 추가
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout_rate), # 드롭아웃 레이어 추가
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid') # 이진 분류를 위한 sigmoid 활성화 함수
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# 3. 모델 훈련
epochs = 100
batch_size = 32

# 과적합 모델 훈련
overfit_model = create_overfit_model()
print("\n--- 과적합 모델 훈련 시작 ---")
history_overfit = overfit_model.fit(X_train, y_train,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    validation_data=(X_test, y_test),
                                    verbose=0) # verbose=0으로 설정하여 훈련 과정 출력 생략

# 드롭아웃 모델 훈련
dropout_model = create_dropout_model(dropout_rate=0.3) # 드롭아웃 비율 설정
print("\n--- 드롭아웃 적용 모델 훈련 시작 ---")
history_dropout = dropout_model.fit(X_train, y_train,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    validation_data=(X_test, y_test),
                                    verbose=0) # verbose=0으로 설정하여 훈련 과정 출력 생략

print("\n--- 모델 훈련 완료 ---")

# 4. 결과 시각화

# 손실 비교
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_overfit.history['loss'], label='Overfit Model - Training Loss')
plt.plot(history_overfit.history['val_loss'], label='Overfit Model - Validation Loss')
plt.plot(history_dropout.history['loss'], label='Dropout Model - Training Loss', linestyle='--')
plt.plot(history_dropout.history['val_loss'], label='Dropout Model - Validation Loss', linestyle='--')
plt.title('Model Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 정확도 비교
plt.subplot(1, 2, 2)
plt.plot(history_overfit.history['accuracy'], label='Overfit Model - Training Accuracy')
plt.plot(history_overfit.history['val_accuracy'], label='Overfit Model - Validation Accuracy')
plt.plot(history_dropout.history['accuracy'], label='Dropout Model - Training Accuracy', linestyle='--')
plt.plot(history_dropout.history['val_accuracy'], label='Dropout Model - Validation Accuracy', linestyle='--')
plt.title('Model Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# (선택 사항) 결정 경계 시각화 함수
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k', cmap=plt.cm.RdYlBu)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

# 결정 경계 시각화
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plot_decision_boundary(overfit_model, X_test, y_test, 'Overfit Model Decision Boundary')

plt.subplot(1, 2, 2)
plot_decision_boundary(dropout_model, X_test, y_test, 'Dropout Model Decision Boundary')

plt.tight_layout()
plt.show()

# 모델 평가
loss_overfit, accuracy_overfit = overfit_model.evaluate(X_test, y_test, verbose=0)
print(f"\n과적합 모델 테스트 정확도: {accuracy_overfit:.4f}, 테스트 손실: {loss_overfit:.4f}")

loss_dropout, accuracy_dropout = dropout_model.evaluate(X_test, y_test, verbose=0)
print(f"드롭아웃 적용 모델 테스트 정확도: {accuracy_dropout:.4f}, 테스트 손실: {loss_dropout:.4f}")