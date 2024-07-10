import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical

# 데이터셋 로드
data = pd.read_csv('fer2013.csv')

# 데이터셋 전처리 함수 정의
def preprocess_pixels(pixel_string):
    pixels = np.array(pixel_string.split(), dtype='int')
    return pixels.reshape((48, 48))

# 훈련 데이터와 테스트 데이터로 분할
train_data = data[data['Usage'] == 'Training']
test_data = data[data['Usage'] == 'PrivateTest']

# 훈련 데이터 전처리
X_train = np.array([preprocess_pixels(pixels) for pixels in train_data['pixels']])
y_train = to_categorical(train_data['emotion'], num_classes=7)

# 테스트 데이터 전처리
X_test = np.array([preprocess_pixels(pixels) for pixels in test_data['pixels']])
y_test = to_categorical(test_data['emotion'], num_classes=7)

# CNN 모델 정의
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 데이터 차원 조정 (CNN 입력 형태에 맞게)
X_train = X_train.reshape(X_train.shape[0], 48, 48, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1).astype('float32')

# 모델 학습
model.fit(X_train, y_train, batch_size=64, epochs=30, validation_data=(X_test, y_test))

# 모델 평가
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')

# 예측 및 평가
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# 분류 보고서 출력
print(classification_report(y_true, y_pred_classes))

# 혼동 행렬 출력
conf_mat = confusion_matrix(y_true, y_pred_classes)
print('Confusion Matrix:')
print(conf_mat)

model.save('fer_model.h5')