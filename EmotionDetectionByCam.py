import cv2
import numpy as np
from keras.models import load_model

# 감정 레이블
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# 학습된 모델 로드
model = load_model('fer_model.h5')  # 학습된 모델 파일

# 웹캠 시작
cap = cv2.VideoCapture(0)

while True:
    # 웹캠 프레임 읽기
    ret, frame = cap.read()
    
    # 프레임 크기 조정 (학습할 때와 같은 크기로)
    small_frame = cv2.resize(frame, (48, 48))
    
    # 그레이스케일 변환
    gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    
    # 모델 입력을 위한 형태로 변환 (CNN 입력 형태에 맞춤)
    input_data = gray_frame.reshape((1, 48, 48, 1)) / 255.0
    
    # 감정 예측
    prediction = model.predict(input_data)
    
    # 예측 결과 확인
    max_index = np.argmax(prediction[0])
    emotion = emotions[max_index]
    
    # 예측 확률 확인 (옵션)
    max_probability = np.max(prediction[0])
    print(f'Predicted emotion: {emotion}, Probability: {max_probability}')
    
    # 프레임에 텍스트 추가
    cv2.putText(frame, emotion, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 화면에 출력
    cv2.imshow('Emotion Detection', frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 캡처 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()