import pickle

import cv2
import mediapipe as mp
import numpy as np
import pyttsx3  # Import thư viện text-to-speech

import requests  # Thêm thư viện requests để gửi HTTP

# Địa chỉ IP của ESP8266 (thay bằng IP thực tế bạn thấy trên Serial Monitor khi ESP8266 khởi động)
ESP8266_IP = '172.20.10.11'  # <-- THAY bằng IP thực của ESP8266

def send_to_esp8266(data):
    try:
        url = f"http://{ESP8266_IP}/char?c={data}"
        response = requests.get(url, timeout=1)
        if response.status_code == 200:
            print(f"✅ Đã gửi ký tự '{data}' đến ESP8266")
        else:
            print(f"❌ Lỗi HTTP: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Lỗi khi gửi đến ESP8266: {e}")

# Khởi tạo engine text-to-speech
engine = pyttsx3.init()
voices = engine.getProperty('voices')
# Chọn giọng đọc tiếng Anh
for voice in voices:
    if 'English' in voice.name or 'en-US' in voice.id:
        engine.setProperty('voice', voice.id)
        break
# Nếu không tìm thấy giọng đọc tiếng Anh cụ thể, có thể sử dụng giọng đọc mặc định

# Load the trained model
try:
    with open('./model.p', 'rb') as f:
        model_dict = pickle.load(f)
    model = model_dict['model']
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file 'model.p'. Đảm bảo bạn đã huấn luyện mô hình trước.")
    exit()
except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}")
    exit()

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for default camera, try other indices if needed
if not cap.isOpened():
    print("Không thể mở camera. Vui lòng kiểm tra kết nối camera.")
    exit()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False,  # Set to False for real-time video
                       max_num_hands=1,  # Adjust if you want to detect multiple hands
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Define label mapping
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M',
               13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
               25: 'Z', 26: 'ZERO', 27: 'ONE', 28: 'TWO', 29: 'THREE', 30: 'FOUR', 31: 'FIVE', 32: 'SIX', 33: 'SEVEN', 34: 'EIGHT', 35: 'NINE'}

predicted_character = ''  # Biến lưu trữ ký tự dự đoán hiện tại
prev_character = ''
while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        print("Không nhận được khung hình. Kết thúc.")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # To improve performance, mark the frame as not writeable to pass by reference.
    frame_rgb.flags.writeable = False
    results = hands.process(frame_rgb)
    frame_rgb.flags.writeable = True
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Extract landmarks for prediction (assuming only one hand is detected)
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            if x_ and y_:  # Ensure landmarks were detected
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                # Ensure the data_aux has the expected number of features
                if len(data_aux) == 42:  # 21 landmarks * 2 coordinates
                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10

                    x2 = int(max(x_) * W) - 10
                    y2 = int(max(y_) * H) - 10

                    data_array = np.asarray(data_aux).reshape(1, -1)  # Reshape for prediction
                    prediction = model.predict(data_array)
                    predicted_character = labels_dict.get(int(prediction[0]), 'Unknown')
                    #thêm dòng này
                    if predicted_character and predicted_character != prev_character:
                        send_to_esp8266(predicted_character)
                        prev_character = predicted_character
                    #đến đây
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                                cv2.LINE_AA)
                else:
                    print(f"Warning: Incorrect number of features ({len(data_aux)}), expected 42.")
            else:
                predicted_character = ''  # Reset nếu không phát hiện tay

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(','):
        break
    elif key == ord(';'):
        if predicted_character:
            engine.say(predicted_character)
            engine.runAndWait()

cap.release()
cv2.destroyAllWindows()