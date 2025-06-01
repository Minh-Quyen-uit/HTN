import pickle
import cv2
import mediapipe as mp
import numpy as np
import requests
import io
import time

ESP8266_IP = '172.20.10.11'
ESP32_IP = '172.20.10.10'
CAPTURE_URL = f'http://{ESP32_IP}/capture'
STREAM_URL = f'http://{ESP32_IP}:81/stream'

def send_to_esp8266(data):
    try:
        url = f"http://{ESP8266_IP}/char?c={data}"
        response = requests.get(url, timeout=1)
        if response.status_code == 200:
            print(f"Đã gửi ký tự '{data}' đến ESP8266")
        else:
            print(f"Lỗi HTTP: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Lỗi khi gửi đến ESP8266: {e}")

# Load model
try:
    with open('./model.p', 'rb') as f:
        model_dict = pickle.load(f)
    model = model_dict['model']
except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}")
    exit()

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Label mapping
labels_dict = {i: chr(65 + i) for i in range(26)}  # A-Z
labels_dict.update({26: '0', 27: '1', 28: '2', 29: '3', 30: '4',
                    31: '5', 32: '6', 33: '7', 34: '8', 35: '9'})

predicted_character = ''
prev_character = ''
last_capture_time = 0
capture_interval = 2
mode = 'idle'
display_frame = np.zeros((480, 640, 3), dtype=np.uint8)

def process_frame(frame):
    global predicted_character, prev_character
    data_aux, x_, y_ = [], [], []

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    processed_frame = frame.copy()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                processed_frame, hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            if x_ and y_:
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

                if len(data_aux) == 42:
                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10
                    x2 = int(max(x_) * W) + 10
                    y2 = int(max(y_) * H) + 10

                    data_array = np.asarray(data_aux).reshape(1, -1)
                    prediction = model.predict(data_array)
                    predicted_character = labels_dict.get(int(prediction[0]), 'Unknown')

                    if predicted_character != prev_character:
                        send_to_esp8266(predicted_character)
                        prev_character = predicted_character

                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(processed_frame, predicted_character, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
    return processed_frame

def get_frame_from_esp32():
    try:
        response = requests.get(CAPTURE_URL, stream=True, timeout=2)
        response.raise_for_status()
        image_bytes = io.BytesIO(response.content)
        frame = cv2.imdecode(np.frombuffer(image_bytes.read(), np.uint8), cv2.IMREAD_COLOR)
        return frame
    except requests.exceptions.RequestException as e:
        print(f"Lỗi lấy ảnh từ ESP32: {e}")
        return None

# ===================== MAIN LOOP =====================
cap_stream = None
while True:
    key = cv2.waitKey(1) & 0xFF

    # ==== Chuyển chế độ ====
    if key == ord('c'):
        print("Chế độ: Chụp mỗi 2 giây")
        mode = 'capture'
        last_capture_time = time.time() - capture_interval
    elif key == ord('r'):
        print("Chế độ: Quay nhận diện liên tục")
        mode = 'stream'
        if cap_stream is None:
            cap_stream = cv2.VideoCapture(STREAM_URL)
    elif key == ord('i'):
        print("Chế độ: Không làm gì (idle)")
        mode = 'idle'
        display_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    elif key == ord('q'):
        break

    # ==== Chế độ capture 5s ====
    if mode == 'capture':
        if time.time() - last_capture_time >= capture_interval:
            print("Đang cố gắng chụp ảnh...")
            while True:
                frame = get_frame_from_esp32()
                if frame is not None:
                    display_frame = process_frame(frame)
                    last_capture_time = time.time()
                    break
                else:
                    print("Chụp lỗi, thử lại sau 2 giây...")
                    time.sleep(2)

    # ==== Chế độ stream realtime ====
    elif mode == 'stream':
        if cap_stream and cap_stream.isOpened():
            ret, frame = cap_stream.read()
            if ret and frame is not None:
                display_frame = process_frame(frame)
            else:
                print("Không đọc được stream.")
                display_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                mode = 'idle'
                if cap_stream:
                    cap_stream.release()
                    cap_stream = None

    cv2.imshow("ESP32-CAM Feed", display_frame)

# ==== Dọn dẹp ====
if cap_stream:
    cap_stream.release()
cv2.destroyAllWindows()
