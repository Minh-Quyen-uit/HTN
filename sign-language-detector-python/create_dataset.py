import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import numpy as np  # Thêm import numpy để xử lý mảng

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Điều chỉnh các tham số cho phù hợp nếu cần
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=1)

DATA_DIR = './data'

data = []
labels = []

# Kiểm tra xem thư mục dữ liệu có tồn tại không
if not os.path.exists(DATA_DIR):
    print(f"Lỗi: Thư mục '{DATA_DIR}' không tồn tại. Vui lòng chạy script thu thập dữ liệu trước.")
    exit()

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue  # Bỏ qua nếu không phải là thư mục

    for img_path in os.listdir(dir_path):
        img_file = os.path.join(dir_path, img_path)
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue  # Bỏ qua nếu không phải là file ảnh

        data_aux = []
        x_ = []
        y_ = []

        try:
            img = cv2.imread(img_file)
            if img is None:
                print(f"Lỗi: Không thể đọc ảnh '{img_file}'.")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                # Lấy thông tin bàn tay đầu tiên phát hiện được (nếu max_num_hands=1)
                hand_landmarks = results.multi_hand_landmarks[0]

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                # Chuẩn hóa tọa độ bằng cách trừ đi giá trị min
                min_x = min(x_)
                min_y = min(y_)
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min_x)
                    data_aux.append(y - min_y)

                data.append(data_aux)
                labels.append(dir_)
            else:
                print(f"Không tìm thấy bàn tay trong ảnh '{img_file}'.")

        except Exception as e:
            print(f"Lỗi khi xử lý ảnh '{img_file}': {e}")

# Kiểm tra xem có dữ liệu được thu thập hay không
if not data:
    print("Không có dữ liệu nào được thu thập. Vui lòng kiểm tra thư mục dữ liệu.")
    exit()

try:
    with open('data.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print("Dữ liệu đã được xử lý và lưu vào 'data.pickle'.")
except Exception as e:
    print(f"Lỗi khi lưu dữ liệu vào file 'data.pickle': {e}")