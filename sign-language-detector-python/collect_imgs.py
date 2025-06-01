import os
import cv2
import numpy as np
import time

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

dataset_size = 100
split_size = 50  # Số lượng ảnh trước khi tạm dừng
classes_to_collect = [32, 33, 34, 35]  # Chỉ thu thập class 25 và 26

# Mở camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không thể mở camera. Vui lòng kiểm tra kết nối camera hoặc thay đổi index.")
    exit()

for j in classes_to_collect:
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Thu thập dữ liệu cho class {j} - Nhấn "," để bắt đầu...')

    # Chờ người dùng nhấn "," để bắt đầu
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không nhận được khung hình. Vui lòng kiểm tra kết nối camera.")
            break
        cv2.putText(frame, 'Ready? Press "," ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord(','):
            break

    # Chụp 50 ảnh đầu
    for counter in range(split_size):
        ret, frame = cap.read()
        if not ret:
            print("Không nhận được khung hình.")
            break
        img_name_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(img_name_path, frame)
        print(f"Đã lưu ảnh: {img_name_path}")
        cv2.imshow('frame', frame)
        cv2.waitKey(25)

    print(f"Đã thu thập được {split_size} ảnh đầu. Tạm dừng 5 giây...")
    time.sleep(5)

    # Chụp 50 ảnh còn lại
    for counter in range(split_size, dataset_size):
        ret, frame = cap.read()
        if not ret:
            print("Không nhận được khung hình.")
            break
        img_name_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(img_name_path, frame)
        print(f"Đã lưu ảnh: {img_name_path}")
        cv2.imshow('frame', frame)
        cv2.waitKey(25)

cap.release()
cv2.destroyAllWindows()
print("Hoàn tất thu thập dữ liệu.")
