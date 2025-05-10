import os
import cv2
import numpy as np  # Thêm import numpy

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 36
dataset_size = 100

# Kiểm tra xem camera có mở được không
cap = cv2.VideoCapture(0)  # Thử camera mặc định (index 0) trước
if not cap.isOpened():
    print("Không thể mở camera. Vui lòng kiểm tra kết nối camera hoặc thay đổi index.")
    exit()

for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không nhận được khung hình. Vui lòng kiểm tra kết nối camera.")
            break
        cv2.putText(frame, 'Ready? Press "," ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord(','):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Không nhận được khung hình. Vui lòng kiểm tra kết nối camera.")
            break
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        img_name_path = os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter))
        cv2.imwrite(img_name_path, frame)
        print(f"Đã lưu ảnh: {img_name_path}")
        counter += 1

cap.release()
cv2.destroyAllWindows()
print("Hoàn tất thu thập dữ liệu.")