import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Kiểm tra xem file 'data.pickle' có tồn tại không
try:
    with open('./data.pickle', 'rb') as f:
        data_dict = pickle.load(f)
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file 'data.pickle'. Đảm bảo bạn đã chạy script thu thập dữ liệu trước.")
    exit()
except Exception as e:
    print(f"Lỗi khi đọc file 'data.pickle': {e}")
    exit()

if 'data' not in data_dict or 'labels' not in data_dict:
    print("Lỗi: File 'data.pickle' không chứa khóa 'data' hoặc 'labels'.")
    exit()

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Kiểm tra xem có đủ dữ liệu để chia tập không
if len(np.unique(labels)) < 2:
    print("Lỗi: Cần ít nhất hai lớp dữ liệu để chia tập huấn luyện và kiểm tra.")
    exit()

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Khởi tạo và huấn luyện mô hình
model = RandomForestClassifier()
try:
    model.fit(x_train, y_train)
except Exception as e:
    print(f"Lỗi trong quá trình huấn luyện mô hình: {e}")
    exit()

# Dự đoán và đánh giá
try:
    y_predict = model.predict(x_test)
    score = accuracy_score(y_predict, y_test)
    print('{}% of samples were classified correctly !'.format(score * 100))
except Exception as e:
    print(f"Lỗi trong quá trình dự đoán hoặc đánh giá: {e}")
    exit()

# Lưu mô hình
try:
    with open('model.p', 'wb') as f:
        pickle.dump({'model': model}, f)
    print("Mô hình đã được lưu vào file 'model.p'.")
except Exception as e:
    print(f"Lỗi khi lưu mô hình vào file 'model.p': {e}")
    exit()