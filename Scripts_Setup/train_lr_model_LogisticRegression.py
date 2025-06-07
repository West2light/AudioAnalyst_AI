import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import joblib
import ast
import os

# Bước 1: Load và xử lý dữ liệu
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(current_dir, "output", "E:\DOAN1\PANNs\output\merged_embeddings.csv"))
    print("✅ Đã đọc file merged_embeddings.csv thành công")
except FileNotFoundError:
    print("❌ Không tìm thấy file merged_embeddings.csv")
    exit(1)

# Hàm để chuyển đổi chuỗi embedding thành mảng numpy
def convert_embedding_string_to_array(embedding_str):
    try:
        return np.array(ast.literal_eval(embedding_str))
    except Exception as e:
        print(f"⚠️ Lỗi khi chuyển đổi embedding: {str(e)}")
        return np.zeros(128)  # Giả sử mỗi embedding có 128 chiều

# Chuyển đổi cột embedding thành numpy array
print("🔄 Đang xử lý embeddings...")
X = np.array([convert_embedding_string_to_array(emb) for emb in df['embedding']])
y = df['label'].values

print(f"�� Kích thước dữ liệu: {X.shape}")
print(f"🎯 Số lượng nhãn: {len(np.unique(y))}")

# Bước 2: Tách train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Bước 3: Chuẩn hóa dữ liệu
print("\n🔄 Đang chuẩn hóa dữ liệu...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Bước 4: Huấn luyện mô hình
print("\n🔄 Đang huấn luyện mô hình Logistic Regression...")
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Bước 5: Đánh giá
y_pred = model.predict(X_test_scaled)

print("\n✅ Classification report:")
print(classification_report(y_test, y_pred))

# Vẽ ma trận nhầm lẫn
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Safe", "Danger"])
disp.plot()
plt.title("Confusion Matrix - Logistic Regression (Mean Embedding)")

# Tạo thư mục output nếu chưa tồn tại
os.makedirs('output', exist_ok=True)
plt.savefig('output/confusion_matrix_lr.png')
plt.close()

# Lưu mô hình và scaler
model_path = 'output/lr_model.joblib'
scaler_path = 'output/lr_scaler.joblib'
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
print(f"\n✅ Đã lưu mô hình tại: {model_path}")
print(f"✅ Đã lưu scaler tại: {scaler_path}")