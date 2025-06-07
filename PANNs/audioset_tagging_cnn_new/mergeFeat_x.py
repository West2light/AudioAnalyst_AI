import pandas as pd
import numpy as np
import ast
import os

# Đọc file CSV
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(current_dir, "audio_embeddings.csv"))
    print("✅ Đã đọc file CSV thành công")
except FileNotFoundError:
    print("❌ Không tìm thấy file audio_embeddings.csv")
    exit(1)
def convert_string_to_array(s):
    """Chuyển chuỗi [0.1 0.2 ...] thành numpy array"""
    try:
        # Loại bỏ dấu ngoặc vuông và khoảng trắng thừa
        s = s.strip('[]')
        # Tách các số và chuyển thành float, bỏ qua '...'
        numbers = [float(x) for x in s.split() if x.strip() and x != '...']
        return np.array(numbers)
    except Exception as e:
        print(f"⚠️ Lỗi khi chuyển đổi chuỗi thành array: {str(e)}")
        return np.zeros(128)  # Giả sử mỗi embedding có 128 chiều

print("🔄 Đang xử lý dữ liệu...")
final_embeddings = []
labels = []

for _, row in df.iterrows():
    features = []
    for i in range(15):  # feat_0 đến feat_14
        f = convert_string_to_array(row[f"feat_{i}"])
        features.append(f)
    
    # Gộp toàn bộ 15 đoạn embedding lại bằng cách lấy trung bình
    final_vector = np.mean(features, axis=0)
    final_embeddings.append(final_vector)
    labels.append(row["label"])

# Chuyển đổi thành numpy array
X = np.array(final_embeddings)
y = np.array(labels)

print(f"📊 Kích thước dữ liệu sau khi gộp: {X.shape}")
print(f"🎯 Số lượng nhãn: {len(np.unique(y))}")

# Lưu kết quả vào file CSV mới
output_df = pd.DataFrame({
    "filename": df["filename"],  # Giữ lại tên file
    "label": labels,
    "embedding": [str(emb.tolist()) for emb in final_embeddings]  # Chuyển numpy array thành string để lưu vào CSV
})

# Tạo thư mục output nếu chưa tồn tại
os.makedirs('output', exist_ok=True)
output_path = os.path.join('output', 'merged_embeddings.csv')
output_df.to_csv(output_path, index=False)
print(f"✅ Đã lưu kết quả vào file: {output_path}")

# In ra một số thông tin về dữ liệu
print("\n📈 Thông tin về dữ liệu:")
print(f"- Số lượng mẫu: {len(final_embeddings)}")
print(f"- Kích thước vector embedding: {final_embeddings[0].shape}")
print(f"- Phân bố nhãn: {np.bincount(y)}")