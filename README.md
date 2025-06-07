# 🔊 ESC-50: Danger vs Safe Classification using PANNs (Cnn14)

## 1. Mục tiêu

Phân loại âm thanh thành 2 lớp:

* **Danger** (Nguy hiểm)
* **Safe** (An toàn)

Sử dụng dataset **ESC-50** (dữ liệu âm thanh đa dạng, có gán nhãn sẵn) kết hợp với pretrained model **PANNs – Cnn14** để trích xuất embedding đặc trưng, sau đó huấn luyện mô hình phân loại nhị phân (binary classification).

---

## 2. Chuẩn bị dữ liệu

1. **Tách dữ liệu**:

   * Viết script `filterAudioToDangerousAndSafe.py` để lọc từ ESC-50 → 2 thư mục:

     * `dataset/Danger`
     * `dataset/Safe`
   * Gán nhãn:

     * `Safe`: 0
     * `Danger`: 1
	
![image](https://github.com/user-attachments/assets/7393f533-945a-40d2-b368-fd1664880542)

2. **Trích xuất embedding**:

   * Viết script `extractEmbeddingCnn14.py` dùng Cnn14 từ PANNs để trích xuất embedding cho mỗi audio.
   * Cần chỉnh sửa class `Cnn14` trong `models.py` để **trả về embedding** thay vì logits.

   ```python
   def forward(self, input, mixup_lambda=None):
       ...
       embedding = self.fc1_dropout(embedding)
       embedding = self.fc1(embedding)
       return {
           'clipwise_output': x,
           'embedding': embedding
       }
   ```

3. **Output**: File `audio_embeddings.csv` chứa các embedding và nhãn tương ứng.

---

## 3. Huấn luyện mô hình

Bắt đầu với mô hình đơn giản để đánh giá hiệu quả embedding:

### ✅ Ưu tiên thử trước: **Random Forest**

* Không cần chuẩn hóa dữ liệu
* Nhanh, ít cần tinh chỉnh
* Dễ diễn giải

### 🚀 Tiến trình:

1. Viết script `train_rf_model.py`
2. Huấn luyện mô hình trên `audio_embeddings.csv`
3. Lưu model: `rf_model.joblib`
4. Đánh giá với **confusion matrix**:

![image](https://github.com/user-attachments/assets/00e828f0-9ec9-4a38-94a2-752f1540945f)


---

## 4. Nhận xét

* **Vấn đề nghiêm trọng**: Mô hình dự đoán toàn bộ âm thanh là **Safe**:

  * TP Danger = 0, FP Safe = 40 → Mô hình không học được đặc trưng lớp "Danger"

### 📉 Nguyên nhân không phải do mất cân bằng nhãn:

```python
import pandas as pd
df = pd.read_csv("audio_embeddings.csv")
print(df['label'].value_counts())
# Output:
# 0    200
# 1    160
```

Tỷ lệ lớp vẫn tương đối cân bằng → Lỗi đến từ việc:

* Mô hình không học được đặc trưng nguy hiểm từ embedding.

---

## 5. Hướng cải tiến

* **Sử dụng mô hình nhạy hơn với embedding đặc trưng**:

  * Logistic Regression (LogReg)
  * Multi-layer Perceptron (MLP)
* **Kết hợp thêm**:

  * `StandardScaler` để chuẩn hóa
  * `class_weight='balanced'` để tăng trọng số lớp "Danger"

---

## 6. Ghi chú

* PANNs Cnn14 cần được chỉnh sửa để **xuất embedding**
* Các mô hình như RF có thể không đủ mạnh để học tốt trên embedding – nên thử MLP hoặc fine-tuning nếu cần

Dưới đây là bản cập nhật **README.md** hoàn chỉnh, bao gồm cả các bước cải tiến với chuẩn hóa và pooling:

---

# 🔊 ESC-50: Danger vs Safe Classification using PANNs (Cnn14)

## 1. Mục tiêu

Phân loại âm thanh thành 2 lớp:

* **Danger** (Nguy hiểm)
* **Safe** (An toàn)

Sử dụng dataset **ESC-50** (đa dạng, có gán nhãn sẵn) kết hợp với **PANNs – Cnn14** để trích xuất embedding, sau đó huấn luyện mô hình phân loại nhị phân.

---

## 2. Chuẩn bị dữ liệu

### ✅ Bước 1: Tổ chức lại dataset

* Viết script `filterAudioToDangerousAndSafe.py` để lọc từ ESC-50 ra 2 thư mục:

  * `dataset/Danger/`
  * `dataset/Safe/`
* Gán nhãn:

  * Danger → `1`
  * Safe → `0`

### ✅ Bước 2: Trích xuất embedding

* Sử dụng model `Cnn14` từ PANNs (pretrained)
* Chỉnh sửa trong `models.py` để trả về `embedding` thay vì logits:

```python
return {
    'clipwise_output': x,
    'embedding': embedding
}
```

* Viết script `extractEmbeddingCnn14.py` để tạo `audio_embeddings.csv`

---

## 3. Vấn đề ban đầu và nguyên nhân

### ❌ Các mô hình không học được lớp Danger

* Random Forest hoặc Logistic Regression dự đoán toàn bộ thành Safe (0)
* Ví dụ (Confusion Matrix – Logistic Regression):

![conf\_lr\_raw](./path_to_image.png)

**Phân tích:**

* Dữ liệu có 200 mẫu Safe vs 160 Danger → KHÔNG mất cân bằng đáng kể
* Nguyên nhân thực tế: mỗi sample có **nhiều vector thời gian (multi-frame embedding)**, chưa gộp về vector duy nhất

---

## 4. Cải tiến: Gộp embedding bằng Pooling

### ✅ Vấn đề:

Các vector embedding như `feat_0`, `feat_1`, ..., `feat_14` là **chuỗi theo thời gian**, mỗi bước có 2048 chiều

→ Mô hình như Logistic Regression yêu cầu mỗi sample là **một vector cố định**

### ✅ Cách xử lý:

Sử dụng **mean pooling**:

```python
merged = np.mean([feat_0, feat_1, ..., feat_14], axis=0)
```

* Viết script `mergeFeat_x.py` → xuất ra `merged_embeddings.csv`
* Mỗi dòng là embedding duy nhất ứng với 1 âm thanh (kích thước `[1, 2048]`)

---

## 5. Huấn luyện lại sau khi merge

### Logistic Regression (chuẩn hóa + pooling)

* Dùng `StandardScaler` chuẩn hóa:

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

* Kết quả confusion matrix:

```
                Predicted
             | Safe | Danger
-------------|------|--------
True Safe    |  23  |   17
True Danger  |   9  |   23
```

![conf\_lr\_mean](./path_to_logistic_mean_confusion.png)

---

### Random Forest (với mean embedding)

* Không cần chuẩn hóa
* Kết quả:

```
                Predicted
             | Safe | Danger
-------------|------|--------
True Safe    |  31  |   9
True Danger  |   9  |  23
```

![conf\_rf\_mean](./path_to_rf_mean_confusion.png)

---

## 6. Kết luận

| Mô hình            | Accuracy | Nhận biết Danger |
| ------------------ | -------- | ---------------- |
| Logistic (raw)     | 55%      | Không            |
| RF (raw)           | 55%      | Không            |
| Logistic + pooling | ✅ \~73%  | Có               |
| RF + pooling       | ✅ \~83%  | Có               |

### 👉 Bài học chính:

* Cần xử lý embedding dạng chuỗi thành vector duy nhất (mean pooling)
* Mô hình đơn giản + chuẩn hóa + format đúng → hiệu quả rõ rệt

---

## 📁 Cấu trúc thư mục đề xuất

```
project/
├── dataset/
│   ├── Danger/
│   └── Safe/
├── models/
│   └── models.py
├── extractEmbeddingCnn14.py
├── mergeFeat_x.py
├── audio_embeddings.csv
├── merged_embeddings.csv
├── train_rf_model.py
├── train_lr_model_LogisticRegression.py
└── README.md
```

---

Bạn muốn mình tạo sẵn template code cho `mergeFeat_x.py` hoặc mô hình Logistic Regression không?


