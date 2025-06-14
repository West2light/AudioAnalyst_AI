import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")

# Load dữ liệu đã xử lý
X = np.load("X_features.npy")
y = np.load("y_labels.npy")

print(f"Data shape: X={X.shape}, y={y.shape}")

# Chuẩn hóa đầu vào
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"Train shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Test shape: X_test={X_test.shape}, y_test={y_test.shape}")

# Mô hình sử dụng tf.keras trực tiếp
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
    loss='binary_crossentropy', 
    metrics=['accuracy']
)

print("Model summary:")
model.summary()

# Huấn luyện
print("Starting training...")
history = model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.2, verbose=1)

# Lưu model và scaler
model.save("audio_classification_model.keras")
joblib.dump(scaler, "scaler.pkl")

# Đánh giá model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")
print("Đã lưu model và scaler vào thư mục deep_learning")