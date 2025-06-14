import numpy as np
import joblib
import tensorflow as tf

def predict_audio_safety(audio_features):
    """
    Dự đoán độ an toàn của audio
    
    Args:
        audio_features: list of 15 vectors (mỗi vector có shape tương tự như training data)
    
    Returns:
        dict: {'probability': float, 'label': str, 'prediction': int}
    """
    # Load model và scaler
    model = tf.keras.models.load_model("audio_classification_model.keras")
    scaler = joblib.load("scaler.pkl")
    
    # Xử lý audio features (giống như trong convertVectors.py)
    if len(audio_features) == 15:
        # Tính trung bình của 15 vectors
        mean_vector = np.mean(audio_features, axis=0).reshape(1, -1)
    else:
        # Nếu chỉ có 1 vector
        mean_vector = np.array(audio_features).reshape(1, -1)
    
    # Chuẩn hóa
    mean_vector_scaled = scaler.transform(mean_vector)
    
    # Dự đoán
    probability = model.predict(mean_vector_scaled)[0][0]
    predicted_label = 1 if probability > 0.5 else 0
    label_text = "Danger" if predicted_label == 1 else "Safe"
    
    return {
        'probability': float(probability),
        'prediction': int(predicted_label),
        'label': label_text,
        'confidence': float(abs(probability - 0.5) * 2)  # Độ tin cậy từ 0-1
    }

# Test function với dữ liệu mẫu
if __name__ == "__main__":
    print("Testing prediction function...")
    
    # Tạo dữ liệu test giả (thay bằng dữ liệu thực tế)
    sample_features = np.random.randn(6)  # 6 features như training data
    
    result = predict_audio_safety(sample_features)
    
    print(f"Prediction Result:")
    print(f"  Label: {result['label']}")
    print(f"  Probability: {result['probability']:.3f}")
    print(f"  Confidence: {result['confidence']:.3f}")