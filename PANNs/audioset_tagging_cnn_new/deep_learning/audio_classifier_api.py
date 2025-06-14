import numpy as np
import joblib
import tensorflow as tf
import os

class AudioSafetyClassifier:
    def __init__(self, model_path="audio_classification_model.keras", scaler_path="scaler.pkl"):
        """Khởi tạo classifier với model và scaler đã train"""
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.load_model()
    
    def load_model(self):
        """Load model và scaler"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = tf.keras.models.load_model(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                print("✅ Model và scaler loaded successfully!")
            else:
                raise FileNotFoundError("Model hoặc scaler file không tồn tại")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def predict(self, audio_features, return_details=False):
        """
        Dự đoán độ an toàn của audio
        
        Args:
            audio_features: numpy array hoặc list với 6 features
            return_details: bool, có trả về thông tin chi tiết không
        
        Returns:
            dict hoặc string tùy theo return_details
        """
        try:
            # Xử lý input
            if isinstance(audio_features, list):
                audio_features = np.array(audio_features)
            
            # Reshape nếu cần
            if audio_features.ndim == 1:
                audio_features = audio_features.reshape(1, -1)
            
            # Chuẩn hóa
            features_scaled = self.scaler.transform(audio_features)
            
            # Dự đoán
            probability = self.model.predict(features_scaled, verbose=0)[0][0]
            predicted_label = 1 if probability > 0.5 else 0
            confidence = abs(probability - 0.5) * 2
            
            # Kết quả
            result = {
                'prediction': int(predicted_label),
                'label': "Danger" if predicted_label == 1 else "Safe",
                'probability': float(probability),
                'confidence': float(confidence),
                'status': 'success'
            }
            
            if return_details:
                return result
            else:
                return result['label']
                
        except Exception as e:
            error_result = {
                'status': 'error',
                'message': str(e),
                'prediction': None,
                'label': 'Unknown'
            }
            return error_result if return_details else 'Error'
    
    def batch_predict(self, batch_features):
        """Dự đoán cho nhiều mẫu cùng lúc"""
        try:
            batch_features = np.array(batch_features)
            if batch_features.ndim == 1:
                batch_features = batch_features.reshape(1, -1)
            
            features_scaled = self.scaler.transform(batch_features)
            probabilities = self.model.predict(features_scaled, verbose=0).flatten()
            
            results = []
            for prob in probabilities:
                predicted_label = 1 if prob > 0.5 else 0
                results.append({
                    'prediction': int(predicted_label),
                    'label': "Danger" if predicted_label == 1 else "Safe",
                    'probability': float(prob),
                    'confidence': float(abs(prob - 0.5) * 2)
                })
            
            return results
            
        except Exception as e:
            return [{'status': 'error', 'message': str(e)}] * len(batch_features)

# Test API
if __name__ == "__main__":
    # Khởi tạo classifier
    classifier = AudioSafetyClassifier()
    
    # Test với dữ liệu mẫu
    print("\n=== Testing API ===")
    
    # Test đơn lẻ
    sample_data = np.random.randn(6)
    result1 = classifier.predict(sample_data, return_details=True)
    result2 = classifier.predict(sample_data, return_details=False)
    
    print(f"Detailed result: {result1}")
    print(f"Simple result: {result2}")
    
    # Test batch
    batch_data = np.random.randn(3, 6)  # 3 mẫu
    batch_results = classifier.batch_predict(batch_data)
    
    print("\nBatch results:")
    for i, result in enumerate(batch_results):
        print(f"  Sample {i+1}: {result['label']} (confidence: {result['confidence']:.2f})")