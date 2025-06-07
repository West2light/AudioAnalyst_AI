import os
import torch
import torchaudio
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import sys
# Thêm đường dẫn thư mục gốc vào sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Thư mục cha (PANNs)
sys.path.append(current_dir)

from pytorch.models import Cnn14

# Cấu hình
SAMPLE_RATE = 32000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model Random Forest và scaler
try:
    model_path = os.path.join(parent_dir, "output", "rf_model.joblib")
    scaler_path = os.path.join(parent_dir, "output", "rf_scaler.joblib")
    
    # Kiểm tra file tồn tại
    if not os.path.exists(model_path):
        print(f"❌ Không tìm thấy file model tại: {model_path}")
        print("⚠️ Vui lòng chạy train_rf_model_RandomForest.py trước để tạo model")
        exit(1)
    if not os.path.exists(scaler_path):
        print(f"❌ Không tìm thấy file scaler tại: {scaler_path}")
        print("⚠️ Vui lòng chạy train_rf_model_RandomForest.py trước để tạo scaler")
        exit(1)
        
    rf_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("✅ Đã load model Random Forest và scaler thành công")
except Exception as e:
    print(f"❌ Lỗi khi load model: {str(e)}")
    exit(1)
# Load model Cnn14
try:
    cnn_model = Cnn14(
        sample_rate=32000, 
        window_size=1024, 
        hop_size=320, 
        mel_bins=64, 
        fmin=50, 
        fmax=14000,
        classes_num=527
    )
    cnn_model_path = os.path.join(current_dir, "Cnn14_mAP=0.431.pth")
    checkpoint = torch.load(cnn_model_path, map_location=DEVICE)
    cnn_model.load_state_dict(checkpoint['model'])
    cnn_model.to(DEVICE)
    cnn_model.eval()
    print("✅ Đã load model Cnn14 thành công")
except Exception as e:
    print(f"❌ Lỗi khi load model Cnn14: {str(e)}")
    exit(1)

def load_audio(file_path):
    """Load và tiền xử lý audio file"""
    waveform, sr = torchaudio.load(file_path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    return waveform.mean(dim=0)  # Chuyển về mono

def extract_embedding(audio_tensor):
    """Trích xuất embedding từ model Cnn14"""
    audio_tensor = audio_tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output_dict = cnn_model.forward(audio_tensor)
        embedding = output_dict['embedding'].squeeze().cpu().numpy()
    return embedding

def predict_safety(filepath):
    """Dự đoán an toàn/nguy hiểm cho file âm thanh"""
    try:
        # Load và xử lý audio
        audio = load_audio(filepath)
        
        # Trích xuất embedding
        embedding = extract_embedding(audio)  # (n_frames, 1024)
        
        # Tính trung bình theo chiều thời gian
        mean_embedding = np.mean(embedding, axis=0)  # (1024,)
        
        # Chọn 6 features đầu tiên (feat_0 đến feat_5)
        selected_features = mean_embedding[:6]  # (6,)
        
        # Reshape để phù hợp với scaler
        selected_features = selected_features.reshape(1, -1)  # (1, 6)
        
        # Chuẩn hóa
        features_scaled = scaler.transform(selected_features)
        
        # Dự đoán
        prediction = rf_model.predict(features_scaled)[0]
        probability = rf_model.predict_proba(features_scaled)[0]
        
        label = "Safe" if prediction == 0 else "Danger"
        confidence = probability[prediction] * 100
        
        return {
            "label": label,
            "confidence": f"{confidence:.2f}%",
            "probabilities": {
                "Safe": f"{probability[0]*100:.2f}%",
                "Danger": f"{probability[1]*100:.2f}%"
            }
        }
        
    except Exception as e:
        print(f"❌ Lỗi khi xử lý file {filepath}: {str(e)}")
        return None

def main():
    # Test với một file âm thanh
    test_file = input("Nhập đường dẫn file âm thanh (.wav): ")
    
    if not os.path.exists(test_file):
        print("❌ Không tìm thấy file!")
        return
        
    if not test_file.endswith('.wav'):
        print("❌ File phải có định dạng .wav!")
        return
    
    print("\n🔄 Đang xử lý file...")
    result = predict_safety(test_file)
    
    if result:
        print("\n✅ Kết quả dự đoán:")
        print(f"Label: {result['label']}")
        print(f"Confidence: {result['confidence']}")
        print("\nXác suất cho từng lớp:")
        print(f"Safe: {result['probabilities']['Safe']}")
        print(f"Danger: {result['probabilities']['Danger']}")

if __name__ == "__main__":
    main()