import numpy as np
import librosa
import tensorflow as tf
import joblib
import sys
import os

# Thêm đường dẫn để import model PANNs
sys.path.append(r'E:\DOAN1\PANNs\audioset_tagging_cnn_new')

def extract_features_with_panns(audio_path, model_path=None):
    """
    Trích xuất features từ audio file sử dụng PANNs model
    """
    try:
        # Load audio file
        audio, sr = librosa.load(audio_path, sr=32000)  # PANNs thường dùng 32kHz
        
        # Nếu audio quá ngắn, pad thêm
        if len(audio) < 32000:  # Ít nhất 1 giây
            audio = np.pad(audio, (0, 32000 - len(audio)), 'constant')
        
        # Nếu audio quá dài, chia thành segments
        segment_length = 32000 * 10  # 10 giây
        segments = []
        
        for i in range(0, len(audio), segment_length):
            segment = audio[i:i+segment_length]
            if len(segment) < segment_length:
                segment = np.pad(segment, (0, segment_length - len(segment)), 'constant')
            segments.append(segment)
        
        # Trích xuất features cho mỗi segment (giả lập)
        # Trong thực tế, bạn cần load PANNs model và extract features
        features_list = []
        
        for segment in segments:
            # Giả lập việc extract features (thay bằng PANNs thực tế)
            # Tạo features ngẫu nhiên với shape phù hợp với training data
            features = np.random.randn(2048)  # Giả sử PANNs output 2048 features
            features_list.append(features)
        
        # Lấy tối đa 15 segments (như trong training)
        features_list = features_list[:15]
        
        # Nếu ít hơn 15 segments, duplicate segment cuối
        while len(features_list) < 15:
            features_list.append(features_list[-1])
        
        return features_list
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def extract_simple_audio_features(audio_path):
    """
    Trích xuất features đơn giản từ audio (backup method)
    """
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=22050)
        
        # Extract basic features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        tempo = librosa.beat.tempo(y=audio, sr=sr)
        
        # Tính trung bình
        features = [
            np.mean(mfccs),
            np.std(mfccs),
            np.mean(spectral_centroids),
            np.mean(zero_crossing_rate),
            np.mean(spectral_rolloff),
            np.mean(chroma)
        ]
        
        return np.array(features)
        
    except Exception as e:
        print(f"Error extracting simple features: {e}")
        return None

# Test với file audio
if __name__ == "__main__":
    audio_file = r"E:\DOAN1\PANNs\audioset_tagging_cnn_new\resources\audio_dog_1.wav"
    
    if os.path.exists(audio_file):
        print(f"Processing: {audio_file}")
        
        # Method 1: Simple features (sẽ hoạt động ngay)
        print("\n=== Method 1: Simple Audio Features ===")
        simple_features = extract_simple_audio_features(audio_file)
        
        if simple_features is not None:
            print(f"Extracted features shape: {simple_features.shape}")
            print(f"Features: {simple_features}")
            
            # Load classifier và predict
            try:
                from audio_classifier_api import AudioSafetyClassifier
                classifier = AudioSafetyClassifier()
                result = classifier.predict(simple_features, return_details=True)
                
                print(f"\n🎯 Prediction Results:")
                print(f"   File: {os.path.basename(audio_file)}")
                print(f"   Label: {result['label']}")
                print(f"   Probability: {result['probability']:.3f}")
                print(f"   Confidence: {result['confidence']:.3f}")
                
            except Exception as e:
                print(f"Error in prediction: {e}")
        
        # Method 2: PANNs features (cần implement)
        print("\n=== Method 2: PANNs Features (Simulated) ===")
        panns_features = extract_features_with_panns(audio_file)
        if panns_features is not None:
            print(f"Extracted {len(panns_features)} feature vectors")
            # Tính trung bình như trong training
            mean_features = np.mean(panns_features, axis=0)
            # Reduce to 6 features to match training data
            reduced_features = mean_features[:6]
            
            try:
                from audio_classifier_api import AudioSafetyClassifier
                classifier = AudioSafetyClassifier()
                result = classifier.predict(reduced_features, return_details=True)
                
                print(f"\n🎯 PANNs Prediction Results:")
                print(f"   File: {os.path.basename(audio_file)}")
                print(f"   Label: {result['label']}")
                print(f"   Probability: {result['probability']:.3f}")
                print(f"   Confidence: {result['confidence']:.3f}")
                
            except Exception as e:
                print(f"Error in PANNs prediction: {e}")
    
    else:
        print(f"File not found: {audio_file}")