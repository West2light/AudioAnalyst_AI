import numpy as np
import joblib
import tensorflow as tf
import librosa
import os

class ImprovedAudioClassifier:
    def __init__(self):
        """Load improved model và scaler"""
        self.model = tf.keras.models.load_model("audio_classification_model_improved.keras")
        self.scaler = joblib.load("scaler_improved.pkl")
        print("✅ Improved model loaded successfully!")
    
    def extract_improved_features(self, audio_path):
        """Extract features theo cách improved"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=22050)
            
            # Extract comprehensive features
            features = []
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
            
            # Combine all features
            all_features = np.concatenate([
                mfccs.flatten(),
                chroma.flatten(),
                spectral_centroid.flatten(),
                spectral_bandwidth.flatten(), 
                spectral_rolloff.flatten(),
                zero_crossing_rate.flatten()
            ])
            
            # Calculate statistics like improved model expects
            improved_features = [
                np.mean(all_features),
                np.std(all_features),
                np.min(all_features),
                np.max(all_features),
                np.median(all_features),
                np.percentile(all_features, 25),
                np.percentile(all_features, 75),
                len(all_features),
                np.sum(all_features > 0),
                np.sum(all_features == 0)
            ]
            
            return np.array(improved_features)
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def predict(self, audio_path, return_details=True):
        """Predict với improved model"""
        features = self.extract_improved_features(audio_path)
        
        if features is None:
            return {'error': 'Could not extract features'}
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict
        probability = self.model.predict(features_scaled, verbose=0)[0][0]
        predicted_label = 1 if probability > 0.5 else 0
        confidence = abs(probability - 0.5) * 2
        
        result = {
            'filename': os.path.basename(audio_path),
            'prediction': int(predicted_label),
            'label': "Danger" if predicted_label == 1 else "Safe",
            'probability': float(probability),
            'confidence': float(confidence),
            'model': 'improved'
        }
        
        if return_details:
            return result
        else:
            return result['label']

def test_phone_ring():
    """Test cụ thể với file chuông điện thoại"""
    
    classifier = ImprovedAudioClassifier()
    
    phone_ring_file = r"E:\DOAN1\PANNs\audioset_tagging_cnn_new\resources\R9_ZSCveAHg_7s.mp3"
    
    if os.path.exists(phone_ring_file):
        print(f"🔊 Testing: {os.path.basename(phone_ring_file)}")
        
        result = classifier.predict(phone_ring_file, return_details=True)
        
        print(f"\n📊 Results with Improved Model:")
        print(f"   File: {result['filename']}")
        print(f"   Prediction: {result['label']}")
        print(f"   Probability: {result['probability']:.3f}")
        print(f"   Confidence: {result['confidence']:.3f}")
        
        if result['label'] == 'Safe':
            print(f"   ✅ Correctly classified as SAFE!")
        else:
            print(f"   ⚠️  Still classified as DANGER")
            
    else:
        print(f"File not found: {phone_ring_file}")

def compare_models():
    """So sánh old model vs improved model"""
    
    # Load old model
    try:
        old_model = tf.keras.models.load_model("audio_classification_model.keras")
        old_scaler = joblib.load("scaler.pkl")
        print("✅ Old model loaded")
    except:
        print("❌ Could not load old model")
        return
    
    # Load improved model
    improved_classifier = ImprovedAudioClassifier()
    
    # Test files
    test_files = [
        r"E:\DOAN1\PANNs\audioset_tagging_cnn_new\resources\R9_ZSCveAHg_7s.mp3",
        r"E:\DOAN1\PANNs\audioset_tagging_cnn_new\resources\audio_dog_1.wav"
    ]
    
    print(f"\n📊 Model Comparison:")
    print(f"{'File':<25} {'Old Model':<15} {'Improved Model':<15}")
    print("-" * 55)
    
    for audio_file in test_files:
        if os.path.exists(audio_file):
            filename = os.path.basename(audio_file)
            
            # Test with improved model
            improved_result = improved_classifier.predict(audio_file, return_details=True)
            
            # Test with old model (simplified)
            try:
                # Simple features for old model
                audio, sr = librosa.load(audio_file, sr=22050)
                mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
                zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
                spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
                chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
                
                old_features = np.array([
                    np.mean(mfccs), np.std(mfccs), np.mean(spectral_centroids),
                    np.mean(zero_crossing_rate), np.mean(spectral_rolloff), np.mean(chroma)
                ])
                
                old_features_scaled = old_scaler.transform(old_features.reshape(1, -1))
                old_prob = old_model.predict(old_features_scaled, verbose=0)[0][0]
                old_label = "Danger" if old_prob > 0.5 else "Safe"
                
            except:
                old_label = "Error"
            
            print(f"{filename:<25} {old_label:<15} {improved_result['label']:<15}")

if __name__ == "__main__":
    print("=== Testing Improved Model ===")
    
    # Test phone ring specifically
    test_phone_ring()
    
    print("\n" + "="*50)
    
    # Compare models
    compare_models()