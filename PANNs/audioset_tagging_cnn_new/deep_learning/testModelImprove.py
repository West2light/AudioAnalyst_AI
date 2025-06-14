import numpy as np
import librosa
import tensorflow as tf
import joblib
import os
import glob

class ImprovedAudioTester:
    def __init__(self):
        """Load improved model"""
        self.model = tf.keras.models.load_model("audio_classification_model_improved.keras")
        self.scaler = joblib.load("scaler_improved.pkl")
        print("✅ Improved model loaded!")
    
    def extract_improved_features(self, audio_path):
        """Extract features giống như training data improved"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=22050)
            print(f"   Audio loaded: {len(audio)/sr:.2f}s, {sr}Hz")
            
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
            
            print(f"   Extracted {len(all_features)} raw features")
            
            # Calculate statistics (10 features như training)
            improved_features = [
                np.mean(all_features),      # 0: mean
                np.std(all_features),       # 1: std
                np.min(all_features),       # 2: min
                np.max(all_features),       # 3: max
                np.median(all_features),    # 4: median
                np.percentile(all_features, 25),  # 5: Q1
                np.percentile(all_features, 75),  # 6: Q3
                len(all_features),          # 7: feature count
                np.sum(all_features > 0),   # 8: positive count
                np.sum(all_features == 0)   # 9: zero count
            ]
            
            return np.array(improved_features)
            
        except Exception as e:
            print(f"   ❌ Error extracting features: {e}")
            return None
    
    def predict_audio(self, audio_path, verbose=True):
        """Predict cho một file audio"""
        
        if verbose:
            print(f"\n🔊 Testing: {os.path.basename(audio_path)}")
        
        # Extract features
        features = self.extract_improved_features(audio_path)
        
        if features is None:
            return {'error': 'Could not extract features'}
        
        if verbose:
            print(f"   Features: {features}")
        
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
            'features': features.tolist()
        }
        
        if verbose:
            print(f"   🎯 Result: {result['label']} (prob: {result['probability']:.3f}, conf: {result['confidence']:.3f})")
        
        return result
    
    def test_directory(self, directory_path):
        """Test tất cả audio files trong directory"""
        
        print(f"\n📁 Testing directory: {directory_path}")
        
        # Find audio files
        audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.aac']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(glob.glob(os.path.join(directory_path, ext)))
            audio_files.extend(glob.glob(os.path.join(directory_path, '**', ext), recursive=True))
        
        if not audio_files:
            print(f"   No audio files found")
            return []
        
        print(f"   Found {len(audio_files)} audio files")
        
        # Test each file
        results = []
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n   [{i}/{len(audio_files)}]", end=" ")
            result = self.predict_audio(audio_file, verbose=True)
            
            if 'error' not in result:
                results.append(result)
        
        return results
    
    def analyze_results(self, results):
        """Phân tích kết quả test"""
        
        if not results:
            print("No results to analyze")
            return
        
        print(f"\n📊 Analysis of {len(results)} files:")
        
        # Count predictions
        safe_count = sum(1 for r in results if r['label'] == 'Safe')
        danger_count = len(results) - safe_count
        
        print(f"   Safe: {safe_count} files ({safe_count/len(results)*100:.1f}%)")
        print(f"   Danger: {danger_count} files ({danger_count/len(results)*100:.1f}%)")
        
        # Confidence statistics
        confidences = [r['confidence'] for r in results]
        probabilities = [r['probability'] for r in results]
        
        print(f"\n   Confidence stats:")
        print(f"     Mean: {np.mean(confidences):.3f}")
        print(f"     Min:  {np.min(confidences):.3f}")
        print(f"     Max:  {np.max(confidences):.3f}")
        
        # Low confidence predictions
        low_conf = [r for r in results if r['confidence'] < 0.6]
        if low_conf:
            print(f"\n   ⚠️  Low confidence predictions ({len(low_conf)} files):")
            for r in low_conf:
                print(f"     {r['filename']}: {r['label']} (conf: {r['confidence']:.3f})")
        
        # Detailed results
        print(f"\n📋 Detailed Results:")
        print(f"{'Filename':<30} {'Prediction':<10} {'Probability':<12} {'Confidence':<10}")
        print("-" * 65)
        
        # Sort by confidence descending
        sorted_results = sorted(results, key=lambda x: x['confidence'], reverse=True)
        
        for r in sorted_results:
            filename = r['filename'][:28] + ".." if len(r['filename']) > 28 else r['filename']
            print(f"{filename:<30} {r['label']:<10} {r['probability']:<12.3f} {r['confidence']:<10.3f}")

def main():
    """Main test function"""
    
    print("🧪 Testing Improved Audio Classification Model with Real Audio Files")
    print("=" * 80)
    
    # Initialize tester
    tester = ImprovedAudioTester()
    
    # Test specific files
    test_files = [
        r"E:\DOAN1\PANNs\audioset_tagging_cnn_new\resources\R9_ZSCveAHg_7s.mp3",
        r"E:\DOAN1\PANNs\audioset_tagging_cnn_new\resources\audio_dog_1.wav",
        r"E:\DOAN1\PANNs\audioset_tagging_cnn_new\resources\yt1s.com - Tiếng xe ngoài đường âm thanh xe chạy ngoài đường tiếng đường phố nghe hết buồn0528887777.mp3",
    ]
    
    print("\n🎯 Testing specific files:")
    specific_results = []
    
    for audio_file in test_files:
        if os.path.exists(audio_file):
            result = tester.predict_audio(audio_file, verbose=True)
            if 'error' not in result:
                specific_results.append(result)
        else:
            print(f"   ❌ File not found: {audio_file}")
    
    # Test resources directory
    resources_dir = r"E:\DOAN1\PANNs\audioset_tagging_cnn_new\resources"
    if os.path.exists(resources_dir):
        directory_results = tester.test_directory(resources_dir)
        
        # Combine results
        all_results = specific_results + directory_results
        
        # Remove duplicates
        seen_files = set()
        unique_results = []
        for r in all_results:
            if r['filename'] not in seen_files:
                unique_results.append(r)
                seen_files.add(r['filename'])
        
        # Analyze
        tester.analyze_results(unique_results)
    
    else:
        print(f"\n❌ Resources directory not found: {resources_dir}")

if __name__ == "__main__":
    main()