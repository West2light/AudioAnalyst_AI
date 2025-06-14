import os
import glob
from extract_audio_features import extract_simple_audio_features
from audio_classifier_api import AudioSafetyClassifier

def test_audio_directory(directory_path):
    """Test tất cả file audio trong thư mục"""
    
    # Tìm tất cả file audio
    audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(directory_path, ext)))
        audio_files.extend(glob.glob(os.path.join(directory_path, '**', ext), recursive=True))
    
    if not audio_files:
        print(f"No audio files found in {directory_path}")
        return
    
    print(f"Found {len(audio_files)} audio files")
    
    # Load classifier
    classifier = AudioSafetyClassifier()
    
    # Test từng file
    results = []
    for audio_file in audio_files:
        print(f"\n🔊 Testing: {os.path.basename(audio_file)}")
        
        features = extract_simple_audio_features(audio_file)
        if features is not None:
            result = classifier.predict(features, return_details=True)
            result['filename'] = os.path.basename(audio_file)
            results.append(result)
            
            print(f"   Result: {result['label']} (confidence: {result['confidence']:.2f})")
        else:
            print(f"   ❌ Failed to extract features")
    
    # Summary
    print(f"\n📊 Summary of {len(results)} files:")
    safe_count = sum(1 for r in results if r['label'] == 'Safe')
    danger_count = len(results) - safe_count
    
    print(f"   Safe: {safe_count}")
    print(f"   Danger: {danger_count}")
    
    return results

if __name__ == "__main__":
    # Test thư mục resources
    resources_dir = r"E:\DOAN1\PANNs\audioset_tagging_cnn_new\resources"
    results = test_audio_directory(resources_dir)