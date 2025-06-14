import pandas as pd
import numpy as np
import pickle
import os

def load_full_embeddings():
    """
    Load embeddings đầy đủ từ file gốc (không bị cắt ngắn)
    """
    try:
        # Thử load từ file pickle nếu có
        embeddings_file = "E:\\DOAN1\\PANNs\\audioset_tagging_cnn_new\\embeddings.pkl"
        if os.path.exists(embeddings_file):
            with open(embeddings_file, 'rb') as f:
                data = pickle.load(f)
            return data
    except:
        pass
    
    # Nếu không có file pickle, cần tạo lại embeddings từ audio files
    print("Cần tạo lại embeddings từ audio files...")
    return None

def find_audio_file(filename):
    """Tìm file audio trong các thư mục"""
    search_dirs = [
        "E:\\DOAN1\\PANNs\\audioset_tagging_cnn_new\\resources",
        "E:\\DOAN1\\PANNs\\audioset_tagging_cnn_new\\audio",
        "E:\\DOAN1\\PANNs\\audioset_tagging_cnn_new"
    ]
    
    for directory in search_dirs:
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                if filename in files:
                    return os.path.join(root, filename)
    return None

# Giải pháp tạm thời: Sử dụng nhiều features hơn từ dữ liệu hiện có
def improve_current_features():
    """Cải thiện features từ dữ liệu CSV hiện có"""
    
    print("Loading CSV data...")
    df = pd.read_csv("E:\\DOAN1\\PANNs\\audioset_tagging_cnn_new\\audio_embeddings.csv")
    print(f"Loaded {len(df)} rows")
    
    def extract_more_features(row):
        all_values = []
        
        for i in range(15):
            feat_col = f"feat_{i}"
            feat_data = str(row[feat_col])
            
            # Parse tất cả số có thể
            feat_data = feat_data.strip('[]').replace('...', ' ')
            values = []
            for val in feat_data.split():
                try:
                    values.append(float(val))
                except:
                    continue
            all_values.extend(values)
        
        # Lấy stats từ tất cả values
        if len(all_values) > 0:
            all_values = np.array(all_values)
            features = [
                np.mean(all_values),
                np.std(all_values),
                np.min(all_values),
                np.max(all_values),
                np.median(all_values),
                np.percentile(all_values, 25),
                np.percentile(all_values, 75),
                len(all_values),  # Số lượng features có được
                np.sum(all_values > 0),  # Số features dương
                np.sum(all_values == 0)   # Số features = 0
            ]
            return np.array(features)
        else:
            return np.zeros(10)
    
    # Process all rows
    print("Extracting improved features...")
    embeddings_list = []
    labels_list = []
    
    for idx, row in df.iterrows():
        features = extract_more_features(row)
        embeddings_list.append(features)
        labels_list.append(row["label"])
        
        if idx % 50 == 0:
            print(f"Processed {idx}/{len(df)}")
    
    # Save improved features
    X = np.array(embeddings_list)
    y = np.array(labels_list)
    
    np.save("X_features_improved.npy", X)
    np.save("y_labels_improved.npy", y)
    
    print(f"Saved improved features: X shape {X.shape}, y shape {y.shape}")
    print(f"Feature statistics:")
    for i in range(X.shape[1]):
        print(f"  Feature {i}: mean={X[:, i].mean():.3f}, std={X[:, i].std():.3f}")
    
    return X, y

if __name__ == "__main__":
    print("=== Starting Feature Improvement ===")
    
    # Thử cải thiện features từ dữ liệu hiện có
    print("\n=== Improving current features ===")
    try:
        X_improved, y_improved = improve_current_features()
        
        print(f"\n✅ Success!")
        print(f"Improved features statistics:")
        print(f"  Shape: {X_improved.shape}")
        print(f"  Labels distribution: {np.bincount(y_improved)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()