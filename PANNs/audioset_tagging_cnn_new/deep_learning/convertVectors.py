import pandas as pd
import numpy as np
import ast

# Load dữ liệu
df = pd.read_csv("E:\\DOAN1\\PANNs\\audioset_tagging_cnn_new\\audio_embeddings.csv")

print(f"DataFrame shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Hàm xử lý mới - không dùng ast.literal_eval
def parse_embedding_new(row):
    try:
        vectors = []
        
        for i in range(15):
            feat_col = f"feat_{i}"
            feat_data = row[feat_col]
            
            # Nếu là string, cần parse khác
            if isinstance(feat_data, str):
                # Loại bỏ dấu ngoặc vuông và split
                feat_data = feat_data.strip('[]')
                # Thay thế ... bằng khoảng trống và split
                feat_data = feat_data.replace('...', ' ')
                # Split và convert thành float
                values = []
                for val in feat_data.split():
                    try:
                        values.append(float(val))
                    except ValueError:
                        continue  # Bỏ qua các giá trị không parse được
                vector = np.array(values)
            else:
                # Nếu đã là array hoặc list
                vector = np.array(feat_data)
            
            print(f"feat_{i} shape: {vector.shape}")
            vectors.append(vector)
        
        # Kiểm tra tất cả vectors có cùng shape không
        shapes = [v.shape for v in vectors]
        if len(set(shapes)) > 1:
            print(f"Warning: Different shapes found: {shapes}")
            # Tìm shape phổ biến nhất
            from collections import Counter
            most_common_shape = Counter(shapes).most_common(1)[0][0]
            vectors = [v for v in vectors if v.shape == most_common_shape]
            print(f"Using {len(vectors)} vectors with shape {most_common_shape}")
        
        if len(vectors) > 0:
            result = np.mean(vectors, axis=0)
            return result
        else:
            return None
            
    except Exception as e:
        print(f"Error processing row: {e}")
        return None

# Test với hàng đầu tiên
print("\n=== Testing first row ===")
first_embedding = parse_embedding_new(df.iloc[0])

if first_embedding is not None:
    print(f"Success! First embedding shape: {first_embedding.shape}")
    
    # Xử lý toàn bộ dataset
    embeddings_list = []
    labels_list = []
    
    for idx, row in df.iterrows():
        embedding = parse_embedding_new(row)
        if embedding is not None:
            embeddings_list.append(embedding)
            labels_list.append(row["label"])
        
        if idx % 50 == 0:
            print(f"Processed {idx}/{len(df)} rows, valid: {len(embeddings_list)}")
    
    # Lưu kết quả
    if len(embeddings_list) > 0:
        X = np.array(embeddings_list)
        y = np.array(labels_list)
        
        np.save("X_features.npy", X)
        np.save("y_labels.npy", y)
        
        print(f"Đã lưu X_features.npy với shape: {X.shape}")
        print(f"Đã lưu y_labels.npy với shape: {y.shape}")
        print(f"Labels distribution: {np.bincount(y)}")
    else:
        print("No valid embeddings found!")
        
else:
    print("Failed to parse first row.")
    # Thử cách khác - đọc trực tiếp từ CSV với converters
    print("Trying alternative method...")
    
    def converter(x):
        try:
            # Loại bỏ ellipsis và parse
            x = str(x).strip('[]').replace('...', ' ')
            values = [float(val) for val in x.split() if val.strip()]
            return np.array(values)
        except:
            return np.array([])
    
    # Đọc lại với converters
    converters = {f'feat_{i}': converter for i in range(15)}
    df_new = pd.read_csv("E:\\DOAN1\\PANNs\\audioset_tagging_cnn_new\\audio_embeddings.csv", 
                         converters=converters)
    
    print("Testing with converters...")
    test_vector = df_new.iloc[0]['feat_0']
    print(f"Converted vector shape: {test_vector.shape}")