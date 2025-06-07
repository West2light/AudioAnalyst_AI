import os
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm
import sys

# Thêm đường dẫn thư mục gốc vào sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from pytorch.models import Cnn14
# --- CONFIG ---
DATASET_PATH = os.path.join(current_dir, "dataset/")
OUTPUT_CSV =   os.path.join(current_dir, "audio_embeddings.csv")
SAMPLE_RATE = 32000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# --- Load pretrained model ---
model = Cnn14(
    sample_rate=32000, 
    window_size=1024, 
    hop_size=320, 
    mel_bins=64, 
    fmin=50, 
    fmax=14000,
    classes_num=527  # Thêm tham số này
)
model_path = os.path.join(current_dir, "Cnn14_mAP=0.431.pth")
checkpoint = torch.load(model_path, map_location=DEVICE)
model.load_state_dict(checkpoint['model'])
model.to(DEVICE)
model.eval()

def load_audio(file_path):
    """Load và tiền xử lý audio file"""
    waveform, sr = torchaudio.load(file_path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    return waveform.mean(dim=0)  # Chuyển về mono

# def extract_embedding(audio_tensor):
#     """Trích xuất embedding từ model"""
#     audio_tensor = audio_tensor.unsqueeze(0).to(DEVICE)
#     with torch.no_grad():
#         # Lấy output từ fc1 layer (2048 dimensions)
#         x = model.forward(audio_tensor)
#         embedding = x.squeeze().cpu().numpy()
#     return embedding
def extract_embedding(audio_tensor):
    """Trích xuất embedding từ model"""
    audio_tensor = audio_tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output_dict = model.forward(audio_tensor)
        embedding = output_dict['embedding'].squeeze().cpu().numpy()
    return embedding

def main():
    rows = []
    
    # Duyệt qua các thư mục safe và danger
    for label_name in ['safe', 'danger']:
        label = 0 if label_name == 'safe' else 1
        folder = os.path.join(DATASET_PATH, label_name)
        
        print(f"\nĐang xử lý thư mục {label_name}...")
        
        # Duyệt qua từng file audio
        for filename in tqdm(os.listdir(folder)):
            if not filename.endswith(".wav"):
                continue
                
            file_path = os.path.join(folder, filename)
            
            try:
                # Load và xử lý audio
                audio = load_audio(file_path)
                
                # Trích xuất embedding
                emb = extract_embedding(audio)
                
                # Lưu kết quả
                row = {
                    'filename': filename,
                    'label': label,
                    **{f'feat_{i}': v for i, v in enumerate(emb)}
                }
                rows.append(row)
                
            except Exception as e:
                print(f"Lỗi xử lý {file_path}: {e}")
    
    # Lưu kết quả vào CSV
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Đã lưu embedding vào {OUTPUT_CSV}")
    print(f"Tổng số file đã xử lý: {len(rows)}")

if __name__ == "__main__":
    main()