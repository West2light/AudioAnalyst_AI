#Tổ chức thư mục dữ liệu viết filterAudioToDangerousAndSafe.py của ESC-50 để có được dataset/Danger vs dataset/Safe

import pandas as pd
import os
import shutil

ESC50_CSV = "meta/esc50.csv"
ESC50_AUDIO = "audio/"
OUTPUT_DIR = "dataset/"

# Sửa nhóm này theo ý mình
danger_labels = ['dog', 'scream', 'crying_baby', 'siren', 'gun_shot', 'chainsaw']
safe_labels = ['rain', 'clock_tick', 'speech', 'wind', 'sea_waves', 'keyboard_typing']

# Tạo thư mục đầu ra
os.makedirs(os.path.join(OUTPUT_DIR, 'danger'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'safe'), exist_ok=True)

df = pd.read_csv(ESC50_CSV)

for _, row in df.iterrows():
    filename = row['filename']
    category = row['category']
    
    src_path = os.path.join(ESC50_AUDIO, filename)

    if category in danger_labels:
        dst_path = os.path.join(OUTPUT_DIR, 'danger', filename)
        shutil.copy(src_path, dst_path)
    elif category in safe_labels:
        dst_path = os.path.join(OUTPUT_DIR, 'safe', filename)
        shutil.copy(src_path, dst_path)

print("✅ Đã chia xong ESC-50 thành 2 nhóm: danger / safe.")
