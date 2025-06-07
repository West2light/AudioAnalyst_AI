import os
import sys
import torch
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.detector import DangerousSoundDetector
from utils.audio_utils import AudioProcessor        
from config.settings import DANGEROUS_SOUNDS, MODEL_CONFIG

def main():
    parser = argparse.ArgumentParser(description='Phát hiện âm thanh nguy hiểm')
    parser.add_argument('--audio_path', type=str, required=True, help=r'E:\DOAN1\PANNs\audioset_tagging_cnn\resources\audio_dog_1.mp3')
    parser.add_argument('--model_path', type=str, default='Cnn14_mAP=0.431.pth', help=r'E:\DOAN1\PANNs\audioset_tagging_cnn\Cnn14_mAP=0.431.pth')
    parser.add_argument('--threshold', type=float, default=0.5, help='Ngưỡng phát hiện')
    parser.add_argument('--cuda', action='store_true', help='Sử dụng GPU nếu có')
    
    args = parser.parse_args()
    
    # Khởi tạo detector
    detector = DangerousSoundDetector(
        model_path=args.model_path,
        threshold=args.threshold,
        device='cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    )
    
    # Phát hiện âm thanh
    result = detector.detect(args.audio_path)
    
    # In kết quả
    if result["is_dangerous"]:
        print("\n⚠️ PHÁT HIỆN ÂM THANH NGUY HIỂM!")
        print("\nChi tiết:")
        for sound in result["dangerous_sounds"]:
            print(f"- {sound['type']}: {sound['label']} ({sound['confidence']:.2f})")
    else:
        print("\n✅ KHÔNG PHÁT HIỆN ÂM THANH NGUY HIỂM")

if __name__ == '__main__':
    main()