import os
import numpy as np
import torch
import torchaudio
import librosa
from scipy import signal

class AudioProcessor:
    def __init__(self, sample_rate=32000, window_size=1024, hop_size=320, 
                 mel_bins=64, fmin=50, fmax=14000):
        """
        Khởi tạo bộ xử lý âm thanh với các tham số cấu hình
        
        Args:
            sample_rate (int): Tần số lấy mẫu
            window_size (int): Kích thước cửa sổ FFT
            hop_size (int): Độ dịch chuyển giữa các cửa sổ
            mel_bins (int): Số lượng mel bins
            fmin (int): Tần số thấp nhất
            fmax (int): Tần số cao nhất
        """
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.mel_bins = mel_bins
        self.fmin = fmin
        self.fmax = fmax
        
        # Tạo mel filterbank
        self.melW = librosa.filters.mel(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax
        )
        self.melW = torch.Tensor(self.melW)

    def load_audio(self, audio_path):
        """
        Đọc file âm thanh và chuyển về tensor
        
        Args:
            audio_path (str): Đường dẫn đến file âm thanh
            
        Returns:
            torch.Tensor: Tensor âm thanh với shape (1, samples)
        """
        try:
            # Đọc file âm thanh
            waveform, sr = torchaudio.load(audio_path)
            
            # Chuyển về mono nếu là stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            # Resample nếu cần
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
                
            return waveform
            
        except Exception as e:
            print(f"Lỗi khi đọc file âm thanh: {e}")
            return None

    def preprocess_audio(self, audio_path):
        """
        Tiền xử lý âm thanh để đưa vào model
        
        Args:
            audio_path (str): Đường dẫn đến file âm thanh
            
        Returns:
            torch.Tensor: Tensor đã được xử lý với shape (1, mel_bins, time_steps)
        """
        # Đọc âm thanh
        waveform = self.load_audio(audio_path)
        if waveform is None:
            return None
            
        # Chuyển về numpy để xử lý
        waveform = waveform.numpy()[0]
        
        # Chuẩn hóa
        waveform = waveform / np.max(np.abs(waveform))
        
        # Tính STFT
        f, t, spectrogram = signal.spectrogram(
            waveform,
            fs=self.sample_rate,
            window='hann',
            nperseg=self.window_size,
            noverlap=self.window_size - self.hop_size,
            detrend=False,
            return_onesided=True,
            mode='magnitude'
        )
        
        # Chuyển về mel spectrogram
        mel_spectrogram = np.dot(self.melW.numpy(), spectrogram)
        
        # Log scale
        mel_spectrogram = np.log(mel_spectrogram + 1e-8)
        
        # Chuyển về tensor
        mel_spectrogram = torch.Tensor(mel_spectrogram).unsqueeze(0)
        
        return mel_spectrogram

    def split_audio(self, audio_path, segment_length=10):
        """
        Chia âm thanh thành các đoạn nhỏ
        
        Args:
            audio_path (str): Đường dẫn đến file âm thanh
            segment_length (int): Độ dài mỗi đoạn (giây)
            
        Returns:
            list: Danh sách các đoạn âm thanh
        """
        waveform = self.load_audio(audio_path)
        if waveform is None:
            return []
            
        # Tính số mẫu cho mỗi đoạn
        samples_per_segment = int(segment_length * self.sample_rate)
        
        # Chia thành các đoạn
        segments = []
        for i in range(0, waveform.shape[1], samples_per_segment):
            segment = waveform[:, i:i + samples_per_segment]
            if segment.shape[1] == samples_per_segment:  # Chỉ lấy đoạn đủ độ dài
                segments.append(segment)
                
        return segments

    def save_audio(self, waveform, output_path):
        """
        Lưu âm thanh ra file
        
        Args:
            waveform (torch.Tensor): Tensor âm thanh
            output_path (str): Đường dẫn lưu file
        """
        try:
            torchaudio.save(output_path, waveform, self.sample_rate)
        except Exception as e:
            print(f"Lỗi khi lưu file âm thanh: {e}")