import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.audio_utils import AudioProcessor
from config.settings import DANGEROUS_SOUNDS
from torchlibrosa.stft import Spectrogram, LogmelFilterBank

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x

class Cnn14(nn.Module):
    def __init__(self, sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527):
        super(Cnn14, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
 
    def forward(self, input):
        """
        Input: (batch_size, data_length)
        """
        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        
        x = torch.mean(x, dim=3)
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc_audioset(x)
        
        return x

class DangerousSoundDetector:
    def __init__(self, model_path, threshold=0.5, device='cpu'):
        self.device = device
        self.threshold = threshold
        self.dangerous_sounds = DANGEROUS_SOUNDS
        self.audio_processor = AudioProcessor(sample_rate=32000, window_size=1024, hop_size=320, 
                                           mel_bins=64, fmin=50, fmax=14000)
        self.model = self.load_model(model_path)
        
    def load_model(self, model_path):
        # Khởi tạo model
        model = Cnn14()
        
        # Load state dict
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Nếu checkpoint là dictionary và có key 'model'
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        # Nếu checkpoint là dictionary và có key 'state_dict'
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        # Nếu checkpoint là state dict trực tiếp
        else:
            state_dict = checkpoint
            
        # Load state dict vào model
        model.load_state_dict(state_dict)
        
        # Chuyển model sang chế độ evaluation
        model.eval()
        model = model.to(self.device)
            
        return model
        
    def detect(self, audio_path):
        # Tiền xử lý âm thanh
        waveform = self.audio_processor.load_audio(audio_path)
        if waveform is None:
            return {"is_dangerous": False, "dangerous_sounds": []}
            
        # Dự đoán
        with torch.no_grad():
            waveform = waveform.to(self.device)
            predictions = self.model(waveform)
            
        # Phân tích kết quả
        dangerous_results = []
        for sound_type, labels in self.dangerous_sounds.items():
            for label in labels:
                if label in predictions and predictions[label] > self.threshold:
                    dangerous_results.append({
                        "type": sound_type,
                        "label": label,
                        "confidence": predictions[label]
                    })
                    
        return {
            "is_dangerous": len(dangerous_results) > 0,
            "dangerous_sounds": dangerous_results
        }