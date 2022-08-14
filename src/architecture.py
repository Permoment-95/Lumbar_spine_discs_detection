import torch
from torch import nn
import segmentation_models_pytorch as smp


class LumbarHmModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(LumbarHmModel, self).__init__()
        self.backbone = self.model_setup(encoder, decoder)
        
    def model_setup(self, encoder_name, decoder_name):
        encoder_weights = 'noisy-student' if encoder_name.startswith('timm-efficientnet') else 'imagenet'
        
        model = getattr(smp, decoder_name)(
                                           encoder_name,
                                           in_channels=1,
                                           activation='sigmoid',
                                           classes=5, 
                                           encoder_weights=encoder_weights,
        )
    
        return model
        
    def forward(self, x):
        x = self.backbone(x)
        return x