import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

MODEL = 'ffdnet_color.pth'


class PixelUnShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        return nn.functional.pixel_unshuffle(x, self.upscale_factor)


class FFDNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=96):
        super(FFDNet, self).__init__()
        sf = 2

        self.m_down = PixelUnShuffle(upscale_factor=sf)

        layers = OrderedDict()
        layers['0'] = nn.Conv2d(in_nc*sf*sf+1, nc, 3, padding=1, bias=True)
        layers['1'] = nn.ReLU(inplace=True)
        for i in range(1, 11):
            layers[str(i*2)] = nn.Conv2d(nc, nc, 3, padding=1, bias=True)
            layers[str(i*2+1)] = nn.ReLU(inplace=True)
        layers['22'] = nn.Conv2d(nc, out_nc*sf*sf, 3, padding=1, bias=True)
        
        self.model = nn.Sequential(layers)
        nn.Sequential()
        self.m_up = nn.PixelShuffle(upscale_factor=sf)

    def forward(self, x, sigma):
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h/2)*2-h)
        paddingRight = int(np.ceil(w/2)*2-w)
        x = torch.nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x = self.m_down(x)
        m = sigma.repeat(1, 1, x.size()[-2], x.size()[-1])
        x = torch.cat((x, m), 1)
        x = self.model(x)
        x = self.m_up(x)
        
        x = x[..., :h, :w]
        return x


state_dict = torch.load(MODEL, map_location='cpu')

model = FFDNet(in_nc=3, out_nc=3, nc=96)
model.load_state_dict(state_dict, strict=False)
model.eval()

dummy_input = torch.randn(1, 3, 256, 256)
dummy_sigma = torch.randn(1, 1, 1, 1)

torch.onnx.export(
    model,
    (dummy_input, dummy_sigma),
    'ffdnet_color.onnx',
    input_names=['input', 'sigma'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch', 2: 'height', 3: 'width'},
        'sigma': {0: 'batch'},
        'output': {0: 'batch', 2: 'height', 3: 'width'}
    },
    opset_version=14
)
print("ONNX model saved to ffdnet_color.onnx")
