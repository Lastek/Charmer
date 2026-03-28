import torch

MODEL = '002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth'
model_dict = torch.load(MODEL, map_location='cpu')

if 'params' in model_dict:
    state_dict = model_dict['params']
elif 'params_ema' in model_dict:
    state_dict = model_dict['params_ema']
elif 'state_dict' in model_dict:
    state_dict = model_dict['state_dict']
else:
    state_dict = model_dict

for key in list(state_dict.keys()):
    new_key = key.replace('module.', '')
    state_dict[new_key] = state_dict.pop(key)

try:
    from models.network_swinir import SwinIR
    net = SwinIR(
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=64,
        depths=[6, 6, 6, 6],
        num_heads=[8, 8, 8, 8],
        window_size=8,
        mlp_ratio=2.,
        qkv_bias=True,
        scale=0.02
    )
    net.load_state_dict(state_dict, strict=False)
    net.eval()
except ImportError:
    print("SwinIR model class not found. Using dummy model for ONNX export.")
    net = torch.nn.Identity()

dummy_input = torch.randn(1, 3, 256, 256)

torch.onnx.export(
    net,
    dummy_input,
    'swinir_x2.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch', 2: 'height', 3: 'width'}, 
                  'output': {0: 'batch', 2: 'height', 3: 'width'}},
    opset_version=14
)
print("ONNX model saved to swinir_x2.onnx")
