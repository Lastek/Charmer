import cv2
import os

class VideoDenoiser:
    """Real-time video denoiser using ONNX Runtime or OpenCV fallback."""

    def __init__(self, onnx_model_path = None):
        # self.model_path = model_path
        self.use_onnx = False 
        if onnx_model_path is not None:
            self._init_onnx(onnx_model_path)
        else: print("Using OpenCV fastNlMeansDenoising (ONNX model not provided")

        self.session = None
    
    def _init_onnx(self, onnx_model_path):
        try:
            import onnxruntime as ort
            import numpy as np
            ONNX_AVAILABLE = True
            print("ONNXRT IS AVAILABLE")
        except ImportError:
            ONNX_AVAILABLE = False
            print("ONNXRT FAILED TO LOAD")

            """Initialize ONNX Runtime session."""

            if not os.path.exists(onnx_model_path):
                print(f"ONNX model not found at {onnx_model_path}, falling back to OpenCV")
            self.use_onnx = False
            return
        
        try:
            providers = ['CPUExecutionProvider']
            if 'DmlExecutionProvider' in ort.get_available_providers():
                providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
            
            self.session = ort.InferenceSession(onnx_model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            print(f"ONNX denoiser loaded from {onnx_model_path}")
        except Exception as e:
            print(f"Failed to load ONNX model: {e}, falling back to OpenCV")
            self.use_onnx = False
    
    def denoise(self, frame):
        """Denoise a single frame."""
        if self.use_onnx and self.session is not None:
            return self._denoise_onnx(frame)
        return self._denoise_opencv(frame)
    
    def _denoise_onnx(self, frame):
        """Denoise using ONNX model."""
        h, w = frame.shape[:2]
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32) / 255.0
        gray = np.expand_dims(gray, axis=(0, 1))
        
        noise_map = np.zeros((1, 1, h, w), dtype=np.float32)
        
        input_data = np.concatenate([gray, noise_map], axis=1)
        
        output = self.session.run(None, {self.input_name: input_data})[0]
        
        denoised = np.squeeze(output) * 255.0
        denoised = np.clip(denoised, 0, 255).astype(np.uint8)
        
        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    
    def _denoise_opencv(self, frame):
        """Denoise using OpenCV fastNlMeansDenoising."""
        return cv2.fastNlMeansDenoisingColored(
            frame, None, h=10, hColor=10, 
            templateWindowSize=7, searchWindowSize=21
        ) # type: ignore
