from ultralytics import YOLO

model = YOLO('best.pt')

model.export(format='onnx', device='cpu')

print("Successfully exported 'best.pt' to 'best.onnx' using CPU. You can now use this file in the main script.")