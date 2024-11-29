import os
import time
import torch
from torchvision import models, transforms
from torch2trt import torch2trt
import sys
from PIL import Image
from jtop import jtop  # Import jtop for power monitoring
import logging
logging.basicConfig(level=logging.DEBUG, filename="gpu_logs_exe_squeezenet_trt_fps.txt", filemode="a+",format="")
# Load Pretrained Model
model_name = "squeezenet"

# Step 1: Load Pretrained Model
if model_name == "resnet50":
    model = models.resnet50(weights= models.ResNet50_Weights.IMAGENET1K_V1).eval().cuda()
elif model_name == "mobilenetv3":
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1).eval().cuda()
elif model_name == "resnet18":
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).eval().cuda()
elif model_name == "resnext50":
    model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1).eval().cuda()
elif model_name == "squeezenet":
    model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1).eval().cuda()
else:
    print(f"Model {model_name} not supported yet.")
    sys.exit(1)
logging.info(f"Loaded PyTorch {model_name} pretrained on ImageNet")
# Preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Convert model to TensorRT
def convert_to_trt(model, fp16=False):
    dummy_input = torch.randn(1, 3, 224, 224).cuda()
    trt_model = torch2trt(
        model,
        [dummy_input],
        fp16_mode=fp16,
        max_workspace_size=1 << 30,
    )
    return trt_model

# Convert model to TensorRT (FP16 example)
model_trt = convert_to_trt(model, fp16=False)

# Function to read images from a directory
def load_images_from_directory(directory, transform):
    images = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(directory, filename)
            image = Image.open(image_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).cuda()
            images.append((filename, input_tensor))
    return images

# Benchmark function to simulate FPS with power monitoring
def benchmark_fps_with_power(model, image_tensors, target_fps, num_frames=100):
    frame_times = []
    delay_per_frame = 1.0 / target_fps  # Time delay to simulate target FPS

    for _ in range(5):
        for _, input_tensor in image_tensors:
            with torch.no_grad():
                _ = model(input_tensor)
    print(f"\nBenchmarking at Target FPS: {target_fps}")
    logging.info(f"\nBenchmarking at Target FPS: {target_fps}\n")
    with jtop() as jetson:
        if not jetson.ok():
            raise RuntimeError("jtop is not running or initialized properly.")

        power_readings = []

        start_time = time.time()
        for i in range(num_frames):
            for _, input_tensor in image_tensors:
                frame_start = time.time()
                with torch.no_grad():
                    _ = model(input_tensor)  # Perform inference
                frame_end = time.time()

                frame_duration = frame_end - frame_start
                frame_times.append(frame_duration)

                # Enforce delay to match target FPS
                sleep_time = max(0, delay_per_frame - frame_duration)
                time.sleep(sleep_time)
                print(f"Jetson stats: {jetson.stats}")
                logging.info(jetson.stats)
                # Record power usage
                power_readings.append(jetson.stats['Power TOT'])  # Collect GPU power usage (in mW)

        end_time = time.time()
        total_time = end_time - start_time 
        achieved_fps = num_frames * len(image_tensors) / total_time
        avg_power_usage = sum(power_readings) / len(power_readings) if power_readings else 0

        print(f"\nTotal Time for {num_frames} x {len(image_tensors)} frames: {total_time:.2f} seconds")
        print(f"Achieved FPS: {achieved_fps:.2f} (Target: {target_fps})")
        print(f"Average Inference Time: {sum(frame_times) / len(frame_times):.4f} seconds/frame")
        print(f"Average Total Power Usage: {avg_power_usage:.2f} mW")
        logging.info(f"\nTotal Time for {num_frames} x {len(image_tensors)} frames: {total_time:.2f} seconds")
        logging.info(f"Achieved FPS: {achieved_fps:.2f} (Target: {target_fps})")
        logging.info(f"Average Inference Time: {sum(frame_times) / len(frame_times):.4f} seconds/frame")
        logging.info(f"Average Total Power Usage: {avg_power_usage:.2f} mW")
# Load images from the specified directory
image_directory = "./images/"  # Change this to your image directory
image_tensors = load_images_from_directory(image_directory, transform)

# Check if images were loaded
if len(image_tensors) == 0:
    print("No valid images found in the directory!")
else:
    print(f"Loaded {len(image_tensors)} images from directory: {image_directory}")

    # Test model with FPS ranging from 1 to 1000
    fps_values = [1, 10, 30, 60, 120, 240, 500, 1000]  # Define FPS to test
    for fps in fps_values:
        benchmark_fps_with_power(model_trt, image_tensors, target_fps=fps, num_frames=20)
        time.sleep(30)
        logging.info("\n\n")