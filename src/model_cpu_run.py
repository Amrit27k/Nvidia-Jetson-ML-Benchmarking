import torch
from torchvision import transforms
from PIL import Image
from enum import Enum
import time
import torchvision.models as models
import datetime
import pytz
import os
import logging
import sys
from jtop import jtop

logging.basicConfig(level=logging.DEBUG, filename="cpu_logs_exe_mobilenet_max.txt", filemode="a+",format="")

def read_classes():
    """
    Load the ImageNet class names.
    """
    with open("imagenet-classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    return categories

device = "cpu"
stop_execution = False
logging.info(f"Device: {device}")

    
def load_model(model_name, precision="float32"):
    if model_name == "resnet50":
        model = models.resnet50(weights= models.ResNet50_Weights.IMAGENET1K_V1)    
    elif model_name == "mobilenetv3":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    elif model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif model_name == "resnext50":
        model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
    elif model_name == "squeezenet":
        model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1)
    else:
        print(f"Model {model_name} not supported yet.")
        sys.exit(1)
    print("After loading model: ",datetime.datetime.now(pytz.timezone('Europe/London')))
    logging.info(f"After loading model: {datetime.datetime.now(pytz.timezone('Europe/London'))}")
    print(f"Loaded PyTorch {model_name} pretrained on ImageNet")
    logging.info(f"Loaded PyTorch {model_name} pretrained on ImageNet")

    if precision == "float16":
        model = model.half()
    return model
    
print("Before loading model: ",datetime.datetime.now(pytz.timezone('Europe/London')))
logging.info(f"Before loading model: {datetime.datetime.now(pytz.timezone('Europe/London'))}")
# Load the pre-trained Mobilenetv3 model
precision = "float16"
model = load_model("mobilenetv3", precision)
model = model.to(device)
model.eval()  # Set the model to evaluation mode
categories = read_classes()

from signal import signal, SIGINT
import cv2
def load_images_into_memory(images_dir, target_size, precision="float32"):
    images = []
    for image_file in sorted(os.listdir(images_dir)):
        image_path = os.path.join(images_dir, image_file)
        if not image_file.endswith(('.jpg', '.jpeg', '.png')):
            continue
        img = cv2.imread(image_path)
        if img is not None:
            img = cv2.resize(img, target_size)
            img = img.transpose(2, 0, 1)  # Change from HxWxC to CxHxW
            img = torch.tensor(img, dtype=torch.float32) / 255.0
            if precision == "float16":
                img = img.half()
            img = img.unsqueeze(0)  # Add batch dimension
            images.append(img)
    return images
    
def signal_handler(sig, frame):
    global stop_execution
    stop_execution = True

def measure_max_fps(model, images):
    total_time = 0
    frame_count = 0
    signal(SIGINT, signal_handler)

    print("Starting FPS measurement. Press Ctrl+C to stop...")
    
    with jtop() as jetson:
        if not jetson.ok():
            raise RuntimeError("jtop is not running or initialized properly.")
        
        power_readings = []
        with torch.no_grad():
            while not stop_execution:
                for img in images:
                    img = img.to(device)
                    start_time = time.time()
                    model(img)
                    total_time += time.time() - start_time
                    frame_count += 1
                    print(f"Jetson stats: {jetson.stats}")
                    logging.info(jetson.stats)
                    # Record power usage
                    power_readings.append(jetson.stats['Power TOT'])
                    if stop_execution:
                        break
    
    avg_power_usage = sum(power_readings) / len(power_readings) if power_readings else 0
    avg_inference_time = total_time / frame_count if frame_count > 0 else 0
    max_fps = 1 / avg_inference_time if avg_inference_time > 0 else 0
    print(f"\nTotal frames processed: {frame_count}")
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Maximum FPS achievable: {max_fps:.2f}")
    print(f"Average Total Power Usage: {avg_power_usage:.2f} mW")
    logging.info(f"\nTotal frames processed: {frame_count}")
    logging.info(f"Total time: {total_time:.4f} seconds")
    logging.info(f"Maximum FPS achievable: {max_fps:.2f}")
    logging.info(f"Average Total Power Usage: {avg_power_usage:.2f} mW")

target_size = (224, 224)
images_dir="./Images"
images = load_images_into_memory(images_dir, target_size, precision)
measure_max_fps(model, images)