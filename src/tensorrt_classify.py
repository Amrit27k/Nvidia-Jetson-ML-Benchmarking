import datetime
import logging
import os
import pytz
import sys
import time
import torch
from torchvision import models, transforms
from torch2trt import torch2trt
from PIL import Image
from signal import signal, SIGINT

logging.basicConfig(level=logging.DEBUG, filename="gpu_logs_exe_resnet50_trt_max.txt", filemode="a+",format="")
# Load ImageNet class names
with open("imagenet-classes.txt", "r") as f:
    class_names = [line.strip() for line in f]

def signal_handler(sig, frame):
    global stop_execution
    stop_execution = True

# Preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

logging.info(f"Before loading model: {datetime.datetime.now(pytz.timezone('Europe/London'))}")
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
logging.info(f"After loading model: {datetime.datetime.now(pytz.timezone('Europe/London'))}")
print(f"Loaded PyTorch {model_name} pretrained on ImageNet")
logging.info(f"Loaded PyTorch {model_name} pretrained on ImageNet")

# Step 2: Convert PyTorch model to TensorRT model
# Function to create FP32 or FP16 TensorRT models
def convert_to_trt(model, fp16=False):
    dummy_input = torch.randn(1, 3, 224, 224).cuda()  # Dummy input for conversion
    trt_model = torch2trt(
        model, [dummy_input], fp16_mode=fp16
    )  # Use fp16_mode=True for FP16 precision
    return trt_model


model_trt_fp32 = convert_to_trt(model, fp16=False)
model_trt_fp16 = convert_to_trt(model, fp16=True)
print("Models converted successfully.")
# Directory containing images
image_dir = "./images"
image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith(('.jpg', '.jpeg', '.png'))]

# Step 3: Predict Function Using TensorRT Model
def predict(image_path):
    logging.info(f"Loading image: {datetime.datetime.now(pytz.timezone('Europe/London'))}")
    image = Image.open(image_path).convert("RGB")
    logging.info(f"Image transformation: {datetime.datetime.now(pytz.timezone('Europe/London'))}")
    input_tensor = transform(image).unsqueeze(0).cuda()  # Add batch dimension

    with torch.no_grad():
        print("During prediction: ", datetime.datetime.now(pytz.timezone('Europe/London')))
        logging.info(f"During prediction: {datetime.datetime.now(pytz.timezone('Europe/London'))}")
        start_time = time.time()
        output = model_trt(input_tensor)
        end_time = time.time()
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get top-5 predictions
    top5_prob, top5_indices = torch.topk(probabilities, 5)
    top5_classes = [class_names[idx] for idx in top5_indices.cpu().numpy()]
    top5_prob = top5_prob.cpu().numpy()
    infer_time = end_time - start_time
    print(f'Inference time: {infer_time:.4f} seconds')
    logging.info(f'Inference time: {infer_time:.4f} seconds')

    return list(zip(top5_classes, top5_prob))

def predict_trt(input_tensor, model_trt):
    with torch.no_grad():
        start = time.time()
        output = model_trt(input_tensor)
        print(f"infer time: {time.time()-start}")
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return probabilities

def measure_fps(model_trt, mode="FP32"):
    print("Measuring FPS...")
    # Preload and preprocess all images
    preprocessed_images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).cuda()  # Add batch dimension
        preprocessed_images.append(input_tensor)

    # Warm-up loop to stabilize GPU performance
    for _ in range(10):
        for input_tensor in preprocessed_images:
            _ = predict_trt(input_tensor, model_trt)

    # Measure inference time over multiple iterations
    num_iterations = 1000
    start_time = time.time()
    count = 10
    for _ in range(num_iterations):
        for input_tensor in preprocessed_images:
            _ = predict_trt(input_tensor, model_trt)
            count += 1
            print(count)
    end_time = time.time()

    # Calculate FPS
    total_images = len(preprocessed_images) * num_iterations
    total_time = end_time - start_time
    fps = total_images / total_time

    print(f"Total Images Processed: {total_images}")
    print(f"Total Time: {total_time:.4f} seconds")
    print(f"Max FPS in {mode}: {fps:.2f}")
    return fps

# Step 4: Run Predictions on All Images in the Directory
#for image_path in image_paths:
#    print(f"Image: {image_path}")
#    predictions = predict(image_path)
#    for i, (class_name, prob) in enumerate(predictions):
#       print(f"  Top-{i+1}: {class_name} ({prob:.4f})")
#        if i == 0:
#            logging.info(f"Top-{i+1}: {class_name} ({prob:.4f})")
#    print()

fps_fp32 = measure_fps(model_trt_fp32, mode="FP32")
print("FP32 successfully completed!\n")

time.sleep(20)
fps_fp16 = measure_fps(model_trt_fp16, mode="FP16")