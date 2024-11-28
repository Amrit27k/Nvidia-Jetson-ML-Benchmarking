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

logging.basicConfig(level=logging.DEBUG, filename="gpu_logs_exe_mobilenet_max.txt", filemode="a+",format="")

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
logging.info(f"Device: {device}")


if device=="cuda":
    try:
        from torch2trt import torch2trt as trt
    except ImportError:
        logging.warning("torch-tensorrt is not installed. Running on CPU mode only.")
        CUDA_AVAILABLE = False

def read_classes():
    """
    Load the ImageNet class names.
    """
    with open("imagenet-classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    return categories


def load_model(model_name, device):
    if model_name == "resnet50":
        model = models.resnet50(weights= models.ResNet50_Weights.IMAGENET1K_V1).eval().to(device)
    elif model_name == "mobilenetv3":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1).to(device)
    elif model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)
    elif model_name == "resnext50":
        model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1).to(device)
    elif model_name == "squeezenet":
        model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1).to(device)
    else:
        print(f"Model {model_name} not supported yet.")
        sys.exit(1)

    print("After loading model: ",datetime.datetime.now(pytz.timezone('Europe/London')))
    logging.info(f"After loading model: {datetime.datetime.now(pytz.timezone('Europe/London'))}")
    print(f"Loaded PyTorch {model_name} pretrained on ImageNet")
    logging.info(f"Loaded PyTorch {model_name} pretrained on ImageNet")
    model = model.to(device).eval()

    # Compile the TorchScript model with TensorRT
    if device=="cuda":
        model = trt(
            model,
            inputs=[torch.ones((1,3,224,224)).cuda()]
        )
    return model

def preprocess_image(image_path, device):
    print("Loading image: ",datetime.datetime.now(pytz.timezone('Europe/London')))
    logging.info(f"Loading image: {datetime.datetime.now(pytz.timezone('Europe/London'))}")
    input_image = Image.open(image_path).convert('RGB')
    if input_image is None:
        print(f"Failed to load image: {image_path}")
        return None, None

    logging.info(f"Image transformation: {datetime.datetime.now(pytz.timezone('Europe/London'))}")
    # Define the image transformation pipeline
    width, height = input_image.size  # Height x Width
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Apply transformations to the input image
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0).to(device)
    return input_batch, width, height


def predict(model, image_path, device):
        """
        Run prediction on the input data using the TensorRT model.

        :param input_data: Data to run the prediction on.
        :param is_benchmark: If True, the prediction is part of a benchmark run.
        :return: Top predictions based on the probabilities.
        """
        # image_path = os.path.join(images_dir, image_file)
        img, width, height = preprocess_image(image_path, device)
        with torch.no_grad():
            print("During prediction: ", datetime.datetime.now(pytz.timezone('Europe/London')))
            logging.info(f"During prediction: {datetime.datetime.now(pytz.timezone('Europe/London'))}")
            start_time = time.time()
            outputs = model(img.to(device))
            h_outputs = outputs.cpu().numpy()
            end_time = time.time()

        predictions = torch.from_numpy(h_outputs).cpu().numpy()
        predicted_class = predictions.argmax()

        inference_time = end_time - start_time
        # Compute the softmax probabilities
        #prob = torch.nn.functional.softmax(outputs[0], dim=0)
        # check the top category that are predicted
        #top5_prob, top5_catid = torch.topk(prob, 1)

        y_pred_prob, y_pred_k = outputs.topk(k=5, dim=1)
        y_pred_k = y_pred_k.t()
        print(y_pred_k)
        print(f"Image: {image_path}")
        logging.info(f"Image: {image_path}")
        logging.info(f"Resolution: {height}x{width}")
        print(f'Inference time: {inference_time:.4f} seconds')
        #print(f"Accuracy: {y_pred_prob[0].item()*100:.3f}%")
        print(f"Argmax Pred class:{categories[predicted_class]}")
        #print(f"TopK Predicted class:{categories[y_pred_k[0]]}")


print("Before loading model: ",datetime.datetime.now(pytz.timezone('Europe/London')))
logging.info(f"Before loading model: {datetime.datetime.now(pytz.timezone('Europe/London'))}")
# Load the pre-trained Mobilenetv3 model
model = load_model("mobilenetv3", device)
# Set the model to evaluation mode
categories = read_classes()
img_path = "./images/cat_0.jpg"
predict(model, img_path, device)