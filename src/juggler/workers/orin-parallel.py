import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import datetime
import pytz
import base64
import logging
import pika
import json
import os
import io
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Dict, Optional
from queue import Queue

class MLInferenceService:
    def __init__(self,model_name="squeezenet", host='localhost',device_type='jetson_orin', queue_prefix='edge_device',num_channels=3):  # Adjust based on GPU memory

        self.queue_name = f"{queue_prefix}_{device_type.split('_')[-1]}"
        self.temp_dir = f"temp_images_{self.queue_name}"
        os.makedirs(self.temp_dir, exist_ok=True)

        # Setup logging
        logging.basicConfig(
            level=logging.DEBUG,
            filename="gpu_logs_inference_service.txt",
            filemode="a+",
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {self.device}")

        # Model management
        self.models: Dict[str, torch.nn.Module] = {}
        self.model_locks: Dict[str, threading.Lock] = {}
        self.num_channels = num_channels

        # Load ImageNet classes
        self.categories = self._read_classes()

        # Setup preprocessing pipeline
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Connection/channel management
        self.host = host
        self.connections = []
        self.channels = []
        self.executor = ThreadPoolExecutor(max_workers=num_channels)

        # Setup connections and channels
        for _ in range(num_channels):
            self._setup_channel()

    def _setup_channel(self):
        """Setup a single channel with its own connection"""
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=self.host,
                heartbeat=600,
                blocked_connection_timeout=300,
                connection_attempts=3,
                retry_delay=5
            )
        )
        channel = connection.channel()

        # Enable publisher confirms
        channel.confirm_delivery()

        # Set higher prefetch count for GPU processing
        channel.basic_qos(prefetch_count=4)  # Adjust based on GPU memory

        channel.queue_declare(self.queue_name, durable=True)

        self.connections.append(connection)
        self.channels.append(channel)

        return channel

    def _read_classes(self):
        with open("imagenet-classes.txt", "r") as f:
            return [s.strip() for s in f.readlines()]

    def _load_model(self, model_name: str) -> torch.nn.Module:
        """Load model with TensorRT optimization if available"""
        logging.info(f"Loading model: {model_name}")

        if self.device == "cuda":
            try:
                from torch2trt import torch2trt as trt
                has_trt = True
            except ImportError:
                has_trt = False
                logging.warning("torch-tensorrt not installed. Running without TRT optimization.")

        model_map = {
            "squeezenet": models.squeezenet1_1,
            "mobilenetv3": models.mobilenet_v3_small,
            "resnet50": models.resnet50,
            "resnet18": models.resnet18,
            "resnext50": models.resnext50_32x4d
        }

        if model_name not in model_map:
            raise ValueError(f"Unsupported model: {model_name}")

        model = model_map[model_name](weights='IMAGENET1K_V1').to(self.device).eval()

        if self.device == "cuda" and has_trt:
            try:
                x = torch.ones((1, 3, 224, 224)).cuda()
                model = trt(model, [x])
                logging.info("TensorRT optimization successful")
            except Exception as e:
                logging.warning(f"TensorRT optimization failed: {str(e)}")

        return model

    def _get_or_load_model(self, model_name: str) -> torch.nn.Module:
        """Get existing model or load new one"""
        if model_name not in self.models:
            self.model_locks[model_name] = threading.Lock()
            with self.model_locks[model_name]:
                if model_name not in self.models:  # Double-check pattern
                    self.models[model_name] = self._load_model(model_name)
        return self.models[model_name]

    def decode_image(self, image_data: str, image_name: str) -> Image.Image:
        """Convert base64 string back to PIL Image"""
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            temp_path = os.path.join(self.temp_dir, image_name)
            image.save(temp_path)
            return image
        except Exception as e:
            logging.error(f"Error decoding image: {str(e)}")
            raise

    def preprocess_image(self, image: Image.Image):
        try:
            width, height = image.size
            input_tensor = self.preprocess(image)
            input_batch = input_tensor.unsqueeze(0).to(self.device)
            return input_batch, width, height
        except Exception as e:
            logging.error(f"Error preprocessing image: {str(e)}")
            raise

        
    def predict(self, image_data, image_name, model_name, model):
        try:
            image = self.decode_image(image_data, image_name)
            img_tensor, width, height = self.preprocess_image(image)

            # Inference
            with torch.no_grad():
                start_time = datetime.datetime.now()
                outputs = model(img_tensor)
                inference_time = (datetime.datetime.now() - start_time).total_seconds()

            # Get predictions
            probabilities, indices = outputs.topk(5)
            predictions = [
                {'class': self.categories[idx], 'probability': float(prob)}
                for prob, idx in zip(probabilities[0], indices[0])
            ]

            return {
                'device': self.device_type,
                'model': model_name,
                'predictions': predictions,
                'inference_time': inference_time,
                'image_size': f"{width}x{height}"
            }
        except Exception as e:
            logging.error(f"Prediction error for {image_name}: {str(e)}")
            return {'error': str(e), 'device': self.device_type}

    def process_message(self, ch, method, properties, body):
        """Process a single message"""
        try:
            data = json.loads(body)
            image_name = data.get('image_name')
            image_data = data.get('image_data')
            model_name = data.get('model_name', 'squeezenet')

            logging.info(f"Processing: {image_name} with model {model_name}")

            # Get or load model
            model = self._get_or_load_model(model_name)

            # Process image and run inference
            result = self.predict(image_data, image_name, model_name, model)

            # Send response
            ch.basic_publish(
                exchange='',
                routing_key=properties.reply_to,
                properties=pika.BasicProperties(correlation_id=properties.correlation_id),
                body=json.dumps(result)
            )

            ch.basic_ack(delivery_tag=method.delivery_tag)

        except Exception as e:
            logging.error(f"Error processing message: {str(e)}")
            ch.basic_publish(
                exchange='',
                routing_key=properties.reply_to,
                properties=pika.BasicProperties(correlation_id=properties.correlation_id),
                body=json.dumps({'error': str(e)})
            )
            ch.basic_ack(delivery_tag=method.delivery_tag)

    def start(self):
        """Start multiple consumer threads"""
        logging.info(f"Starting ML Inference Service with {self.num_channels} channels")

        def consumer_thread(channel_id):
            channel = self.channels[channel_id]
            channel.basic_consume(
                queue=self.queue_name,
                on_message_callback=self.process_message
            )
            try:
                channel.start_consuming()
            except Exception as e:
                logging.error(f"Channel {channel_id} error: {str(e)}")

        # Start consumer threads
        threads = []
        for i in range(self.num_channels):
            thread = threading.Thread(target=consumer_thread, args=(i,))
            thread.daemon = True
            thread.start()
            threads.append(thread)

        # Wait for threads
        try:
            for thread in threads:
                thread.join()
        except KeyboardInterrupt:
            logging.info("Shutting down...")
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        for channel in self.channels:
            if channel.is_open:
                channel.stop_consuming()

        for connection in self.connections:
            if connection.is_open:
                connection.close()

        self.executor.shutdown()

if __name__ == "__main__":
    # Start service with multiple channels
    service = MLInferenceService(
        host='192.168.50.100',
        num_channels=6  # Adjust based on GPU memory
    )
    service.start()