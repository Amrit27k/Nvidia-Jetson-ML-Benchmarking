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
from queue import Queue
from typing import Dict, Optional

class MLInferenceService:
    def __init__(self, model_name="squeezenet", host='localhost', device_type='jetson_orin', queue_prefix='edge_device', num_channels=3):
        self.queue_name = f"{queue_prefix}_{device_type.split('_')[-1]}"
        self.temp_dir = f"temp_images_{self.queue_name}"
        os.makedirs(self.temp_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.DEBUG,
            filename="gpu_logs_inference_service.txt",
            filemode="a+",
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {self.device}")

        self.models: Dict[str, torch.nn.Module] = {}
        self.model_locks: Dict[str, threading.Lock] = {}
        self.num_channels = num_channels

        self.categories = self._read_classes()

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.host = host
        self.connections = []
        self.channels = []
        self.executor = ThreadPoolExecutor(max_workers=num_channels)

        self.streams = [torch.cuda.Stream() for _ in range(num_channels)] if self.device == "cuda" else None

        for _ in range(num_channels):
            self._setup_channel()

    def _setup_channel(self):
        connection = pika.SelectConnection(
            pika.ConnectionParameters(
                host=self.host,
                heartbeat=600,
                blocked_connection_timeout=300,
                connection_attempts=3,
                retry_delay=5
            ),
            on_open_callback=self.on_connection_open
        )
        self.connections.append(connection)

    def on_connection_open(self, connection):
        channel = connection.channel(on_open_callback=lambda ch: self.on_channel_open(ch, connection))

    def on_channel_open(self, channel, connection):
        channel.confirm_delivery()
        channel.basic_qos(prefetch_count=4)
        channel.queue_declare(self.queue_name, durable=False)

        channel.basic_consume(
            queue=self.queue_name,
            on_message_callback=self.process_message,
            auto_ack=True
        )

        self.channels.append(channel)

    def _read_classes(self):
        with open("imagenet-classes.txt", "r") as f:
            return [s.strip() for s in f.readlines()]

    def _load_model(self, model_name: str) -> torch.nn.Module:
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
                x = x.detach()
                model = trt(model, [x])
                logging.info("TensorRT optimization successful")
            except Exception as e:
                logging.warning(f"TensorRT optimization failed: {str(e)}")

        return model

    def _get_or_load_model(self, model_name: str) -> torch.nn.Module:
        if model_name not in self.models:
            self.model_locks[model_name] = threading.Lock()
            with self.model_locks[model_name]:
                if model_name not in self.models:
                    self.models[model_name] = self._load_model(model_name)
        return self.models[model_name]

    def decode_image(self, image_data: str, image_name: str) -> Image.Image:
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

    def infer(self, model, img_tensor, stream):
        with torch.cuda.stream(stream):
            with torch.no_grad():
                output = model(img_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
        return probabilities.cpu().detach().numpy()

    def predict(self, image_data, image_name, model_name):
        try:
            image = self.decode_image(image_data, image_name)
            img_tensor, width, height = self.preprocess_image(image)

            thread_id = threading.get_ident() % self.num_channels
            stream = self.streams[thread_id] if self.device == "cuda" else None

            model = self._get_or_load_model(model_name)
            probabilities = self.infer(model, img_tensor, stream)

            probabilities, indices = torch.tensor(probabilities).topk(5)
            predictions = [
                {'class': self.categories[idx], 'probability': float(prob)}
                for prob, idx in zip(probabilities, indices)
            ]

            return {
                'device': self.device,
                'model': model_name,
                'predictions': predictions,
                'image_size': f"{width}x{height}"
            }
        except Exception as e:
            logging.error(f"Prediction error for {image_name}: {str(e)}")
            return {'error': str(e), 'device': self.device}

    def process_message(self, ch, method, properties, body):
        try:
            data = json.loads(body)
            image_name = data.get('image_name')
            image_data = data.get('image_data')
            model_name = data.get('model_name', 'squeezenet')

            logging.info(f"Processing: {image_name} with model {model_name}")

            future = self.executor.submit(self.predict, image_data, image_name, model_name)
            result = future.result()

            ch.basic_publish(
                exchange='',
                routing_key=properties.reply_to,
                properties=pika.BasicProperties(correlation_id=properties.correlation_id),
                body=json.dumps(result)
            )

        except Exception as e:
            logging.error(f"Error processing message: {str(e)}")
            ch.basic_publish(
                exchange='',
                routing_key=properties.reply_to,
                properties=pika.BasicProperties(correlation_id=properties.correlation_id),
                body=json.dumps({'error': str(e)})
            )

    def start(self):
        logging.info(f"Starting ML Inference Service with {self.num_channels} channels")

        def consumer_thread(connection_id):  # Use connection_id
            connection = self.connections[connection_id]
            try:
                connection.ioloop.start()  # Start ioloop for each connection
            except Exception as e:
                logging.error(f"Connection {connection_id} error: {str(e)}")

        threads = [threading.Thread(target=consumer_thread, args=(i,), daemon=True) for i in range(self.num_channels)]
        for thread in threads:
            thread.start()

        try:
            for thread in threads:
                thread.join()
        except KeyboardInterrupt:
            logging.info("Shutting down...")
            self.cleanup()

    def cleanup(self):
        for connection in self.connections:
            for channel in self.channels:  # Stop consuming first
                if channel.is_open:
                    channel.stop_consuming()
            if connection.is_open:
                connection.ioloop.stop()  # Stop the ioloop
                # if connection.is_open:
                #     connection.close()  # Close the connection
        self.executor.shutdown()


if __name__ == "__main__":
    service = MLInferenceService(host='192.168.50.100', num_channels=6)
    service.start()