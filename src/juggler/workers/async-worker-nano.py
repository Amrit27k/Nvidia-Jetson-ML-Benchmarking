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
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
import threading

class ParallelMLInferenceService:
    def __init__(self, model_name="squeezenet", host='localhost', device_type='jetson_nano', 
                 queue_prefix='edge_device', num_workers=4):
        # Setup logging
        logging.basicConfig(
            level=logging.DEBUG,
            filename="gpu_logs_inference_service.txt",
            filemode="a+",
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

        self.device_type = device_type
        self.exchange_name = 'ml_inference'
        self.temp_dir = f"temp_images_{device_type}"
        os.makedirs(self.temp_dir, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {self.device}")

        # Thread pool for parallel processing
        self.num_workers = num_workers
        self.thread_pool = ThreadPoolExecutor(max_workers=num_workers)
        
        # Model management
        self.models = {}
        self.model_locks = {}
        self.categories = self._read_classes()

        # Setup preprocessing pipeline
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # RabbitMQ setup with direct exchange
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=host))
        self.channel = self.connection.channel()
        
        # Declare direct exchange
        self.channel.exchange_declare(
            exchange=self.exchange_name,
            exchange_type='direct',
            durable=True
        )

        # Declare queue and bind with routing keys
        result = self.channel.queue_declare(queue='', exclusive=True)
        self.queue_name = result.method.queue

        # Bind queue to exchange with routing keys for supported models
        supported_models = ['squeezenet', 'mobilenetv3', 'resnet50', 'resnet18', 'resnext50']
        for model in supported_models:
            routing_key = f"{device_type.split('_')[-1]}.{model}"
            self.channel.queue_bind(
                exchange=self.exchange_name,
                queue=self.queue_name,
                routing_key=routing_key
            )

    def _load_model(self, model_name):
        logging.info(f"Loading model: {model_name}")
        if model_name not in self.models:
            if self.device == "cuda":
                try:
                    from torch2trt import torch2trt as trt
                except ImportError:
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

            model = model_map[model_name](pretrained=True).to(self.device).eval()

            if self.device == "cuda":
                try:
                    model = trt(model, inputs=[torch.ones((1,3,224,224)).cuda()])
                except:
                    logging.warning("TensorRT optimization failed, using standard model")

            self.models[model_name] = model
            self.model_locks[model_name] = Lock()

        return self.models[model_name]

    def _read_classes(self):
        with open("imagenet-classes.txt", "r") as f:
            return [s.strip() for s in f.readlines()]

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

    def predict(self, image_data, image_name, model_name):
        try:
            image = self.decode_image(image_data, image_name)
            img_tensor, width, height = self.preprocess_image(image)

            # Get or load model with lock
            with self.model_locks.get(model_name, Lock()):
                if model_name not in self.models:
                    self._load_model(model_name)
                model = self.models[model_name]

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
        """Process a single message in a worker thread"""
        try:
            data = json.loads(body)
            image_name = data.get('image_name')
            image_data = data.get('image_data')
            model_name = data.get('model_name', 'squeezenet')
            
            logging.info(f"Processing request for: {image_name} with model: {model_name}")
            result = self.predict(image_data, image_name, model_name)
            
            # Send result back
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

    def callback(self, ch, method, properties, body):
        """Submit message processing to thread pool"""
        self.thread_pool.submit(self.process_message, ch, method, properties, body)

    def start(self):
        logging.info(f"Starting Parallel ML Inference Service with {self.num_workers} workers")
        # Set prefetch count to match number of workers
        self.channel.basic_qos(prefetch_count=self.num_workers)
        self.channel.basic_consume(
            queue=self.queue_name,
            on_message_callback=self.callback
        )
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            self.channel.stop_consuming()
            self.connection.close()
            self.thread_pool.shutdown()

if __name__ == "__main__":
    service = ParallelMLInferenceService(
        host='192.168.50.100',
        device_type='jetson_nano',
        num_workers=4  # Adjust based on your GPU capability
    )
    service.start()