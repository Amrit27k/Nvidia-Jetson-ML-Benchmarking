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

class MLInferenceService:
    def __init__(self, model_name="squeezenet", host='localhost', device_type='jetson_nano', queue_prefix='edge_device'):
        # Create a temporary directory for received images
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
        logging.info(f"Using processor: {self.device} and queue: {self.queue_name}")

        # Load model once during initialization
        self.model = self._load_model(model_name)
        self.model_lock = Lock()  # For thread-safe predictions

        # Load ImageNet classes
        self.categories = self._read_classes()

        # Setup preprocessing pipeline
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # RabbitMQ setup
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=host))
        self.channel = self.connection.channel()
        self.channel.queue_declare(self.queue_name, durable=True)

    def _load_model(self, model_name):
        logging.info(f"Loading model: {model_name} and queue: {self.queue_name}")
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

        return model

    def _read_classes(self):
        with open("imagenet-classes.txt", "r") as f:
            return [s.strip() for s in f.readlines()]

    def decode_image(self, image_data: str, image_name: str) -> Image.Image:
        """Convert base64 string back to PIL Image"""
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))

            # Save temporarily for debugging if needed
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
            logging.error(f"Error preprocessing image {image_path}: {str(e)}")
            raise

    def predict(self, image_data, image_name):
        try:
            image = self.decode_image(image_data, image_name)
            img_tensor, width, height = self.preprocess_image(image)

            # Use lock for thread-safe prediction
            with self.model_lock:
                with torch.no_grad():
                    start_time = datetime.datetime.now()
                    outputs = self.model(img_tensor)
                    inference_time = (datetime.datetime.now() - start_time).total_seconds()

            # Get top-5 predictions
            probabilities, indices = outputs.topk(5)
            predictions = []
            for prob, idx in zip(probabilities[0], indices[0]):
                predictions.append({
                    'class': self.categories[idx],
                    'probability': float(prob)
                })

            return {
                'device': self.queue_name,
                'predictions': predictions,
                'inference_time': inference_time,
                'image_size': f"{width}x{height}"
            }
        except Exception as e:
            logging.error(f"Prediction error for {image_name}: {str(e)}")
            return {'error': str(e), 'device': self.queue_name}

    def callback(self, ch, method, properties, body):
        try:
            data = json.loads(body)
            image_name = data.get('image_name')
            image_data = data.get('image_data')

            logging.info(f"Received prediction request for: {image_name} from queue: {self.queue_name}")
            result = self.predict(image_data, image_name)
            reply_to = str(properties.reply_to) if properties.reply_to else ''
            # Send result back through reply queue
            ch.basic_publish(
                exchange='',
                routing_key=reply_to,
                properties=pika.BasicProperties(correlation_id=properties.correlation_id),
                body=json.dumps(result)
            )
            ch.basic_ack(delivery_tag=method.delivery_tag)

        except Exception as e:
            logging.error(f"Error processing message: {str(e)}")
            reply_to = str(properties.reply_to) if properties.reply_to else ''
            # Send error response
            ch.basic_publish(
                exchange='',
                routing_key=reply_to,
                properties=pika.BasicProperties(correlation_id=properties.correlation_id),
                body=json.dumps({'error': str(e)})
            )
            ch.basic_ack(delivery_tag=method.delivery_tag)

    def start(self):
        logging.info("Starting ML Inference Service from queue: edge_device_nano")
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(
            queue='edge_device_nano',
            on_message_callback=self.callback
        )
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            self.channel.stop_consuming()
            self.connection.close()

if __name__ == "__main__":
    service = MLInferenceService(host='192.168.50.100')
    service.start()