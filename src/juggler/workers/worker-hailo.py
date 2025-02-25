import os
import sys
import io
import logging
import json
import base64
import datetime
import pika
import numpy as np
import cv2
from PIL import Image
from enum import Enum
from threading import Lock
from hailo_platform import (HEF, ConfigureParams, FormatType, HailoSchedulingAlgorithm, 
                            HailoStreamInterface, InferVStreams, InputVStreamParams, 
                            OutputVStreamParams, VDevice)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class HailoMLInferenceService:
    def __init__(self, 
                 model_name="RESNET50", 
                 host='localhost', 
                 device_type='pi', 
                 queue_prefix='edge_device'):
        # Logging setup
        logging.basicConfig(
            level=logging.DEBUG,
            filename="hailo_inference_logs.txt",
            filemode="a+",
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

        # Temporary directory for received images
        self.queue_name = f"{queue_prefix}_{device_type.split('_')[-1]}"
        self.temp_dir = f"temp_images_{self.queue_name}"
        os.makedirs(self.temp_dir, exist_ok=True)

        # Model and Hailo setup
        self.model_lock = Lock()
        self.model, self.input_shape = self._load_hailo_model(model_name)
        
        # Load ImageNet classes
        self.categories = self._read_classes()

        # RabbitMQ setup
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=host))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=self.queue_name, durable=True)

    def _load_hailo_model(self, model_name):
        model_paths = {
            "RESNET50": os.path.join(BASE_DIR, "models", "resnet_v1_50.hef"),
            "MOBILENETV3": os.path.join(BASE_DIR, "models", "mobilenet_v3.hef"),
            "RESNET18": os.path.join(BASE_DIR, "models", "resnet_v1_18.hef"),
            "RESNEXT": os.path.join(BASE_DIR, "models", "resnext50_32x4d.hef"),
            "SQUEEZENET": os.path.join(BASE_DIR, "models", "squeezenet_v1.1.hef")
        }

        try:
            model_path = model_paths[model_name]
        except KeyError:
            logging.error(f"Unsupported model: {model_name}")
            raise ValueError(f"Unsupported model: {model_name}")

        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        target = VDevice(params=params)

        hef = HEF(model_path)
        configure_params = ConfigureParams.create_from_hef(
            hef=hef, 
            interface=HailoStreamInterface.PCIe
        )
        network_groups = target.configure(hef, configure_params)
        network_group = network_groups[0]
        network_group_params = network_group.create_params()

        input_vstreams_params = InputVStreamParams.make(
            network_group, 
            quantized=False, 
            format_type=FormatType.FLOAT32
        )
        output_vstreams_params = OutputVStreamParams.make(
            network_group, 
            quantized=True, 
            format_type=FormatType.UINT8
        )

        input_vstream_info = hef.get_input_vstream_infos()[0]
        output_vstream_info = hef.get_output_vstream_infos()[0]
        input_shape = (input_vstream_info.shape[1], input_vstream_info.shape[0])

        model_config = (target, network_group, network_group_params, 
                        input_vstreams_params, output_vstreams_params, 
                        input_vstream_info, output_vstream_info)

        return model_config, input_shape

    def _read_classes(self):
        try:
            with open("imagenet-classes.txt", "r") as f:
                return [s.strip() for s in f.readlines()]
        except FileNotFoundError:
            logging.error("ImageNet classes file not found")
            return []

    def decode_image(self, image_data: str, image_name: str) -> Image.Image:
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            temp_path = os.path.join(self.temp_dir, image_name)
            image.save(temp_path)
            return image
        except Exception as e:
            logging.error(f"Image decoding error: {str(e)}")
            raise

    def preprocess_image(self, image: Image.Image):
        try:
            width, height = image.size
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            img_resized = cv2.resize(img_cv, self.input_shape)
            img_normalized = img_resized.astype(np.float32) / 255.0
            img_expanded = np.expand_dims(img_normalized, axis=0)
            return img_expanded, width, height
        except Exception as e:
            logging.error(f"Image preprocessing error: {str(e)}")
            raise

    def predict(self, image_data, image_name):
        try:
            image = self.decode_image(image_data, image_name)
            img_tensor, width, height = self.preprocess_image(image)

            with self.model_lock:
                start_time = datetime.datetime.now()
                (target, network_group, network_group_params, input_vstreams_params, 
                 output_vstreams_params, input_vstream_info, output_vstream_info) = self.model

                input_data = {input_vstream_info.name: img_tensor}
                with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
                    with network_group.activate(network_group_params):
                        infer_results = infer_pipeline.infer(input_data)
                        outputs = infer_results[output_vstream_info.name]

                inference_time = (datetime.datetime.now() - start_time).total_seconds()

            top_indices = np.argsort(outputs[0])[-5:][::-1]
            predictions = [
                {
                    'class': self.categories[idx] if idx < len(self.categories) else 'Unknown',
                    'probability': float(outputs[0][idx])
                }
                for idx in top_indices
            ]

            return {
                'device': self.queue_name,
                'predictions': predictions,
                'inference_time': inference_time,
                'image_size': f"{width}x{height}"
            }
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            return {'error': str(e), 'device': self.queue_name}

    def callback(self, ch, method, properties, body):
        try:
            data = json.loads(body)
            image_name = data.get('image_name')
            image_data = data.get('image_data')
            model_name = data.get('model_name', 'squeezenet')
            logging.info(f"Received prediction request for: {image_name}")
            result = self.predict(image_data, image_name, model_name)
            reply_to = str(properties.reply_to) if properties.reply_to else ''
            
            ch.basic_publish(
                exchange='',
                routing_key=reply_to,
                properties=pika.BasicProperties(correlation_id=properties.correlation_id),
                body=json.dumps(result)
            )
            ch.basic_ack(delivery_tag=method.delivery_tag)

        except Exception as e:
            logging.error(f"Message processing error: {str(e)}")
            reply_to = str(properties.reply_to) if properties.reply_to else ''
            ch.basic_publish(
                exchange='',
                routing_key=reply_to,
                properties=pika.BasicProperties(correlation_id=properties.correlation_id),
                body=json.dumps({'error': str(e)})
            )
            ch.basic_ack(delivery_tag=method.delivery_tag)

    def start(self):
        logging.info("Starting Hailo ML Inference Service")
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(
            queue=self.queue_name,
            on_message_callback=self.callback
        )
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            self.channel.stop_consuming()
            self.connection.close()

if __name__ == "__main__":
    service = HailoMLInferenceService(host='192.168.50.100')
    service.start()