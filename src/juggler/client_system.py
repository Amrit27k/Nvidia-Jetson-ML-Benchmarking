# client_system.py
import pika
import json
import base64
from PIL import Image
import io
import os

class ImagePublisher:
    def __init__(self, host='localhost', queue_prefix='edge_device'):
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=host))
        self.channel = self.connection.channel()
        
        self.queues = {
            'jetson_orin': f'{queue_prefix}_orin',
            'jetson_nano': f'{queue_prefix}_nano',
            'raspberry_pi': f'{queue_prefix}_pi'
        }
        
        for queue in self.queues.values():
            self.channel.queue_declare(queue=queue, durable=True)
    
    def encode_image(self, image_path):
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            return base64.b64encode(img_byte_arr).decode('utf-8')
    
    def publish_image(self, image_path, device_type, additional_params=None):
        if device_type not in self.queues:
            raise ValueError(f"Invalid device type. Must be one of {list(self.queues.keys())}")
        
        message = {
            'image': self.encode_image(image_path),
            'filename': os.path.basename(image_path),
            'params': additional_params or {}
        }
        
        self.channel.basic_publish(
            exchange='',
            routing_key=self.queues[device_type],
            body=json.dumps(message),
            properties=pika.BasicProperties(
                delivery_mode=2,
            )
        )
    
    def close(self):
        self.connection.close()

# Example usage on your system
if __name__ == "__main__":
    publisher = ImagePublisher(host='localhost')
    print("Sending image to Jetson Orin...")
    # Send to Jetson Orin
    publisher.publish_image(
        "E:/NCL/NCL-Intern/Jetson_Benchmarking/images/cat_2.jpg",
        "jetson_orin",
        additional_params={"model": "yolov5"}
    )
    print("Image sent to Jetson Orin")
    publisher.close()