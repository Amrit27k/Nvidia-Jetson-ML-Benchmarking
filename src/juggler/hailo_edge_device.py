# edge_device.py (for Jetson)
import pika
import json
import base64
from PIL import Image
import io
import os
import docker

class EdgeDeviceConsumer:
    def __init__(self, host='localhost', device_type='jetson_orin', queue_prefix='edge_device'):
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=host))
        self.channel = self.connection.channel()
        
        self.queue_name = f"{queue_prefix}_{device_type.split('_')[-1]}"
        self.channel.queue_declare(queue=self.queue_name, durable=True)
        
        self.docker_client = docker.from_env()
    
    
    def process_message(self, ch, method, properties, body):
        try:
            message = json.loads(body)
            image_data = base64.b64decode(message['image'])
            image = Image.open(io.BytesIO(image_data))
            
            temp_path = f"/tmp/{message['filename']}"
            image.save(temp_path)
            
            # Add your model inference code here
            # Example:
            # model.predict(temp_path)
            
            ch.basic_ack(delivery_tag=method.delivery_tag)
            os.remove(temp_path)
            
        except Exception as e:
            print(f"Error processing message: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
            
    def start_consuming(self):
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(
            queue=self.queue_name,
            on_message_callback=self.process_message
        )
        print(f" [*] Waiting for messages on {self.queue_name}")
        self.channel.start_consuming()
    
    def close(self):
        self.connection.close()

# Run this on Jetson devices
if __name__ == "__main__":
    consumer = EdgeDeviceConsumer(
        host='your_rabbitmq_server_ip',
        device_type='raspberry_pi'
    )
    try:
        consumer.start_consuming()
    except KeyboardInterrupt:
        consumer.close()