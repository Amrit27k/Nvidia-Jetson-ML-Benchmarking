# edge_device.py (for Jetson)
import pika
import json
import base64
from PIL import Image
import io
import os
import docker
import tarfile
class EdgeDeviceConsumer:
    def __init__(self, host='localhost', device_type='jetson_orin', queue_prefix='edge_device'):
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=host))
        self.channel = self.connection.channel()

        self.queue_name = f"{queue_prefix}_{device_type.split('_')[-1]}"
        self.channel.queue_declare(queue=self.queue_name, durable=True)

        self.docker_client = docker.from_env()

    def create_tar(self, source_path, filename):
        """Create a tar archive containing the image file"""
        tar_path = f"/tmp/{filename}.tar"
        with tarfile.open(tar_path, "w") as tar:
            tar.add(source_path, arcname=filename)
        return tar_path

    def process_message(self, ch, method, properties, body):
        try:
            message = json.loads(body)
            image_data = base64.b64decode(message['image'])
            image = Image.open(io.BytesIO(image_data))

            temp_path = f"/tmp/{message['filename']}"
            image.save(temp_path)
            tar_path = self.create_tar(temp_path, message['filename'])
            img_path = f"{message['filename']}"
            container_name = "e26c77204c21"
            container = self.docker_client.containers.get(container_name)

            # Step 1: Copy the image to the container
            print(f"Copying {img_path} to {container_name}:/input/{message['filename']}")

            with open(tar_path, 'rb') as f:
                data = f.read()
                container.put_archive('/input', data)
            
            exec_result = container.exec_run(
                f"python3 /app/tensorrt-infer.py --image /input/{img_path} --model {model}"
            )

            print("Script Output:")
            print(exec_result.output.decode())

            ch.basic_ack(delivery_tag=method.delivery_tag)
            print(temp_path)
            os.remove(temp_path)
            os.remove(tar_path)
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
        host='192.168.50.100',
        device_type='jetson_orin'  # or 'jetson_nano'
    )
    try:
        consumer.start_consuming()
    except KeyboardInterrupt:
        consumer.close()