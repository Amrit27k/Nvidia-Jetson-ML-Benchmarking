import pika
import uuid
import json

RABBITMQ_HOST = 'localhost'
QUEUE_NAME = 'ml_inference'
REPLY_QUEUE = 'ml_responses'

# RabbitMQ connection
connection = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_HOST))
channel = connection.channel()

# Declare reply queue
result = channel.queue_declare(queue='', exclusive=True)
callback_queue = result.method.queue

response = None
corr_id = str(uuid.uuid4())

def on_response(ch, method, properties, body):
    global response
    if corr_id == properties.correlation_id:
        response = json.loads(body)

# Set up response handling
channel.basic_consume(queue=callback_queue, on_message_callback=on_response, auto_ack=True)

# Publish inference request
image_path = 'path/to/image.jpg'
request = {
    'image_path': image_path,
}
channel.basic_publish(
    exchange='',
    routing_key=QUEUE_NAME,
    body=json.dumps(request),
    properties=pika.BasicProperties(
        reply_to=callback_queue,
        correlation_id=corr_id
    )
)

print("Waiting for inference result...")
while response is None:
    connection.process_data_events()

print("Inference Result:", response)