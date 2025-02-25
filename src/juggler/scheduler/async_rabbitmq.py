import pika
import json
import time
import logging
from datetime import datetime
import threading
from queue import Queue
import signal
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HighPerformancePublisher:
    def __init__(self, host='localhost', queue_name='jetson_queue'):
        self.host = host
        self.queue_name = queue_name
        self.connection = None
        self.channel = None
        self.should_stop = False
        self.confirm_delivery = False  # Disable confirms for max speed
        self.batch_size = 1000  # Increased batch size
        self.message_batch = []
        self.last_publish_time = time.time()
        
    def connect(self):
        try:
            # Optimize connection parameters
            parameters = pika.ConnectionParameters(
                host=self.host,
                heartbeat=0,  # Disable heartbeat for max performance
                blocked_connection_timeout=300,
                socket_timeout=2,
                # TCP settings
                tcp_options=dict(
                    tcp_keepalive=True,
                    tcp_keepalive_idle=60,
                    tcp_keepalive_interval=30
                ),
                # Connection settings
                connection_attempts=3,
                retry_delay=1,
                # Channel settings
                channel_max=9  # No channel limit
            )
            
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            
            # Configure channel for maximum throughput
            self.channel.queue_declare(
                queue=self.queue_name,
                durable=True,
                arguments={
                    'x-queue-mode': 'lazy',  # Optimize for throughput over latency
                    'x-max-length': 1000000,  # Set queue length limit
                    'x-overflow': 'drop-head'  # Drop oldest messages if queue full
                }
            )
            
            if self.confirm_delivery:
                self.channel.confirm_delivery()
            
            logger.info("Publisher connected to RabbitMQ")
            return True
            
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False

    def publish_batch(self):
        if not self.message_batch:
            return
        
        try:
            # Use basic_publish_batch for better performance
            for message in self.message_batch:
                self.channel.basic_publish(
                    exchange='',
                    routing_key=self.queue_name,
                    body=message,
                    properties=pika.BasicProperties(
                        delivery_mode=1,  # Non-persistent for speed
                        content_type='application/json'
                    )
                )
            
            self.message_batch = []
            self.last_publish_time = time.time()
            
        except Exception as e:
            logger.error(f"Batch publishing error: {e}")
            # Try to reconnect
            self.connect()

    def publish_message(self, data):
        try:
            message = json.dumps({
                'timestamp': datetime.now().isoformat(),
                'data': data
            }).encode()
            
            self.message_batch.append(message)
            
            # Publish batch if size threshold reached or time threshold exceeded
            if len(self.message_batch) >= self.batch_size or \
               (time.time() - self.last_publish_time) > 0.1:  # 100ms max delay
                self.publish_batch()
                
            return True
            
        except Exception as e:
            logger.error(f"Publishing error: {e}")
            return False

    def close(self):
        try:
            # Publish any remaining messages
            if self.message_batch:
                self.publish_batch()
            if self.connection:
                self.connection.close()
                logger.info("Publisher connection closed")
        except Exception as e:
            logger.error(f"Error closing connection: {e}")

def run_publisher():
    publisher = HighPerformancePublisher()
    
    def signal_handler(signum, frame):
        logger.info("Received signal to stop...")
        publisher.should_stop = True
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    if not publisher.connect():
        return
    
    try:
        sequence = 0
        start_time = time.time()
        messages_sent = 0
        
        while not publisher.should_stop:
            data = {
                'sensor_id': 1,
                'value': sequence % 100,
                'sequence': sequence
            }
            
            if publisher.publish_message(data):
                sequence += 1
                messages_sent += 1
                
                # Print statistics every 10000 messages
                if messages_sent % 10000 == 0:
                    elapsed = time.time() - start_time
                    rate = messages_sent / elapsed
                    logger.info(f"Publishing rate: {rate:.2f} messages/second")
            
    except KeyboardInterrupt:
        logger.info("Publisher stopping...")
    finally:
        publisher.close()

run_publisher()