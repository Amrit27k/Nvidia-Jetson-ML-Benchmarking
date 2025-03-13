import os
import pika
import uuid
import json
import base64
import logging
from typing import Optional, Dict, Any, List
import time
import sys
from concurrent.futures import ThreadPoolExecutor
import threading

class MLInferenceClient:
    VALID_MODELS = ['squeezenet', 'mobilenetv3', 'resnet50', 'resnet18', 'resnext50']
    
    def __init__(self, host='localhost', queue_prefix='edge_device', num_channels=5):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        self.queues = {
            'jetson_orin': f'{queue_prefix}_orin',
            'jetson_nano': f'{queue_prefix}_nano',
            'raspberry_pi': f'{queue_prefix}_pi'
        }
        
        self.host = host
        self.connection = None
        self.channel = None
        self.callback_queue = None
        self.response = None
        self.corr_id = None
        self.responses = {}
        self.response_ready = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=num_channels)

    def connect(self):
        """Create and return a new connection to RabbitMQ"""
        params = pika.ConnectionParameters(
            host=self.host,
            heartbeat=600,
            blocked_connection_timeout=300,
            connection_attempts=3,
            retry_delay=5
        )
        return pika.SelectConnection(
            parameters=params,
            on_open_callback=self.on_connection_open,
            on_open_error_callback=self.on_connection_open_error,
            on_close_callback=self.on_connection_closed
        )

    def on_connection_open(self, _unused_connection):
        """Called when connection is established"""
        self.open_channel()

    def on_connection_open_error(self, _unused_connection, err):
        """Called if connection opening fails"""
        logging.error(f"Connection open failed: {err}")
        self._reconnect()

    def on_connection_closed(self, _unused_connection, reason):
        """Called when connection is closed"""
        self.channel = None
        if not self._closing:
            self._connection.ioloop.call_later(5, self._reconnect)

    def _reconnect(self):
        """Reconnect to RabbitMQ"""
        self.connection = self.connect()

    def open_channel(self):
        """Open a new channel"""
        self.connection.channel(on_open_callback=self.on_channel_open)

    def on_channel_open(self, channel):
        """Called when channel is opened"""
        self.channel = channel
        self.channel.add_on_close_callback(self.on_channel_closed)
        self.setup_queue()

    def on_channel_closed(self, channel, reason):
        """Called when channel is closed"""
        logging.warning(f"Channel closed: {reason}")
        self.connection.close()

    def setup_queue(self):
        """Setup the callback queue"""
        self.channel.queue_declare(
            queue='',
            exclusive=True,
            callback=self.on_queue_declared
        )

    def on_queue_declared(self, frame):
        """Called when callback queue is declared"""
        self.callback_queue = frame.method.queue
        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True
        )

    def on_response(self, ch, method, props, body):
        """Handle incoming responses from the ML service"""
        if props.correlation_id in self.responses:
            self.responses[props.correlation_id] = json.loads(body)
            self.response_ready.set()

    def encode_image(self, image_path: str) -> str:
        """Convert image to base64 string with error handling"""
        try:
            with open(image_path, 'rb') as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logging.error(f"Error encoding image {image_path}: {str(e)}")
            raise

    def predict_single(self, args) -> Optional[Dict[str, Any]]:
        """Handle single prediction request"""
        image_path, device_type, model_name, timeout = args
        
        try:
            if device_type not in self.queues:
                raise ValueError(f"Invalid device type. Must be one of {list(self.queues.keys())}")
            
            if model_name not in self.VALID_MODELS:
                raise ValueError(f"Invalid model. Must be one of {self.VALID_MODELS}")
            
            image_data = self.encode_image(image_path)
            correlation_id = str(uuid.uuid4())
            self.responses[correlation_id] = None
            self.response_ready.clear()
            
            request = {
                'image_name': os.path.basename(image_path),
                'image_data': image_data,
                'model_name': model_name
            }
            
            self.channel.basic_publish(
                exchange='',
                routing_key=self.queues[device_type],
                properties=pika.BasicProperties(
                    reply_to=self.callback_queue,
                    correlation_id=correlation_id,
                    delivery_mode=2
                ),
                body=json.dumps(request)
            )
            
            # Wait for response with timeout
            if self.response_ready.wait(timeout=timeout):
                response = self.responses[correlation_id]
                del self.responses[correlation_id]
                return response
            else:
                del self.responses[correlation_id]
                logging.error(f"Request timed out for image: {image_path}")
                return None
            
        except Exception as e:
            logging.error(f"Error during prediction request: {str(e)}")
            return None

    def predict_batch(self, image_paths: List[str], device_type: str, model_name: str = 'squeezenet', 
                     timeout: int = 100) -> List[Optional[Dict[str, Any]]]:
        """Process multiple predictions in parallel"""
        futures = []
        for image_path in image_paths:
            future = self.executor.submit(self.predict_single, (image_path, device_type, model_name, timeout))
            futures.append(future)
        
        results = []
        for future in futures:
            results.append(future.result())
        
        return results

    def run(self):
        """Start the client"""
        self.connection = self.connect()
        self.connection.ioloop.start()

    def stop(self):
        """Stop the client"""
        self.executor.shutdown()
        if self.connection:
            self.connection.close()

def process_batch(image_paths: list, device_type: str, model_name: str = 'squeezenet', 
                 batch_size: int = 20, num_iterations: int = 1000, num_channels: int = 5):
    """Process batches with improved parallel processing"""
    client = MLInferenceClient(num_channels=num_channels)
    
    # Start client in a separate thread
    client_thread = threading.Thread(target=client.run)
    client_thread.daemon = True
    client_thread.start()
    
    # Wait for connection to be established
    time.sleep(2)  # Give time for connection setup
    
    total_inf_time = 0
    total_proc_time = 0
    all_results = []
    
    try:
        for iteration in range(num_iterations):
            start_iteration = time.time()
            
            # Process batch
            inf_start = time.time()
            results = client.predict_batch(image_paths, device_type, model_name)
            inf_end = time.time()
            
            # Process results
            for result in results:
                if result:
                    if 'error' in result:
                        logging.error(f"Error processing image: {result['error']}")
                    else:
                        top_pred = result['predictions'][0]
                        logging.info(f"Top prediction: {top_pred['class']} ({top_pred['probability']:.2%})")
                        all_results.append(result)
                else:
                    logging.error("No response received for image")
            
            iteration_time = time.time() - start_iteration
            total_proc_time += iteration_time
            total_inf_time += inf_end - inf_start
            
            fps = len(image_paths) / (inf_end - inf_start) if (inf_end - inf_start) > 0 else 0
            overall_fps = (iteration + 1) * len(image_paths) / total_inf_time if total_inf_time > 0 else 0
            print(f"Iteration {iteration + 1} took {iteration_time:.2f} seconds. FPS: {fps:.2f}. Overall FPS: {overall_fps:.2f}")
            
    finally:
        client.stop()
    
    return all_results


image_paths = [
    "E:/NCL/NCL-Intern/Jetson_Benchmarking/images/cat_1.jpg",
    "E:/NCL/NCL-Intern/Jetson_Benchmarking/images/cat_2.jpg",
    "E:/NCL/NCL-Intern/Jetson_Benchmarking/images/cat_3.jpg",
    "E:/NCL/NCL-Intern/Jetson_Benchmarking/images/dog_0.jpg",
    "E:/NCL/NCL-Intern/Jetson_Benchmarking/images/dog_1.jpg",
    "E:/NCL/NCL-Intern/Jetson_Benchmarking/images/dog_2.jpg",
]

# Increase num_channels for more parallelism
results = process_batch(
    image_paths, 
    device_type='jetson_orin',
    model_name='squeezenet',
    batch_size=20,
    num_iterations=10,
    num_channels=3  # Adjust based on your system's capabilities
)