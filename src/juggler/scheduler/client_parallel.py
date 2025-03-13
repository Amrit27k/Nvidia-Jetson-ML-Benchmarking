import os
import pika
import uuid
import json
import base64
import logging
from typing import Optional, Dict, Any, List
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

class MLInferenceClient:
    VALID_MODELS = ['squeezenet', 'mobilenetv3', 'resnet50', 'resnet18', 'resnext50']
    
    def __init__(self, host='localhost', queue_prefix='edge_device', num_channels=5):
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.getLogger("pika").setLevel(logging.WARNING)
        self.queues = {
            'jetson_orin': f'{queue_prefix}_orin',
            'jetson_nano': f'{queue_prefix}_nano',
            'raspberry_pi': f'{queue_prefix}_pi'
        }
        
        # Create connection parameters with heartbeat and blocked connection timeout
        self.params = pika.ConnectionParameters(
            host=host,
            heartbeat=600,
            blocked_connection_timeout=300,
            connection_attempts=3,
            retry_delay=5
        )
        
        # Create multiple channels for parallel processing
        self.channels = []
        self.callback_queues = []
        self.connections = []
        
        for _ in range(num_channels):
            connection = pika.BlockingConnection(self.params)
            channel = connection.channel()
            
            # Enable publisher confirms
            channel.confirm_delivery()
            
            # Set QoS prefetch to 1 to ensure even distribution
            channel.basic_qos(prefetch_count=1)
            
            # Declare device queues
            for queue in self.queues.values():
                channel.queue_declare(queue=queue, durable=True)
            
            # Declare callback queue
            result = channel.queue_declare(queue='', exclusive=True)
            callback_queue = result.method.queue
            
            channel.basic_consume(
                queue=callback_queue,
                on_message_callback=self.on_response,
                auto_ack=True
            )
            
            self.channels.append(channel)
            self.callback_queues.append(callback_queue)
            self.connections.append(connection)
        
        self.responses = {}
        self.channel_index = 0
        
        # Create thread pool for parallel image processing
        self.executor = ThreadPoolExecutor(max_workers=num_channels)

    def on_response(self, ch, method, props, body):
        """Handle incoming responses from the ML service"""
        if props.correlation_id in self.responses:
            self.responses[props.correlation_id] = json.loads(body)
        
    def get_next_channel(self):
        """Round-robin channel selection"""
        channel = self.channels[self.channel_index]
        callback_queue = self.callback_queues[self.channel_index]
        self.channel_index = (self.channel_index + 1) % len(self.channels)
        return channel, callback_queue

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
            
            channel, callback_queue = self.get_next_channel()
            image_data = self.encode_image(image_path)
            correlation_id = str(uuid.uuid4())
            self.responses[correlation_id] = None
            
            request = {
                'image_name': os.path.basename(image_path),
                'image_data': image_data,
                'model_name': model_name
            }
            
            # dict_size = sys.getsizeof(request) + sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in request.items()) 
            # print(f"Size of dictionary: {dict_size} bytes")
            # Publish with mandatory flag and publisher confirms
            channel.basic_publish(
                exchange='',
                routing_key=self.queues[device_type],
                properties=pika.BasicProperties(
                    reply_to=callback_queue,
                    correlation_id=correlation_id,
                    delivery_mode=2  # Make message persistent
                ),
                body=json.dumps(request),
                mandatory=True
            )
            
            start_time = time.time()
            while self.responses[correlation_id] is None:
                channel.connection.process_data_events()
                if time.time() - start_time > timeout:
                    del self.responses[correlation_id]
                    logging.error(f"Request timed out for image: {image_path}")
                    return None
                time.sleep(0.01)  # Reduced sleep time
            
            response = self.responses[correlation_id]
            del self.responses[correlation_id]
            return response
            
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
        for future in as_completed(futures):
            results.append(future.result())
        
        return results

    def close(self):
        """Clean up resources"""
        self.executor.shutdown()
        for connection in self.connections:
            connection.close()

def process_batch(image_paths: list, device_type: str, model_name: str = 'squeezenet', 
                 batch_size: int = 20, num_iterations: int = 1000, num_channels: int = 5):
    """Process batches with improved parallel processing"""
    client = MLInferenceClient(num_channels=num_channels)
    total_inf_time = 0
    total_proc_time = 0
    all_results = []
    
    try:
        for iteration in range(num_iterations):
            start_iteration = time.time()
            
            # Process entire batch in parallel
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
            print(f"Iteration {iteration + 1} took {iteration_time:.2f} seconds. Inference time: {inf_end - inf_start:.2f} seconds. FPS (Iteration): {fps:.2f}. Overall FPS: {overall_fps:.2f}")
            
    finally:
        client.close()

    avg_inf_time = total_inf_time / num_iterations if num_iterations > 0 else 0
    avg_proc_time = total_proc_time / num_iterations if num_iterations > 0 else 0
    
    print(f"Average inference time per iteration: {avg_inf_time:.2f} seconds")
    print(f"Average processing time per iteration: {avg_proc_time:.2f} seconds")

    overall_fps = (len(image_paths) * num_iterations) / total_inf_time if total_inf_time > 0 else 0
    print(f"Overall Average FPS: {overall_fps:.2f}")
    return all_results


# image_paths = [
#     "E:/NCL/NCL-Intern/Jetson_Benchmarking/images/cat_1.jpg",
#     "E:/NCL/NCL-Intern/Jetson_Benchmarking/images/cat_2.jpg",
#     "E:/NCL/NCL-Intern/Jetson_Benchmarking/images/cat_3.jpg",
#     "E:/NCL/NCL-Intern/Jetson_Benchmarking/images/dog_0.jpg",
#     "E:/NCL/NCL-Intern/Jetson_Benchmarking/images/dog_1.jpg",
#     "E:/NCL/NCL-Intern/Jetson_Benchmarking/images/dog_2.jpg",
# ]

# # Increase num_channels for more parallelism
# results = process_batch(
#     image_paths, 
#     device_type='jetson_orin',
#     model_name='resnet18',
#     batch_size=20,
#     num_iterations=100,
#     num_channels=3  # Adjust based on your system's capabilities
# )