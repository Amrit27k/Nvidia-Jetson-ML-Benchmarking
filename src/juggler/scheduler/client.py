import os
import pika
import uuid
import json
import base64
import logging
from typing import Optional, Dict, Any
import time

class MLInferenceClient:
    VALID_MODELS = ['squeezenet', 'mobilenetv3', 'resnet50', 'resnet18', 'resnext50']
    
    def __init__(self, host='localhost', queue_prefix='edge_device'):
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        self.queues = {
            'jetson_orin': f'{queue_prefix}_orin',
            'jetson_nano': f'{queue_prefix}_nano',
            'raspberry_pi': f'{queue_prefix}_pi'
        }
        # Initialize RabbitMQ connection
        self.connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
        self.channel = self.connection.channel()
        
        # Declare callback queue for receiving responses
        for queue in self.queues.values():
            result = self.channel.queue_declare(queue=queue, durable=True)
        
        # Declare callback queue for receiving responses
        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue
        
        # Setup consumer for the callback queue
        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True
        )
            
        # Dictionary to store correlation IDs and their corresponding responses
        self.responses = {}

    def on_response(self, ch, method, props, body):
        """Handle incoming responses from the ML service"""
        if props.correlation_id in self.responses:
            self.responses[props.correlation_id] = json.loads(body)

    def encode_image(self, image_path: str) -> str:
        """Convert image to base64 string"""
        try:
            with open(image_path, 'rb') as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logging.error(f"Error encoding image {image_path}: {str(e)}")
            raise

    def predict(self, image_path: str, device_type, model_name:str = 'squeezenet', timeout: int = 10) -> Optional[Dict[str, Any]]:
        """
        Send a prediction request for an image and wait for the response
        
        Args:
            image_path: Path to the image file
            timeout: Maximum time to wait for response in seconds
            
        Returns:
            Dictionary containing prediction results or None if timeout occurs
        """
        try:
            if device_type not in self.queues:
                raise ValueError(f"Invalid device type. Must be one of {list(self.queues.keys())}")
            
            if model_name not in self.VALID_MODELS:
                raise ValueError(f"Invalid model. Must be one of {self.VALID_MODELS}")
            # Encode image to base64
            start_processing = time.time()
            image_data = self.encode_image(image_path)
            # Generate unique correlation ID for this request
            correlation_id = str(uuid.uuid4())
            self.responses[correlation_id] = None
            
            # Prepare the request message
            request = {
                'image_name': os.path.basename(image_path),
                'image_data': image_data,
                'model_name': model_name
            }
            print(f"Image processing time: {time.time() - start_processing}")
            # Publish the request
            self.channel.basic_publish(
                exchange='',
                routing_key=self.queues[device_type],
                properties=pika.BasicProperties(
                    reply_to=self.callback_queue,
                    correlation_id=correlation_id,
                ),
                body=json.dumps(request)
            )
            print(f"Image classification time: {time.time() - start_processing}")
            # Wait for the response with timeout
            start_time = time.time()
            while self.responses[correlation_id] is None:
                self.connection.process_data_events()
                if time.time() - start_time > timeout:
                    del self.responses[correlation_id]
                    logging.error(f"Request timed out for image: {image_path}")
                    return None
                time.sleep(0.1)
            
            # Get and clean up the response
            response = self.responses[correlation_id]
            del self.responses[correlation_id]
            
            return response
            
        except Exception as e:
            logging.error(f"Error during prediction request: {str(e)}")
            return None
        
    def close(self):
        """Close the RabbitMQ connection"""
        self.connection.close()

def process_batch(image_paths: list, device_type, model_name: str = 'squeezenet', batch_size: int = 20, num_iterations: int = 100):
    """
    Process a batch of images with the ML service
    
    Args:
        image_paths: List of paths to image files
        batch_size: Number of concurrent requests to process
    """
    client = MLInferenceClient()
    results = []
    total_time = 0
    
    try:
        # Process images in batches
        for _ in range(num_iterations):
            start_iteration = time.time()
            for i in range(0, len(image_paths), batch_size):
                batch = image_paths[i:i + batch_size]
                logging.info(f"Processing batch {i//batch_size + 1}")
                
                for image_path in batch:
                    start_request = time.time()
                    result = client.predict(image_path, device_type, model_name)
                    end_request = time.time()
                    if result:
                        if 'error' in result:
                            logging.error(f"Error processing {image_path}: {result['error']}")
                        else:
                            logging.info(f"Successfully processed {image_path}")
                            # Print top prediction
                            top_pred = result['predictions'][0]
                            print(f"Image: {image_path}")
                            print(f"Model: {model_name}")
                            print(f"Top prediction: {top_pred['class']} ({top_pred['probability']:.2%})")
                            print(f"Inference time: {result['inference_time']:.3f} seconds")
                            print(f"Image size: {result['image_size']}")
                            print("-" * 50)
                            
                            results.append(result)
                    else:
                        logging.error(f"No response received for {image_path}")
                    logging.info(f"Request time for {image_path} : {end_request - start_request}") # Log time for each request
            end_iteration = time.time()  # End time for the current iteration
            iteration_time = end_iteration - start_iteration
            total_time += iteration_time
            logging.info(f"Iteration {_ + 1} took {iteration_time:.2f} seconds")  # Log time for each iteration
                        
    finally:
        client.close()

    avg_time_per_iteration = total_time / num_iterations if num_iterations > 0 else 0
    logging.info(f"Average time per iteration: {avg_time_per_iteration:.2f} seconds")    
    return results, avg_time_per_iteration

if __name__ == "__main__":
    # Example usage
    image_paths = [
        "E:/NCL/NCL-Intern/Jetson_Benchmarking/images/cat_1.jpg",
        "E:/NCL/NCL-Intern/Jetson_Benchmarking/images/cat_2.jpg",
        "E:/NCL/NCL-Intern/Jetson_Benchmarking/images/cat_3.jpg",
        "E:/NCL/NCL-Intern/Jetson_Benchmarking/images/dog_0.jpg",
        "E:/NCL/NCL-Intern/Jetson_Benchmarking/images/dog_1.jpg",
        "E:/NCL/NCL-Intern/Jetson_Benchmarking/images/dog_2.jpg",
    ]
    device = 'raspberry_pi'
    model = 'squeezenet'
    
    results, avg_time = process_batch(image_paths, device_type=device, model_name=model)
    print(f"Successfully processed {len(results)} images with {device} & model {model}")
    print(f"Average time per iteration: {avg_time:.2f} seconds")