import random
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
import time
import logging
from queue import PriorityQueue
import threading
from client_parallel import MLInferenceClient
class DeviceStatus(Enum):
    AVAILABLE = 1
    BUSY = 2
    ERROR = 3

@dataclass
class DeviceInfo:
    name: str
    status: DeviceStatus
    last_response_time: float
    total_processed: int
    failed_requests: int
    queue_length: int
    moving_avg_latency: float
    weight: float = 1.0

class MLScheduler:
    def __init__(self, host='localhost', num_channels_per_device=3):
        self.host = host
        self.num_channels_per_device = num_channels_per_device
        
        # Initialize device clients
        self.clients = {
            'jetson_orin': MLInferenceClient(host=host, num_channels=num_channels_per_device),
            'jetson_nano': MLInferenceClient(host=host, num_channels=num_channels_per_device),
            'raspberry_pi': MLInferenceClient(host=host, num_channels=num_channels_per_device)
        }
        
        # Initialize device status tracking
        self.devices = {
            'jetson_orin': DeviceInfo('jetson_orin', DeviceStatus.AVAILABLE, 0, 0, 0, 0, 0.0, 1.0),
            'jetson_nano': DeviceInfo('jetson_nano', DeviceStatus.BUSY, 0, 0, 0, 0, 0.0, 0.7),
            'raspberry_pi': DeviceInfo('raspberry_pi', DeviceStatus.BUSY, 0, 0, 0, 0, 0.0, 0.4)
        }
        
        # Task queue for each device
        self.task_queues = {
            device: PriorityQueue() for device in self.devices
        }
        
        # Lock for thread-safe operations
        self.lock = threading.Lock()
        
        # Start monitoring threads
        self.stop_monitoring = False
        self.monitor_threads = {}
        for device in self.devices:
            thread = threading.Thread(target=self._monitor_device, args=(device,))
            thread.daemon = True
            thread.start()
            self.monitor_threads[device] = thread

    def _update_device_metrics(self, device_name: str, response_time: float, success: bool):
        with self.lock:
            device = self.devices[device_name]
            device.total_processed += 1
            if not success:
                device.failed_requests += 1
            
            # Update moving average latency
            alpha = 0.1  # Smoothing factor
            device.moving_avg_latency = (alpha * response_time + (1 - alpha) * device.moving_avg_latency)
            
            # Update device weight based on performance
            success_rate = 1 - (device.failed_requests / device.total_processed)
            latency_factor = 1 / (1 + device.moving_avg_latency)
            device.weight = success_rate * latency_factor

    def _monitor_device(self, device_name: str):
        while not self.stop_monitoring:
            try:
                # Check queue length
                queue_length = self.task_queues[device_name].qsize()
                with self.lock:
                    self.devices[device_name].queue_length = queue_length
                
                # Check if device is responsive
                if time.time() - self.devices[device_name].last_response_time > 60:
                    with self.lock:
                        self.devices[device_name].status = DeviceStatus.ERROR
                
                time.sleep(1)  # Check every second
            except Exception as e:
                logging.error(f"Error monitoring device {device_name}: {str(e)}")

    def _select_device(self, model_name: str) -> str:
        """Select best device based on current load and performance"""
        available_devices = []
        with self.lock:
            for device_name, device in self.devices.items():
                if device.status != DeviceStatus.ERROR or device.status != DeviceStatus.BUSY:
                    # Calculate score based on weight, queue length, and latency
                    score = device.weight / (1 + device.queue_length * device.moving_avg_latency)
                    available_devices.append((score, device_name))
        print(f"line 104: {available_devices}")
        if not available_devices:
            raise RuntimeError("No available devices")
        
        # Select device with highest score
        available_devices.sort(reverse=True)
        return available_devices[0][1]

    async def process_images(self, image_paths: List[str], model_name: str = 'squeezenet', 
                           timeout: int = 100) -> List[Optional[Dict]]:
        results = []
        futures = []
        
        for image_path in image_paths:
            # Select best device for this task
            # with self.lock:
            device = self._select_device(model_name)
            # Add task to device queue
            priority = time.time()  # Use timestamp as priority
            self.task_queues[device].put((priority, image_path))
            
            # Process image
            start_time = time.time()
            try:
                result = self.clients[device].predict_single(
                    (image_path, device, model_name, timeout)
                )
                
                # results = process_batch(
                #     image_paths, 
                #     device_type='jetson_orin',
                #     model_name='resnet18',
                #     batch_size=20,
                #     num_iterations=100,
                #     num_channels=3  # Adjust based on your system's capabilities
                # )
                response_time = time.time() - start_time
                self._update_device_metrics(device, response_time, result is not None)
                
                results.append(result)
                
            except Exception as e:
                logging.error(f"Error processing image on {device}: {str(e)}")
                results.append(None)
                self._update_device_metrics(device, time.time() - start_time, False)
            
            # Update device status
            with self.lock:
                self.devices[device].last_response_time = time.time()
        
        return results

    def close(self):
        """Clean up resources"""
        self.stop_monitoring = True
        for thread in self.monitor_threads.values():
            thread.join()
        for client in self.clients.values():
            client.close()

# Example usage
async def main():
    scheduler = MLScheduler()
    image_paths = [
        "E:/NCL/NCL-Intern/Jetson_Benchmarking/images/cat_1.jpg",
        "E:/NCL/NCL-Intern/Jetson_Benchmarking/images/cat_2.jpg",
        "E:/NCL/NCL-Intern/Jetson_Benchmarking/images/cat_3.jpg",
        "E:/NCL/NCL-Intern/Jetson_Benchmarking/images/dog_0.jpg",
        "E:/NCL/NCL-Intern/Jetson_Benchmarking/images/dog_1.jpg",
        "E:/NCL/NCL-Intern/Jetson_Benchmarking/images/dog_2.jpg",
    ]
    try:
        results = await scheduler.process_images(
            image_paths,
            model_name='resnet18',
            timeout=30
        )
        # print(results)
        # Process results
        for result in results:
            if result and 'predictions' in result:
                top_pred = result['predictions'][0]
                print(f"Top prediction: {top_pred['class']} ({top_pred['probability']:.2%})")
            else:
                logging.error("No response received for image")
    finally:
        scheduler.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())