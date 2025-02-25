import random
from dataclasses import dataclass
from typing import List, Dict, Optional, Set
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
class ModelInfo:
    name: str
    memory_requirement: float  # in MB
    compute_requirement: float  # normalized score 0-1
    supported_devices: Set[str]
    avg_inference_time: Dict[str, float]  # device_name -> time in ms

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
    available_memory: float = 0.0  # in MB
    compute_capacity: float = 1.0  # normalized score 0-1
    supported_models: Set[str] = None

    def __post_init__(self):
        if self.supported_models is None:
            self.supported_models = set()

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
        
        # Initialize device status tracking with extended information
        self.devices = {
            'jetson_orin': DeviceInfo(
                'jetson_orin', DeviceStatus.ERROR, time.time(), 0, 0, 0, 0.0, 1.0,
                available_memory=32768,  # 32GB
                compute_capacity=1.0,
                supported_models={'resnet18', 'resnet50', 'squeezenet', 'mobilenet', 'resnext50'}
            ),
            'jetson_nano': DeviceInfo(
                'jetson_nano', DeviceStatus.AVAILABLE, time.time(), 0, 0, 0, 0.0, 0.7,
                available_memory=4096,   # 4GB
                compute_capacity=0.6,
                supported_models={'squeezenet', 'mobilenet', 'resnet18'}
            ),
            'raspberry_pi': DeviceInfo(
                'raspberry_pi', DeviceStatus.AVAILABLE, time.time(), 0, 0, 0, 0.0, 0.4,
                available_memory=2048,   # 2GB
                compute_capacity=0.3,
                supported_models={'squeezenet', 'mobilenet', 'resnet18', 'resnext50', 'resnet50'}
            )
        }
        
        # Initialize model information with more lenient requirements
        self.models = {
            'resnet50': ModelInfo(
                name='resnet50',
                memory_requirement=90.0,
                compute_requirement=0.7,
                supported_devices={'jetson_orin', 'raspberry_pi'},
                avg_inference_time={'jetson_orin': 130.0, 'raspberry_pi': 30.0}
            ),
            'resnet18': ModelInfo(
                name='resnet18',
                memory_requirement=40.0,
                compute_requirement=0.4,
                supported_devices={'jetson_orin', 'jetson_nano', 'raspberry_pi'},
                avg_inference_time={'jetson_orin': 300.0, 'jetson_nano': 50.0, 'raspberry_pi': 280.0}
            ),
            'squeezenet': ModelInfo(
                name='squeezenet',
                memory_requirement=5.0,
                compute_requirement=0.2,
                supported_devices={'jetson_orin', 'jetson_nano', 'raspberry_pi'},
                avg_inference_time={'jetson_orin': 800.0, 'jetson_nano': 130.0, 'raspberry_pi': 620.0}
            ),
            'mobilenet': ModelInfo(
                name='mobilenet',
                memory_requirement=15.0,
                compute_requirement=0.3,
                supported_devices={'jetson_orin', 'jetson_nano', 'raspberry_pi'},
                avg_inference_time={'jetson_orin': 850.0, 'jetson_nano': 150.0, 'raspberry_pi': 100.0}
            ),
            'resnext50': ModelInfo(
                name='resnext50',
                memory_requirement=98.7,
                compute_requirement=0.8,
                supported_devices={'jetson_orin'},
                avg_inference_time={'jetson_orin': 80.0}
            )
        }
        
        # Task queues for each device
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

    def _update_device_metrics(self, device_name: str, response_time: float, success: bool, model_name: str):
        with self.lock:
            device = self.devices[device_name]
            device.total_processed += 1
            if not success:
                device.failed_requests += 1
            
            # Update moving average latency
            alpha = 0.1  # Smoothing factor
            device.moving_avg_latency = (alpha * response_time + (1 - alpha) * device.moving_avg_latency)
            
            # Update model's average inference time
            model = self.models[model_name]
            current_avg = model.avg_inference_time.get(device_name, response_time)
            model.avg_inference_time[device_name] = alpha * response_time + (1 - alpha) * current_avg
            
            # Update device weight based on performance and current load
            success_rate = 1 - (device.failed_requests / max(1, device.total_processed))
            latency_factor = 1 / (1 + device.moving_avg_latency)
            memory_factor = device.available_memory / (device.available_memory + self.models[model_name].memory_requirement)
            device.weight = success_rate * latency_factor * memory_factor * device.compute_capacity

    def _select_device(self, model_name: str) -> str:
        """Select best device based on model requirements and device capabilities"""
        model = self.models[model_name]
        available_devices = []
        
        with self.lock:
            for device_name, device in self.devices.items():
                # Only check if device supports the model and is not in ERROR state
                if device.status != DeviceStatus.ERROR and model_name in device.supported_models:
                    # Calculate score based on available resources and performance
                    expected_latency = model.avg_inference_time.get(device_name, 100.0)  # default to 100ms if unknown
                    queue_impact = 1 + (device.queue_length * 0.1)  # Reduce queue impact
                    memory_availability = max(0.1, device.available_memory / max(1, model.memory_requirement))
                    compute_match = device.compute_capacity / max(0.1, model.compute_requirement)
                    
                    score = (device.weight * memory_availability * compute_match) / queue_impact
                    available_devices.append((score, device_name))
                    logging.debug(f"Device {device_name} score for {model_name}: {score}")
        
        if not available_devices:
            # Fallback to any device that supports the model
            for device_name, device in self.devices.items():
                if model_name in device.supported_models and device.status != DeviceStatus.ERROR:
                    logging.warning(f"Using fallback device {device_name} for model {model_name}")
                    return device_name
            raise RuntimeError(f"No device supports model {model_name}")
        
        # Select device with highest score
        available_devices.sort(reverse=True)
        selected_device = available_devices[0][1]
        logging.info(f"Selected device {selected_device} for model {model_name}")
        return selected_device

    def _monitor_device(self, device_name: str):
        while not self.stop_monitoring:
            try:
                # Check queue length
                queue_length = self.task_queues[device_name].qsize()
                with self.lock:
                    self.devices[device_name].queue_length = queue_length
                    
                    # Calculate base memory for device
                    base_memory = {
                        'jetson_orin': 32768,
                        'jetson_nano': 4096,
                        'raspberry_pi': 2048
                    }[device_name]
                    
                    # Simulate memory usage more leniently
                    total_memory_required = sum(
                        self.models[task[2]].memory_requirement * 0.8  # Reduce memory impact
                        for task in list(self.task_queues[device_name].queue)
                    )
                    
                    device = self.devices[device_name]
                    device.available_memory = max(base_memory * 0.1,  # Ensure some memory is always available
                                               base_memory - total_memory_required)
                
                # Check if device is responsive - increase timeout to 300 seconds
                if time.time() - self.devices[device_name].last_response_time > 300:
                    with self.lock:
                        self.devices[device_name].status = DeviceStatus.ERROR
                        logging.warning(f"Device {device_name} marked as ERROR due to timeout")
                
                time.sleep(1)  # Check every second
            except Exception as e:
                logging.error(f"Error monitoring device {device_name}: {str(e)}")

    async def process_images(self, image_paths: List[str], model_name: str = 'squeezenet', 
                           timeout: int = 100) -> List[Optional[Dict]]:
        if model_name not in self.models:
            raise ValueError(f"Unsupported model: {model_name}")
            
        results = []
        futures = []
        
        for image_path in image_paths:
            try:
                # Select best device for this task based on model requirements
                device = self._select_device(model_name)
                
                # Add task to device queue with model information
                priority = time.time()  # Use timestamp as priority
                self.task_queues[device].put((priority, image_path, model_name))
                
                # Process image
                start_time = time.time()
                try:
                    result = self.clients[device].predict_single(
                        (image_path, device, model_name, timeout)
                    )
                    
                    response_time = time.time() - start_time
                    self._update_device_metrics(device, response_time, result is not None, model_name)
                    
                    results.append(result)
                    
                except Exception as e:
                    logging.error(f"Error processing image on {device}: {str(e)}")
                    results.append(None)
                    self._update_device_metrics(device, time.time() - start_time, False, model_name)
                
                # Update device status
                with self.lock:
                    self.devices[device].last_response_time = time.time()
                    
            except Exception as e:
                logging.error(f"Error in process_images: {str(e)}")
                results.append(None)
        
        return results

    def close(self):
        """Clean up resources"""
        self.stop_monitoring = True
        for thread in self.monitor_threads.values():
            thread.join()
        for client in self.clients.values():
            client.close()


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
            model_name='resnet50',
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