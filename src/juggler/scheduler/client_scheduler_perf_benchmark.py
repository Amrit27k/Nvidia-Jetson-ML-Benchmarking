import random
from dataclasses import dataclass
from typing import List, Dict, Optional, Set
from enum import Enum
import time
import logging
from queue import PriorityQueue
import threading
from client_parallel import MLInferenceClient
import asyncio
from concurrent.futures import ThreadPoolExecutor
import statistics
import os

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
# Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     filename="scheduling_logs_inference_benchmark_18_02_1100.txt",
#     filemode="a+",
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )

class DeviceStatus(Enum):
    AVAILABLE = 1
    BUSY = 2
    ERROR = 3

class SchedulingDecision:
    def __init__(self, timestamp: float, model: str, selected_device: str, 
                 considered_devices: List[Dict], reason: str):
        self.timestamp = timestamp
        self.model = model
        self.selected_device = selected_device
        self.considered_devices = considered_devices  # List of {device: score} pairs
        self.reason = reason
        
    def __str__(self):
        devices_info = [f"{d['device']}: {d['score']:.3f}" for d in self.considered_devices]
        return (f"Time: {time.strftime('%H:%M:%S', time.localtime(self.timestamp))} | "
                f"Model: {self.model} | Selected: {self.selected_device} | "
                f"Scores: {', '.join(devices_info)} | Reason: {self.reason}")
    

@dataclass
class ModelInfo:
    name: str
    memory_requirement: float  # in MB
    compute_requirement: float  # normalized score 0-1
    supported_devices: Set[str]
    avg_inference_time: Dict[str, float]  # device_name -> time in ms

@dataclass
class PerformanceMetrics:
    total_images: int = 0
    total_time: float = 0
    successful_inferences: int = 0
    failed_inferences: int = 0
    latencies: List[float] = None
    throughput: float = 0
    fps: float = 0
    
    def __post_init__(self):
        if self.latencies is None:
            self.latencies = []
    
    def update(self, inference_time: float, success: bool):
        self.total_images += 1
        self.total_time += inference_time
        if success:
            self.successful_inferences += 1
            self.latencies.append(inference_time)
        else:
            self.failed_inferences += 1
    
    def calculate_metrics(self):
        if self.total_time > 0:
            self.throughput = self.successful_inferences / self.total_time
            self.fps = self.throughput
        
    def get_statistics(self) -> Dict:
        if not self.latencies:
            return {
                "avg_latency": 0,
                "min_latency": 0,
                "max_latency": 0,
                "p95_latency": 0,
                "p99_latency": 0
            }
        
        sorted_latencies = sorted(self.latencies)
        p95_idx = int(len(sorted_latencies) * 0.95)
        p99_idx = int(len(sorted_latencies) * 0.99)
        
        return {
            "avg_latency": statistics.mean(self.latencies),
            "min_latency": min(self.latencies),
            "max_latency": max(self.latencies),
            "p95_latency": sorted_latencies[p95_idx],
            "p99_latency": sorted_latencies[p99_idx]
        }

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
    available_memory: float = 0.0
    compute_capacity: float = 1.0
    supported_models: Set[str] = None
    performance_metrics: Optional[PerformanceMetrics] = None
    last_model: str = None
    scheduling_history: List[SchedulingDecision] = None
    
    def __post_init__(self):
        if self.supported_models is None:
            self.supported_models = set()
        if self.performance_metrics is None:
            self.performance_metrics = PerformanceMetrics()
        if self.scheduling_history is None:
            self.scheduling_history = []

class MLScheduler:
    # ... (previous initialization code remains the same until process_images) ...
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
                'jetson_orin', DeviceStatus.AVAILABLE, time.time(), 0, 0, 0, 0.0, 1.0,
                available_memory=32768,  # 32GB
                compute_capacity=1.0,
                supported_models={'resnet18', 'resnet50', 'squeezenet', 'mobilenet', 'resnext50'}
            ),
            'jetson_nano': DeviceInfo(
                'jetson_nano', DeviceStatus.ERROR, time.time(), 0, 0, 0, 0.0, 0.4,
                available_memory=4096,   # 4GB
                compute_capacity=0.4,
                supported_models={'squeezenet', 'mobilenet', 'resnet18'}
            ),
            'raspberry_pi': DeviceInfo(
                'raspberry_pi', DeviceStatus.AVAILABLE, time.time(), 0, 0, 0, 0.0, 0.7,
                available_memory=6144,   # 6GB
                compute_capacity=0.6,
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

        # Add scheduling history
        self.scheduling_history = []
        self.decision_log_file = f"scheduling_decisions_{int(time.time())}.log"
        
        # Initialize logging for scheduling decisions
        self.scheduler_logger = logging.getLogger('scheduler_decisions')
        file_handler = logging.FileHandler(self.decision_log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.scheduler_logger.addHandler(file_handler)
        self.scheduler_logger.setLevel(logging.INFO)

    
    def _get_device_change_reason(self, device: DeviceInfo, model_name: str) -> str:
        """Determine the reason for device change"""
        reasons = []
        
        if device.last_model != model_name:
            reasons.append(f"New model request ({model_name})")
        
        if device.status == DeviceStatus.BUSY:
            reasons.append("Device busy")
        
        if device.queue_length > 5:  # Arbitrary threshold
            reasons.append(f"High queue length ({device.queue_length})")
        
        if device.moving_avg_latency > 1.0:  # Arbitrary threshold in seconds
            reasons.append(f"High latency ({device.moving_avg_latency:.2f}s)")
        
        if not reasons:
            reasons.append("Load balancing")
            
        return " | ".join(reasons)

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
        device_scores = []
        
        with self.lock:
            for device_name, device in self.devices.items():
                if device.status != DeviceStatus.ERROR and model_name in device.supported_models:
                    # Calculate score based on available resources and performance
                    expected_latency = model.avg_inference_time.get(device_name, 100.0)
                    queue_impact = 1 + (device.queue_length * 0.1)
                    memory_availability = max(0.1, device.available_memory / max(1, model.memory_requirement))
                    compute_match = device.compute_capacity / max(0.1, model.compute_requirement)
                    
                    score = (device.weight * memory_availability * compute_match) / queue_impact
                    device_scores.append({
                        'device': device_name,
                        'score': score,
                        'queue_length': device.queue_length,
                        'memory': memory_availability,
                        'compute': compute_match
                    })
        
        if not device_scores:
            # Fallback to any device that supports the model
            for device_name, device in self.devices.items():
                if model_name in device.supported_models:
                    reason = "Emergency fallback - no optimal devices available"
                    self._log_scheduling_decision(model_name, device_name, 
                                               [{'device': device_name, 'score': 0.0}], 
                                               reason)
                    return device_name
            raise RuntimeError(f"No device supports model {model_name}")
        
        # Sort devices by score
        device_scores.sort(key=lambda x: x['score'], reverse=True)
        selected_device = device_scores[0]['device']
        
        # Get reason for device selection
        reason = self._get_device_change_reason(self.devices[selected_device], model_name)
        
        # Log the decision
        self._log_scheduling_decision(model_name, selected_device, device_scores, reason)
        
        # Update device's last model
        self.devices[selected_device].last_model = model_name
        
        return selected_device

    def _log_scheduling_decision(self, model_name: str, selected_device: str, 
                               device_scores: List[Dict], reason: str):
        """Log scheduling decision with detailed information"""
        decision = SchedulingDecision(
            timestamp=time.time(),
            model=model_name,
            selected_device=selected_device,
            considered_devices=device_scores,
            reason=reason
        )
        
        # Add to global history
        self.scheduling_history.append(decision)
        
        # Add to device-specific history
        self.devices[selected_device].scheduling_history.append(decision)
        
        # Log the decision
        self.scheduler_logger.info(str(decision))
        
        # Print detailed decision info
        detail_msg = (
            f"\nScheduling Decision:"
            f"\n  Model: {model_name}"
            f"\n  Selected Device: {selected_device}"
            f"\n  Reason: {reason}"
            f"\n  Device Scores:"
        )
        for score_info in device_scores:
            detail_msg += (f"\n    {score_info['device']}: {score_info['score']:.3f} "
                         f"(Queue: {score_info['queue_length']}, "
                         f"Memory: {score_info['memory']:.2f}, "
                         f"Compute: {score_info['compute']:.2f})")
        # logging.info(detail_msg)

    def get_scheduling_report(self) -> Dict:
        """Generate a report of scheduling decisions"""
        report = {
            "total_decisions": len(self.scheduling_history),
            "per_device": {},
            "per_model": {},
            "device_changes": []
        }
        
        prev_device = None
        for decision in self.scheduling_history:
            # Track per-device statistics
            if decision.selected_device not in report["per_device"]:
                report["per_device"][decision.selected_device] = {
                    "total_selections": 0,
                    "models_run": set(),
                    "reasons": []
                }
            report["per_device"][decision.selected_device]["total_selections"] += 1
            report["per_device"][decision.selected_device]["models_run"].add(decision.model)
            report["per_device"][decision.selected_device]["reasons"].append(decision.reason)
            
            # Track per-model statistics
            if decision.model not in report["per_model"]:
                report["per_model"][decision.model] = {
                    "devices_used": set(),
                    "total_runs": 0
                }
            report["per_model"][decision.model]["devices_used"].add(decision.selected_device)
            report["per_model"][decision.model]["total_runs"] += 1
            
            # Track device changes
            if prev_device and prev_device != decision.selected_device:
                report["device_changes"].append({
                    "time": decision.timestamp,
                    "from": prev_device,
                    "to": decision.selected_device,
                    "reason": decision.reason
                })
            prev_device = decision.selected_device
        
        return report

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
                           timeout: int = 100, batch_size: int = 1) -> List[Optional[Dict]]:
        if model_name not in self.models:
            raise ValueError(f"Unsupported model: {model_name}")
            
        results = []
        futures = []
        
        # Process images in batches
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]
            batch_futures = []
            
            for image_path in batch:
                try:
                    device = self._select_device(model_name)
                    priority = time.time()
                    self.task_queues[device].put((priority, image_path, model_name))
                    
                    start_time = time.time()
                    try:
                        result = self.clients[device].predict_single(
                            (image_path, device, model_name, timeout)
                        )
                        
                        inference_time = time.time() - start_time
                        success = result is not None
                        
                        self._update_device_metrics(device, inference_time, success, model_name)
                        self.devices[device].performance_metrics.update(inference_time, success)
                        
                        results.append(result)
                        
                    except Exception as e:
                        logging.error(f"Error processing image on {device}: {str(e)}")
                        results.append(None)
                        self._update_device_metrics(device, time.time() - start_time, False, model_name)
                        self.devices[device].performance_metrics.update(time.time() - start_time, False)
                    
                    with self.lock:
                        self.devices[device].last_response_time = time.time()
                        
                except Exception as e:
                    logging.error(f"Error in process_images: {str(e)}")
                    results.append(None)
            
            # Optional: Add a small delay between batches to prevent overwhelming
            await asyncio.sleep(0.001)     
        
        return results

    def get_performance_report(self) -> Dict:
        """Generate a comprehensive performance report"""
        report = {
            "overall": PerformanceMetrics(),
            "devices": {}
        }
        
        for device_name, device in self.devices.items():
            metrics = device.performance_metrics
            metrics.calculate_metrics()
            stats = metrics.get_statistics()
            
            report["devices"][device_name] = {
                "total_processed": metrics.total_images,
                "successful_inferences": metrics.successful_inferences,
                "failed_inferences": metrics.failed_inferences,
                "fps": metrics.fps,
                "throughput": metrics.throughput,
                **stats
            }
            
            # Update overall metrics
            report["overall"].total_images += metrics.total_images
            report["overall"].successful_inferences += metrics.successful_inferences
            report["overall"].failed_inferences += metrics.failed_inferences
            report["overall"].total_time = max(report["overall"].total_time, metrics.total_time)
            report["overall"].latencies.extend(metrics.latencies)
        
        report["overall"].calculate_metrics()
        overall_stats = report["overall"].get_statistics()
        report["overall"] = {
            "total_processed": report["overall"].total_images,
            "successful_inferences": report["overall"].successful_inferences,
            "failed_inferences": report["overall"].failed_inferences,
            "fps": report["overall"].fps,
            "throughput": report["overall"].throughput,
            **overall_stats
        }
        
        return report
    
    def close(self):
        """Clean up resources"""
        self.stop_monitoring = True
        for thread in self.monitor_threads.values():
            thread.join()
        for client in self.clients.values():
            client.close()

# Example usage with performance testing
async def main():
    scheduler = MLScheduler()
    
    # Test parameters
    num_images = 1000  # Number of images to process
    batch_size = 6     # Process images in batches
    duration = 60      # Test duration in seconds
    model_name = 'resnet18'
    
    # Create a list of test images (repeatedly using the same images)
    base_images = [
        "E:/NCL/NCL-Intern/Jetson_Benchmarking/images/cat_1.jpg",
        "E:/NCL/NCL-Intern/Jetson_Benchmarking/images/cat_2.jpg",
        "E:/NCL/NCL-Intern/Jetson_Benchmarking/images/cat_3.jpg",
        "E:/NCL/NCL-Intern/Jetson_Benchmarking/images/dog_0.jpg",
        "E:/NCL/NCL-Intern/Jetson_Benchmarking/images/dog_1.jpg",
        "E:/NCL/NCL-Intern/Jetson_Benchmarking/images/dog_2.jpg",
    ]
    
    # Repeat images to reach desired count
    test_images = (base_images * (num_images // len(base_images) + 1))[:num_images]
    
    try:
        logging.info(f"Starting performance test with {num_images} images")
        start_time = time.time()
        
        results = await scheduler.process_images(
            test_images,
            model_name=model_name,
            timeout=30,
            batch_size=batch_size
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        print("\n=== Scheduling Decision Report ===")
        # logging.info("\n=== Scheduling Decision Report ===")
        report = scheduler.get_scheduling_report()
        
        print(f"\nTotal Scheduling Decisions: {report['total_decisions']}")
        # logging.info(f"\nTotal Scheduling Decisions: {report['total_decisions']}")
        print("\nPer-Device Statistics:")
        # logging.info("\nPer-Device Statistics:")
        for device, stats in report["per_device"].items():
            print(f"\n{device}:")
            print(f"  Total Selections: {stats['total_selections']}")
            print(f"  Models Run: {', '.join(stats['models_run'])}")
            # logging.info(f"\n{device}:")
            # logging.info(f"  Total Selections: {stats['total_selections']}")
            # logging.info(f"  Models Run: {', '.join(stats['models_run'])}")
            if stats['reasons']:
                common_reasons = set(stats['reasons'])
                # logging.info(f"  Common Reasons: {', '.join(common_reasons)}")
                print(f"  Common Reasons: {', '.join(common_reasons)}")
        
        print("\nPer-Model Statistics:")
        # logging.info("\nPer-Model Statistics:")
        for model, stats in report["per_model"].items():
            print(f"\n{model}:")
            print(f"  Devices Used: {', '.join(stats['devices_used'])}")
            print(f"  Total Runs: {stats['total_runs']}")
            # logging.info(f"\n{model}:")
            # logging.info(f"  Devices Used: {', '.join(stats['devices_used'])}")
            # logging.info(f"  Total Runs: {stats['total_runs']}")
        
        print("\nDevice Changes:")
        # logging.info("\nDevice Changes:")
        for change in report["device_changes"]:
            print(f"\nTime: {time.strftime('%H:%M:%S', time.localtime(change['time']))}")
            print(f"From: {change['from']} -> To: {change['to']}")
            print(f"Reason: {change['reason']}")
            # logging.info(f"\nTime: {time.strftime('%H:%M:%S', time.localtime(change['time']))}")
            # logging.info(f"From: {change['from']} -> To: {change['to']}")
            # logging.info(f"Reason: {change['reason']}")

        
        # Generate and print performance report
        report = scheduler.get_performance_report()
        
        for result in results:
            if result and 'predictions' in result:
                top_pred = result['predictions'][0]
                print(f"Top prediction: {top_pred['class']} ({top_pred['probability']:.2%})")
                # logging.info(f"Top prediction: {top_pred['class']} ({top_pred['probability']:.2%})")
            else:
                logging.error("No response received for image")
                # logging.info("No response received for image")
                
        print("\n=== Performance Test Results ===")
        print(f"Test Duration: {total_time:.2f} seconds")
        print(f"Total Images Processed: {report['overall']['total_processed']}")
        print(f"Successful Inferences: {report['overall']['successful_inferences']}")
        print(f"Failed Inferences: {report['overall']['failed_inferences']}")
        print(f"Overall FPS: {report['overall']['fps']:.2f}")
        print(f"Average Latency: {report['overall']['avg_latency']*1000:.2f} ms")
        print(f"P95 Latency: {report['overall']['p95_latency']*1000:.2f} ms")
        print(f"P99 Latency: {report['overall']['p99_latency']*1000:.2f} ms")
        
        print("\nPer-Device Performance:")

        # logging.info("\n=== Performance Test Results ===")
        # logging.info(f"Test Duration: {total_time:.2f} seconds")    
        # logging.info(f"Total Images Processed: {report['overall']['total_processed']}")
        # logging.info(f"Successful Inferences: {report['overall']['successful_inferences']}")
        # logging.info(f"Failed Inferences: {report['overall']['failed_inferences']}")
        # logging.info(f"Overall FPS: {report['overall']['fps']:.2f}")
        # logging.info(f"Average Latency: {report['overall']['avg_latency']*1000:.2f} ms")
        # logging.info(f"P95 Latency: {report['overall']['p95_latency']*1000:.2f} ms")
        # logging.info(f"P99 Latency: {report['overall']['p99_latency']*1000:.2f} ms")
        # logging.info("\nPer-Device Performance:")
        for device, metrics in report['devices'].items():
            if metrics['total_processed'] == 0:
                continue
            print(f"\n{device}:")
            print(f"  FPS: {metrics['fps']:.2f}")
            print(f"  Average Latency: {metrics['avg_latency']*1000:.2f} ms")
            print(f"  P95 Latency: {metrics['p95_latency']*1000:.2f} ms")
            print(f"  Success Rate: {(metrics['successful_inferences']/metrics['total_processed']*100):.2f}%")
            # logging.info(f"\n{device}:")
            # logging.info(f"  FPS: {metrics['fps']:.2f}")
            # logging.info(f"  Average Latency: {metrics['avg_latency']*1000:.2f} ms")
            # logging.info(f"  P95 Latency: {metrics['p95_latency']*1000:.2f} ms")
            # logging.info(f"  Success Rate: {(metrics['successful_inferences']/metrics['total_processed']*100):.2f}%")
        
    finally:
        scheduler.close()

if __name__ == "__main__":
    asyncio.run(main())