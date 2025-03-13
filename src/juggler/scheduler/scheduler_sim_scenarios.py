from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
from collections import defaultdict
from queue import Queue
import time
from dataclasses import dataclass, field
import logging
# Make sure log-images directory exists
# os.makedirs("log-images", exist_ok=True)
timestamp = str(datetime.now().strftime('%Y-%m-%dT%H-%M'))
# os.makedirs(f"log-images/{timestamp}", exist_ok=True)

# logging.basicConfig(
#     level=logging.INFO,
#     filename=f"log-images/{timestamp}/logs_inference_benchmark.txt",
#     filemode="a+",
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )
# Define data structures
@dataclass
class Model:
    name: str
    max_fps: float  # Maximum FPS this model can achieve on any device

@dataclass
class Device:
    name: str
    
    # Power consumption coefficients (ax³ + bx² + cx + d)
    # where x is the FPS value
    power_coefficients: dict  # Maps model_name -> [a, b, c, d]
    
    # Maximum FPS achievable for each model on this device
    max_fps: dict  # Maps model_name -> max_fps
    
    # Current allocation of FPS for each model
    current_fps: dict  # Maps model_name -> current_fps
    
    # Current power consumption
    current_power: float = 0.0
    
    # Request queue for this device
    request_queue: Queue = field(default_factory=Queue)

IMAGE_ACCURACY = {
    "1.jpg": {
        "ResNet50": 19.451,
        "MobileNetV3": 40.332,
        "ResNet18": 27.209,
        "ResNeXt50": 52.723,
        "SqueezeNet": 36.797
    },
    "cat_0.jpg": {
        "ResNet50": 18.899,
        "MobileNetV3": 14.485,
        "ResNet18": 18.534,
        "ResNeXt50": 18.552,
        "SqueezeNet": 21.240
    },
    "cat_1.jpg": {
        "ResNet50": 64.667,
        "MobileNetV3": 53.589,
        "ResNet18": 64.716,
        "ResNeXt50": 80.243,
        "SqueezeNet": 50.415
    },
    "cat_2.jpg": {
        "ResNet50": 99.542,
        "MobileNetV3": 99.132,
        "ResNet18": 99.256,
        "ResNeXt50": 99.994,
        "SqueezeNet": 99.602
    },
    "dog_0.jpg": {
        "ResNet50": 95.679,
        "MobileNetV3": 69.595,
        "ResNet18": 89.873,
        "ResNeXt50": 98.354,
        "SqueezeNet": 60.378
    },
    "dog_1.jpg": {
        "ResNet50": 88.961,
        "MobileNetV3": 81.686,
        "ResNet18": 63.549,
        "ResNeXt50": 99.476,
        "SqueezeNet": 28.500
    }
}

# Initialize models
models = [
    Model(name="ResNet50", max_fps=175),
    Model(name="MobileNet", max_fps=1000),
    Model(name="ResNet18", max_fps=425),
    Model(name="ResNext50", max_fps=110),
    Model(name="SqueezeNet", max_fps=950)
]

# Initialize devices
devices = [
    Device(
        name="Device1 (Jetson Orin Nano)", 
        power_coefficients={
            "ResNet50": [6.072131e-04, -1.300612e-01, 4.557199e+01, 4285.86], 
            "MobileNet": [3.521553e-06, -8.418713e-03, 5.996142e+00, 4232.51], 
            "ResNet18": [-4.366841e-05, 7.913778e-03, 1.663735e+01, 4308.14], 
            "ResNext50": [-6.402410e-04, 7.870599e-02, 5.164135e+01, 4239.72], 
            "SqueezeNet": [1.073624e-05, -1.839516e-02, 1.024424e+01, 4160.02] 
        },
        max_fps={
            "ResNet50": 174.96,
            "MobileNet": 967.68,
            "ResNet18": 413.15,
            "ResNext50": 108.93,
            "SqueezeNet": 922.49
        },
        current_fps={model.name: 0 for model in models}
    ),
    Device(
        name="Device2 (Jetson Nano)", 
        power_coefficients={
            "ResNet50": [-2.057230e+01, 6.123255e+02, -4.061582e+03, 6333.14],
            "MobileNet": [1.854938e-03, -6.409779e-01, 6.233325e+01, 1911.11],
            "ResNet18": [-1.888558e-02, 9.252537e-01, 1.157932e+02, 2803.937],
            "ResNext50": [-1.988446e+02, 4.591937e+03, -2.792636e+04, 26012.155],
            "SqueezeNet": [1.590855e-03, -5.632611e-01, 5.534153e+01, 2407.379]
        },
        max_fps={
            "ResNet50": 19.19,
            "MobileNet": 201.96,
            "ResNet18": 63.8,
            "ResNext50": 12.3,
            "SqueezeNet": 153.29
        },
        current_fps={model.name: 0 for model in models}
    ),
    Device(
        name="Device3 (Raspberry Pi 4)", 
        power_coefficients={
            "ResNet50": [0, -0.312, 25.71, 4484.83],
            "MobileNet": [0, -0.04, 7.56, 4849.85],
            "ResNet18": [0, 0.0018, 0.15, 5107.62],
            "ResNext50": [0, -0.446, 31.81, 4724.72],
            "SqueezeNet": [0, -0.00071, 0.53, 5069.21]
        },
        max_fps={
            "ResNet50": 44,
            "MobileNet": 138,
            "ResNet18": 318,
            "ResNext50": 45,
            "SqueezeNet": 689
        },
        current_fps={model.name: 0 for model in models}
    )
]


# Function to calculate power consumption based on FPS
def calculate_power(fps, coefficients):
    a, b, c, d = coefficients
    power = a * (fps ** 3) + b * (fps ** 2) + c * fps + d
    return max(0, power)  # Power should never be negative

# Update each device's power consumption
def update_device_power(device):
    power = 0
    for model_name, fps in device.current_fps.items():
        if fps > 0:
            coefficients = device.power_coefficients[model_name]
            power += calculate_power(fps, coefficients)
    device.current_power = power
    return power

# AHP implementation for decision making
class AHPDecisionMaker:
    def __init__(self):
        # Saaty scale for pairwise comparisons
        self.saaty_scale = {
            1: "Equal importance",
            3: "Moderate importance",
            5: "Strong importance",
            7: "Very strong importance",
            9: "Extreme importance"
        }
        # 2, 4, 6, 8 are intermediate values
    
    def calculate_weights(self, matrix):
        """Calculate weights from pairwise comparison matrix using eigenvector method"""
        # Normalize the matrix by column
        normalized_matrix = matrix / matrix.sum(axis=0)
        
        # Calculate weights as the average of normalized values by row
        weights = normalized_matrix.mean(axis=1)
        
        # Normalize weights to sum to 1
        weights = weights / weights.sum()
        
        return weights
    
    def evaluate_alternatives(self, alternatives, criteria, criteria_weights, scoring_funcs):
        """Evaluate alternatives based on criteria and weights"""
        n_alternatives = len(alternatives)
        n_criteria = len(criteria)
        scores = np.zeros((n_alternatives, n_criteria))
        
        # Calculate scores for each alternative on each criterion
        for i, alternative in enumerate(alternatives):
            for j, criterion in enumerate(criteria):
                scores[i, j] = scoring_funcs[j](alternative)
        
        # Normalize scores by column
        for j in range(n_criteria):
            if np.max(scores[:, j]) - np.min(scores[:, j]) > 0:
                scores[:, j] = (scores[:, j] - np.min(scores[:, j])) / (np.max(scores[:, j]) - np.min(scores[:, j]))
        
        # Calculate final scores
        final_scores = np.dot(scores, criteria_weights)
        
        return final_scores

# Queue-based AHP Scheduler
class AHPQueueScheduler:
    def __init__(self, devices, models):
        self.devices = devices
        self.models = models
        self.request_history = []  # To track request allocations
        self.ahp = AHPDecisionMaker()
        self.last_used_device = None
        self.last_used_model = None
        self.device_cooldown = {}  # Track cooldown periods for overloaded devices
        self.model_switch_time = {}  # Track last time a model was switched
        self.device_switch_time = time.time()  # Track last time a device was switched
        self.current_image = None
        # Initialize cooldown tracking for all devices
        for device in self.devices:
            self.device_cooldown[device.name] = {
                "overloaded": False,
                "recovery_until": 0,  # Timestamp when device will be back to normal
                "workload_before_overload": {}  # Store workload before overload
            }
            
        # Initialize model switch tracking
        for model in self.models:
            self.model_switch_time[model.name] = 0
    
    def select_model_using_ahp(self, required_fps, image=None):
        """Use AHP to select the best model based on criteria"""
        # Define criteria for model selection
        criteria = ["performance", "power_efficiency", "availability"]
        
        # Define criteria weights
        criteria_weights = np.array([0.6, 0.3, 0.1])
        
        # Define scoring functions for each criterion
        def score_performance(model):
            # Higher max_fps = better performance
            return model.max_fps
        
        def score_power_efficiency(model):
            # Estimate average power efficiency across devices
            power_efficiencies = []
            for device in self.devices:
                if model.name in device.max_fps and device.max_fps[model.name] > 0:
                    # Calculate power at half of max FPS as a rough estimate
                    half_fps = device.max_fps[model.name] / 2
                    power = calculate_power(half_fps, device.power_coefficients[model.name])
                    if power > 0:
                        efficiency = half_fps / power  # FPS per watt
                        power_efficiencies.append(efficiency)
            
            if power_efficiencies:
                return np.mean(power_efficiencies)
            return 0
        
        def score_availability(model):
            # Higher availability = more available FPS across devices that are not in recovery
            available_fps = 0
            available_devices = 0
            
            for device in self.devices:
                # Skip devices in recovery mode
                if self.device_cooldown[device.name]["overloaded"]:
                    continue
                    
                if model.name in device.max_fps:
                    available_fps += (device.max_fps[model.name] - device.current_fps[model.name])
                    available_devices += 1
            
            if available_devices > 0:
                return available_fps / available_devices
            return 0
        
        scoring_funcs = [score_performance, score_power_efficiency, score_availability]
        
        # Evaluate models
        alternatives = [model for model in self.models]

        if image and image in IMAGE_ACCURACY:
        # Include accuracy in the decision
            self.current_image = image
            scores = self.ahp.evaluate_alternatives_with_accuracy(
            alternatives, criteria, criteria_weights, scoring_funcs, image, accuracy_weight=0.4)
        else:
        # Standard AHP without accuracy
            scores = self.ahp.evaluate_alternatives(
            alternatives, criteria, criteria_weights, scoring_funcs)

        scores = self.ahp.evaluate_alternatives(alternatives, criteria, criteria_weights, scoring_funcs)
        
        # Find models that can satisfy the required FPS on non-overloaded devices
        viable_models = []
        for i, model in enumerate(alternatives):
            for device in self.devices:
                # Skip devices in recovery mode
                if self.device_cooldown[device.name]["overloaded"]:
                    continue
                    
                if (model.name in device.max_fps and 
                    device.max_fps[model.name] >= required_fps and
                    device.current_fps[model.name] + required_fps <= device.max_fps[model.name]):
                    viable_models.append((model, scores[i]))
                    break
        
        if viable_models:
            # Sort by AHP score (descending)
            viable_models.sort(key=lambda x: x[1], reverse=True)
            selected_model = viable_models[0][0]
            
            # Update tracking
            self.last_used_model = selected_model
            self.model_switch_time[selected_model.name] = time.time()
            
            return selected_model
        elif self.last_used_model:
            return self.last_used_model
        else:
            # Fall back to highest score model
            best_idx = np.argmax(scores)
            return alternatives[best_idx]
    
    def select_device_using_ahp(self, model_name, required_fps):
        """Use AHP to select the best device for a given model and FPS"""
        # Define criteria for device selection
        criteria = ["power_efficiency", "load_balance", "performance_headroom"]
        
        # Define criteria weights
        criteria_weights = np.array([0.5, 0.3, 0.2])
        
        # Get viable devices (those that can handle the request and are not in recovery)
        viable_devices = []
        for device in self.devices:
            # Skip devices in recovery mode
            if self.device_cooldown[device.name]["overloaded"]:
                continue
                
            if (model_name in device.max_fps and 
                device.max_fps[model_name] >= required_fps and
                device.current_fps[model_name] + required_fps <= device.max_fps[model_name]):
                viable_devices.append(device)
        
        # Define scoring functions for each criterion
        def score_power_efficiency(device):
            """Lower power increase = better efficiency"""
            # Calculate current power
            current_power = update_device_power(device)
            
            # Calculate power with additional FPS
            original_fps = device.current_fps[model_name]
            device.current_fps[model_name] += required_fps
            new_power = update_device_power(device)
            
            # Reset to original state
            device.current_fps[model_name] = original_fps
            update_device_power(device)
            
            # Calculate power increase (lower is better)
            power_increase = new_power - current_power
            
            # Return inverse since lower power increase is better
            if power_increase > 0:
                return 1 / power_increase
            return 1000  # Very high score for no power increase
        
        def score_load_balance(device):
            """More balanced load across models = better"""
            # Calculate current load percentage
            total_fps_capacity = sum(device.max_fps.values())
            current_fps_usage = sum(device.current_fps.values())
            
            if total_fps_capacity > 0:
                return 1 - (current_fps_usage / total_fps_capacity)  # Higher score for less loaded devices
            return 0
        
        def score_performance_headroom(device):
            """More headroom for the specific model = better"""
            max_fps = device.max_fps[model_name]
            current_fps = device.current_fps[model_name]
            
            if max_fps > 0:
                headroom = (max_fps - current_fps) / max_fps
                return headroom
            return 0
        
        scoring_funcs = [score_power_efficiency, score_load_balance, score_performance_headroom]
        
        if viable_devices:
            # Evaluate devices
            scores = self.ahp.evaluate_alternatives(viable_devices, criteria, criteria_weights, scoring_funcs)
            
            # Find best device
            best_idx = np.argmax(scores)
            best_device = viable_devices[best_idx]
            
            # Calculate power increase for the best device
            current_power = update_device_power(best_device)
            original_fps = best_device.current_fps[model_name]
            best_device.current_fps[model_name] += required_fps
            new_power = update_device_power(best_device)
            best_device.current_fps[model_name] = original_fps
            update_device_power(best_device)
            
            power_increase = new_power - current_power
            
            # Update tracking
            self.last_used_device = best_device
            self.device_switch_time = time.time()
            
            return best_device, power_increase
        else:
            # Find the device with the most available capacity for this model (not in recovery)
            best_device = None
            max_available_fps = -1
            
            for device in self.devices:
                # Skip devices in recovery mode
                if self.device_cooldown[device.name]["overloaded"]:
                    continue
                    
                if model_name in device.max_fps:
                    available_fps = device.max_fps[model_name] - device.current_fps[model_name]
                    if available_fps > max_available_fps:
                        max_available_fps = available_fps
                        best_device = device
            
            if best_device is not None:
                # Calculate power increase
                current_power = update_device_power(best_device)
                original_fps = best_device.current_fps[model_name]
                best_device.current_fps[model_name] += min(required_fps, max_available_fps)
                new_power = update_device_power(best_device)
                best_device.current_fps[model_name] = original_fps
                update_device_power(best_device)
                
                power_increase = new_power - current_power
                
                # Update tracking
                self.last_used_device = best_device
                self.device_switch_time = time.time()
                
                return best_device, power_increase
            
            # If all devices are in recovery, return the first device
            return self.devices[0], float('inf')
    
    def handle_request(self, model_name=None, required_fps=0, image=None):
        """Process an incoming request, using AHP to select model and device if not specified"""
        if required_fps <= 0:
            return False, "Invalid FPS requirement"
        
        if image:
            self.current_image = image
        # If model not specified, select one using AHP
        if model_name is None:
            selected_model = self.select_model_using_ahp(required_fps)
            model_name = selected_model.name
            self.last_used_model = selected_model
        
        # Select the optimal device using AHP
        best_device, power_increase = self.select_device_using_ahp(model_name, required_fps)
        
        # Ensure best_device is not None (should never happen with the modified code)
        if best_device is None:
            # As a last resort, use the first device
            best_device = self.devices[0]
            power_increase = 0  # We don't know the power increase
        
        self.last_used_device = best_device
        
        # Check if the device is in recovery mode or queue has requests
        if self.device_cooldown[best_device.name]["overloaded"] or not best_device.request_queue.empty():
            # Queue this request for later processing
            request = {
                "model_name": model_name,
                "required_fps": required_fps,
                "timestamp": time.time(),
                "power_increase": power_increase,
                "image": image
            }
            best_device.request_queue.put(request)
            
            queue_position = best_device.request_queue.qsize()
            
            self.request_history.append({
                "timestamp": len(self.request_history),
                "action": "queue_request",
                "model": model_name,
                "requested_fps": required_fps,
                "device": best_device.name,
                "queue_position": queue_position,
                "reason": "device_overloaded" if self.device_cooldown[best_device.name]["overloaded"] else "queue_not_empty"
            })
            
            return True, f"Request for {required_fps} FPS of {model_name} queued on {best_device.name} (position {queue_position})"
        
        # Determine how much FPS we can actually allocate
        max_available_fps = best_device.max_fps.get(model_name, 0) - best_device.current_fps.get(model_name, 0)
        allocatable_fps = min(required_fps, max_available_fps)
        
        # Check if we'll be overloading the device
        will_overload = allocatable_fps < required_fps
        
        # If we're going to overload, check utilization threshold
        utilization_threshold = 0.9
        if will_overload:
            current_utilization = sum(best_device.current_fps.values()) / sum(best_device.max_fps.values())
            new_utilization = (sum(best_device.current_fps.values()) + allocatable_fps) / sum(best_device.max_fps.values())
            
            if new_utilization > utilization_threshold:
                # Device will be overloaded - store its current state before we continue
                self.device_cooldown[best_device.name]["overloaded"] = True
                self.device_cooldown[best_device.name]["workload_before_overload"] = best_device.current_fps.copy()
                
                # Set recovery time (5 seconds from now)
                recovery_time = time.time() + 5.0
                self.device_cooldown[best_device.name]["recovery_until"] = recovery_time
        
        # Allocate what we can
        if allocatable_fps > 0:
            best_device.current_fps[model_name] += allocatable_fps
            update_device_power(best_device)
            
            self.request_history.append({
                "timestamp": len(self.request_history),
                "action": "allocate_request",
                "model": model_name,
                "requested_fps": required_fps,
                "allocated_fps": allocatable_fps,
                "device": best_device.name,
                "power_increase": power_increase,
                "total_device_power": best_device.current_power,
                "overloaded": will_overload
            })
            
            if will_overload:
                return True, f"Allocated {allocatable_fps}/{required_fps} FPS of {model_name} to {best_device.name} (device marked for recovery)"
            else:
                return True, f"Allocated {allocatable_fps} FPS of {model_name} to {best_device.name}"
        else:
            # If we couldn't allocate anything, queue the request
            request = {
                "model_name": model_name,
                "required_fps": required_fps,
                "timestamp": time.time(),
                "power_increase": power_increase
            }
            best_device.request_queue.put(request)
            
            queue_position = best_device.request_queue.qsize()
            
            self.request_history.append({
                "timestamp": len(self.request_history),
                "action": "queue_request_no_capacity",
                "model": model_name,
                "requested_fps": required_fps,
                "device": best_device.name,
                "queue_position": queue_position
            })
            
            return True, f"No capacity available for {required_fps} FPS of {model_name}, queued on {best_device.name} (position {queue_position})"
    
    def process_device_queue(self, device):
        """Process queued requests for a specific device"""
        # Skip if device is in recovery or queue is empty
        if self.device_cooldown[device.name]["overloaded"] or device.request_queue.empty():
            return False, f"Device {device.name} not ready for queue processing"
        
        # Get the next request from the queue
        request = device.request_queue.get()
        
        model_name = request["model_name"]
        required_fps = request["required_fps"]
        queued_time = time.time() - request["timestamp"]
        image = request.get("image", None)
        # Determine how much FPS we can actually allocate
        max_available_fps = device.max_fps.get(model_name, 0) - device.current_fps.get(model_name, 0)
        allocatable_fps = min(required_fps, max_available_fps)
        
        # Check if we'll be overloading the device
        will_overload = allocatable_fps < required_fps
        
        # If we're going to overload, check utilization threshold
        utilization_threshold = 0.9
        if will_overload:
            current_utilization = sum(device.current_fps.values()) / sum(device.max_fps.values())
            new_utilization = (sum(device.current_fps.values()) + allocatable_fps) / sum(device.max_fps.values())
            
            if new_utilization > utilization_threshold:
                # Device would be overloaded - put the request back in the queue
                device.request_queue.put(request)
                
                self.request_history.append({
                    "timestamp": len(self.request_history),
                    "action": "requeue_request",
                    "model": model_name,
                    "requested_fps": required_fps,
                    "device": device.name,
                    "reason": "would_overload",
                    "queued_time": queued_time,
                    "image": image
                })
                
                return False, f"Request for {required_fps} FPS of {model_name} requeued on {device.name}"
        
        # Allocate what we can
        if allocatable_fps > 0:
            device.current_fps[model_name] += allocatable_fps
            update_device_power(device)
            
            self.request_history.append({
                "timestamp": len(self.request_history),
                "action": "process_queued_request",
                "model": model_name,
                "requested_fps": required_fps,
                "allocated_fps": allocatable_fps,
                "device": device.name,
                "queued_time": queued_time,
                "total_device_power": device.current_power,
                "overloaded": will_overload,
                "image": image
            })
            
            return True, f"Processed queued request: {allocatable_fps}/{required_fps} FPS of {model_name} on {device.name}"
        else:
            # If we couldn't allocate anything, put the request back in the queue
            device.request_queue.put(request)
            
            self.request_history.append({
                "timestamp": len(self.request_history),
                "action": "requeue_request",
                "model": model_name,
                "requested_fps": required_fps,
                "device": device.name,
                "reason": "no_capacity",
                "queued_time": queued_time,
                "image": image
            })
            
            return False, f"No capacity for queued request on {device.name}, requeued"
    
    def process_all_queues(self):
        """Process queues for all devices"""
        results = []
        for device in self.devices:
            # Skip overloaded devices
            if self.device_cooldown[device.name]["overloaded"]:
                continue
                
            # Process queue until it's empty or we can't process more requests
            while not device.request_queue.empty():
                success, message = self.process_device_queue(device)
                results.append(message)
                
                # Stop if we couldn't process the request
                if not success:
                    break
        
        return results
    
    def check_recovery_status(self):
        """Check and update recovery status of overloaded devices"""
        current_time = time.time()
        updates = []
        
        for device in self.devices:
            device_name = device.name
            if self.device_cooldown[device_name]["overloaded"]:
                recovery_time = self.device_cooldown[device_name]["recovery_until"]
                
                if current_time >= recovery_time:
                    # Device has recovered
                    self.device_cooldown[device_name]["overloaded"] = False
                    updates.append(f"Device {device_name} recovered from overload")
                    
                    # Process queued requests for this device
                    while not device.request_queue.empty():
                        success, message = self.process_device_queue(device)
                        updates.append(message)
                        
                        # Stop if we couldn't process the request
                        if not success:
                            break
        
        return updates
    
    def release_request(self, model_name, fps_to_release, device_name):
        """Release resources from a completed request"""
        for device in self.devices:
            if device.name == device_name:
                # Check if device is in recovery mode
                if self.device_cooldown[device.name]["overloaded"]:
                    # See if we can bring the device out of recovery mode
                    device.current_fps[model_name] -= min(device.current_fps[model_name], fps_to_release)
                    update_device_power(device)
                    
                    # Check if we're back to pre-overload levels
                    current_total_fps = sum(device.current_fps.values())
                    original_total_fps = sum(self.device_cooldown[device.name]["workload_before_overload"].values())
                    
                    if current_total_fps <= original_total_fps:
                        # Device is back to normal
                        self.device_cooldown[device.name]["overloaded"] = False
                        
                        # Process queued requests
                        queue_results = []
                        while not device.request_queue.empty():
                            success, message = self.process_device_queue(device)
                            queue_results.append(message)
                            
                            # Stop if we couldn't process the request
                            if not success:
                                break
                        
                        return True, f"Released {fps_to_release} FPS of {model_name} from {device.name} (device recovered, processed {len(queue_results)} queued requests)"
                    
                    return True, f"Released {fps_to_release} FPS of {model_name} from {device.name} (device still in recovery)"
                else:
                    # Normal release
                    if device.current_fps[model_name] >= fps_to_release:
                        device.current_fps[model_name] -= fps_to_release
                        update_device_power(device)
                        
                        # Process queued requests
                        queue_results = []
                        while not device.request_queue.empty():
                            success, message = self.process_device_queue(device)
                            queue_results.append(message)
                            
                            # Stop if we couldn't process the request
                            if not success:
                                break
                        
                        queue_message = f" (processed {len(queue_results)} queued requests)" if queue_results else ""
                        return True, f"Released {fps_to_release} FPS of {model_name} from {device.name}{queue_message}"
                    else:
                        # Release only what's available
                        available_fps = device.current_fps[model_name]
                        device.current_fps[model_name] = 0
                        update_device_power(device)
                        
                        # Process queued requests
                        queue_results = []
                        while not device.request_queue.empty():
                            success, message = self.process_device_queue(device)
                            queue_results.append(message)
                            
                            # Stop if we couldn't process the request
                            if not success:
                                break
                        
                        queue_message = f" (processed {len(queue_results)} queued requests)" if queue_results else ""
                        return True, f"Released available {available_fps} FPS of {model_name} from {device.name}{queue_message}"
        return False, "Device not found"
    
    def system_status(self):
        """Get the current system status"""
        # First check if any devices have recovered
        recovery_updates = self.check_recovery_status()
        
        # Also process queues periodically
        process_results = self.process_all_queues()
        if process_results:
            recovery_updates.extend(process_results)
        
        status = []
        for device in self.devices:
            update_device_power(device)
            
            # Calculate overall utilization
            total_capacity = sum(device.max_fps.values())
            total_used = sum(device.current_fps.values())
            overall_utilization = (total_used / total_capacity * 100) if total_capacity > 0 else 0
            
            status.append({
                "device": device.name,
                "current_power": device.current_power,
                "models": {model: fps for model, fps in device.current_fps.items() if fps > 0},
                "utilization": {model: (fps / device.max_fps[model] * 100) if device.max_fps[model] > 0 else 0 
                               for model, fps in device.current_fps.items() if fps > 0},
                "overall_utilization": overall_utilization,
                "overloaded": self.device_cooldown[device.name]["overloaded"],
                "recovery_time_left": max(0, self.device_cooldown[device.name]["recovery_until"] - time.time()) 
                                     if self.device_cooldown[device.name]["overloaded"] else 0,
                "queue_size": device.request_queue.qsize()
            })
        
        return {"devices": status, "recovery_events": recovery_updates}

# Reset devices to initial state
def reset_devices():
    new_devices = []
    for device in devices:
        new_device = Device(
            name=device.name,
            power_coefficients=device.power_coefficients.copy(),
            max_fps=device.max_fps.copy(),
            current_fps={model_name: 0 for model_name in device.current_fps},
            current_power=0
        )
        new_devices.append(new_device)
    return new_devices

# Generate a set of requests that will be used across all scenarios
def generate_workload(num_requests=100):
    workload = []
    for i in range(num_requests):
        # Random model
        model = random.choice(models)
        # Random FPS requirement (10% chance of high demand)
        if random.random() < 0.1:
            # High demand - might cause overload
            fps_request = random.uniform(0.8, 1.2) * model.max_fps
        else:
            # Normal demand
            fps_request = random.uniform(0.1, 0.6) * model.max_fps
        
        # Generate a random request duration
        duration = random.uniform(0.5, 2.0)
        
        workload.append({
            "id": i,
            "model_name": model.name,
            "fps_request": fps_request,
            "duration": duration
        })
    
    return workload

# Scenario 1: Scheduler decides both device and model
def run_scenario_scheduler_decides_all(workload, num_requests=100):
    logging.info("\nRunning Scenario 1: Scheduler decides both device and model")
    # Initialize devices and scheduler
    local_devices = reset_devices()
    scheduler = AHPQueueScheduler(local_devices, models)
    
    # Tracking structures for metrics
    metrics = {
        "timestamps": [],
        "power_consumption": {device.name: [] for device in local_devices},
        "fps_allocation": {device.name: [] for device in local_devices},
        "total_power": [],
        "total_fps": [],
        "efficiency": [],
        "utilization": {device.name: [] for device in local_devices},
        "model_distribution": {model.name: 0 for model in models},
        "device_distribution": {device.name: 0 for device in local_devices},
        "allocation_success": 0,
        "requests_queued": 0,
        "queue_sizes": {device.name: [] for device in local_devices},
        "overload_events": []
    }
    
    # Tracking active requests
    active_requests = {}  # Maps request_id -> (completion_time, device, model, fps)
    
    # Simulation time
    current_time = 0
    
    for i, request in enumerate(workload[:num_requests]):
        # Update time
        current_time += random.uniform(0.1, 0.3)
        
        # Process completed requests
        completed_requests = []
        for req_id, (completion_time, device_name, model_name, fps) in list(active_requests.items()):
            if current_time >= completion_time:
                # Release resources
                success, message = scheduler.release_request(model_name, fps, device_name)
                completed_requests.append(req_id)
        
        # Remove completed requests
        for req_id in completed_requests:
            if req_id in active_requests:
                del active_requests[req_id]
        
        # Process the new request - Scheduler decides both model and device
        fps_request = request["fps_request"]
        success, message = scheduler.handle_request(model_name=None, required_fps=fps_request)
        
        if success:
            # Parse the allocation result
            if "queued" in message:
                metrics["requests_queued"] += 1
            else:
                # Find allocated FPS amount from the message
                try:
                    # Extract information from the most recent request in history
                    latest_req = scheduler.request_history[-1]
                    
                    if "action" in latest_req and latest_req["action"] == "allocate_request":
                        allocated_fps = latest_req["allocated_fps"]
                        device_name = latest_req["device"]
                        model_name = latest_req["model"]
                        
                        # Schedule completion
                        completion_time = current_time + request["duration"]
                        active_requests[f"req-{i}"] = (completion_time, device_name, model_name, allocated_fps)
                        
                        metrics["allocation_success"] += 1
                        metrics["model_distribution"][model_name] += 1
                        metrics["device_distribution"][device_name] += 1
                        
                        # Check if this caused an overload
                        if latest_req.get("overloaded", False):
                            for device in local_devices:
                                if device.name == device_name:
                                    recovery_time = scheduler.device_cooldown[device.name]["recovery_until"]
                                    
                                    metrics["overload_events"].append({
                                        "time": current_time,
                                        "device": device_name,
                                        "recovery_time": recovery_time
                                    })
                except Exception as e:
                    logging.info(f"Error processing request result: {e}")
        
        # Get system status and record metrics
        status = scheduler.system_status()
        
        # Record metrics
        for device_status in status["devices"]:
            device_name = device_status["device"]
            metrics["power_consumption"][device_name].append(device_status["current_power"])
            total_fps = sum(device_status["models"].values())
            metrics["fps_allocation"][device_name].append(total_fps)
            metrics["utilization"][device_name].append(device_status["overall_utilization"])
            metrics["queue_sizes"][device_name].append(device_status["queue_size"])
        
        metrics["timestamps"].append(current_time)
        total_power = sum(device_status["current_power"] for device_status in status["devices"])
        total_fps = sum(sum(device_status["models"].values()) for device_status in status["devices"])
        
        metrics["total_power"].append(total_power)
        metrics["total_fps"].append(total_fps)
        
        # Calculate efficiency (power per FPS)
        if total_fps > 0:
            metrics["efficiency"].append(total_power / total_fps)
        else:
            metrics["efficiency"].append(0)
    
    logging.info(f"Completed {metrics['allocation_success']} allocations, {metrics['requests_queued']} queued")
    
    return metrics

# Scenario 2: User decides device, scheduler decides model
def run_scenario_user_decides_device(workload, num_requests=100):
    logging.info("\nRunning Scenario 2: User decides device, scheduler decides model")
    
    # Initialize devices and scheduler
    local_devices = reset_devices()
    scheduler = AHPQueueScheduler(local_devices, models)
    
    # Tracking structures for metrics
    metrics = {
        "timestamps": [],
        "power_consumption": {device.name: [] for device in local_devices},
        "fps_allocation": {device.name: [] for device in local_devices},
        "total_power": [],
        "total_fps": [],
        "efficiency": [],
        "utilization": {device.name: [] for device in local_devices},
        "model_distribution": {model.name: 0 for model in models},
        "device_distribution": {device.name: 0 for device in local_devices},
        "allocation_success": 0,
        "requests_queued": 0,
        "queue_sizes": {device.name: [] for device in local_devices},
        "overload_events": []
    }
    
    # Tracking active requests
    active_requests = {}  # Maps request_id -> (completion_time, device, model, fps)
    
    # Simulation time
    current_time = 0
    
    for i, request in enumerate(workload[:num_requests]):
        # Update time
        current_time += random.uniform(0.1, 0.3)
        
        # Process completed requests
        completed_requests = []
        for req_id, (completion_time, device_name, model_name, fps) in list(active_requests.items()):
            if current_time >= completion_time:
                # Release resources
                success, message = scheduler.release_request(model_name, fps, device_name)
                completed_requests.append(req_id)
        
        # Remove completed requests
        for req_id in completed_requests:
            if req_id in active_requests:
                del active_requests[req_id]
        
        # Process the new request - User decides device, scheduler decides model
        fps_request = request["fps_request"]
        
        # User selects a device (round-robin for simulation)
        user_device = local_devices[i % len(local_devices)]
        
        # we'll select a model using AHP first, then handle the request with that model
        # but track that it should go to the user's selected device
        
        # Let scheduler select model
        selected_model = scheduler.select_model_using_ahp(fps_request)
        model_name = selected_model.name
        
        # Check if user's device is overloaded
        if scheduler.device_cooldown[user_device.name]["overloaded"]:
            # Queue the request on user's device directly
            user_device.request_queue.put({
                "model_name": model_name,
                "required_fps": fps_request,
                "timestamp": current_time
            })
            
            metrics["requests_queued"] += 1
        else:
            # Determine how much FPS we can allocate on user's device
            available_fps = user_device.max_fps[model_name] - user_device.current_fps[model_name]
            allocatable_fps = min(fps_request, available_fps)
            
            if allocatable_fps > 0:
                # Allocate resources on user's device
                user_device.current_fps[model_name] += allocatable_fps
                update_device_power(user_device)
                
                # Check if this will overload the device
                utilization = sum(user_device.current_fps.values()) / sum(user_device.max_fps.values())
                will_overload = utilization > 0.9 or allocatable_fps < fps_request
                
                if will_overload:
                    # Mark as overloaded
                    recovery_time = current_time + random.uniform(2.0, 4.0)
                    scheduler.device_cooldown[user_device.name]["overloaded"] = True
                    scheduler.device_cooldown[user_device.name]["recovery_until"] = recovery_time
                    scheduler.device_cooldown[user_device.name]["workload_before_overload"] = user_device.current_fps.copy()
                    
                    metrics["overload_events"].append({
                        "time": current_time,
                        "device": user_device.name,
                        "recovery_time": recovery_time
                    })
                
                # Schedule completion
                completion_time = current_time + request["duration"]
                active_requests[f"req-{i}"] = (completion_time, user_device.name, model_name, allocatable_fps)
                
                metrics["allocation_success"] += 1
                metrics["model_distribution"][model_name] += 1
                metrics["device_distribution"][user_device.name] += 1
            else:
                # Queue the request
                user_device.request_queue.put({
                    "model_name": model_name,
                    "required_fps": fps_request,
                    "timestamp": current_time
                })
                
                metrics["requests_queued"] += 1
        
        # Get system status and record metrics
        status = scheduler.system_status()
        
        # Record metrics
        for device_status in status["devices"]:
            device_name = device_status["device"]
            metrics["power_consumption"][device_name].append(device_status["current_power"])
            total_fps = sum(device_status["models"].values())
            metrics["fps_allocation"][device_name].append(total_fps)
            metrics["utilization"][device_name].append(device_status["overall_utilization"])
            metrics["queue_sizes"][device_name].append(device_status["queue_size"])
        
        metrics["timestamps"].append(current_time)
        total_power = sum(device_status["current_power"] for device_status in status["devices"])
        total_fps = sum(sum(device_status["models"].values()) for device_status in status["devices"])
        
        metrics["total_power"].append(total_power)
        metrics["total_fps"].append(total_fps)
        
        # Calculate efficiency (power per FPS)
        if total_fps > 0:
            metrics["efficiency"].append(total_power / total_fps)
        else:
            metrics["efficiency"].append(0)
    
    logging.info(f"Completed {metrics['allocation_success']} allocations, {metrics['requests_queued']} queued")
    return metrics

# Scenario 3: User decides model, scheduler decides device
def run_scenario_user_decides_model(workload, num_requests=100):
    logging.info("\nRunning Scenario 3: User decides model, scheduler decides device")
    
    # Initialize devices and scheduler
    local_devices = reset_devices()
    scheduler = AHPQueueScheduler(local_devices, models)
    
    # Tracking structures for metrics
    metrics = {
        "timestamps": [],
        "power_consumption": {device.name: [] for device in local_devices},
        "fps_allocation": {device.name: [] for device in local_devices},
        "total_power": [],
        "total_fps": [],
        "efficiency": [],
        "utilization": {device.name: [] for device in local_devices},
        "model_distribution": {model.name: 0 for model in models},
        "device_distribution": {device.name: 0 for device in local_devices},
        "allocation_success": 0,
        "requests_queued": 0,
        "queue_sizes": {device.name: [] for device in local_devices},
        "overload_events": []
    }
    
    # Tracking active requests
    active_requests = {}  # Maps request_id -> (completion_time, device, model, fps)
    
    # Simulation time
    current_time = 0
    
    for i, request in enumerate(workload[:num_requests]):
        # Update time
        current_time += random.uniform(0.1, 0.3)
        
        # Process completed requests
        completed_requests = []
        for req_id, (completion_time, device_name, model_name, fps) in list(active_requests.items()):
            if current_time >= completion_time:
                # Release resources
                success, message = scheduler.release_request(model_name, fps, device_name)
                completed_requests.append(req_id)
        
        # Remove completed requests
        for req_id in completed_requests:
            if req_id in active_requests:
                del active_requests[req_id]
        
        # Process the new request - User decides model, scheduler decides device
        fps_request = request["fps_request"]
        model_name = request["model_name"]  # Use the model from workload
        
        # This scenario is directly supported by the scheduler API
        success, message = scheduler.handle_request(model_name=model_name, required_fps=fps_request)
        
        if success:
            # Parse the allocation result
            if "queued" in message:
                metrics["requests_queued"] += 1
            else:
                # Find allocated FPS amount from the message
                try:
                    # Extract information from the most recent request in history
                    latest_req = scheduler.request_history[-1]
                    
                    if "action" in latest_req and latest_req["action"] == "allocate_request":
                        allocated_fps = latest_req["allocated_fps"]
                        device_name = latest_req["device"]
                        
                        # Schedule completion
                        completion_time = current_time + request["duration"]
                        active_requests[f"req-{i}"] = (completion_time, device_name, model_name, allocated_fps)
                        
                        metrics["allocation_success"] += 1
                        metrics["model_distribution"][model_name] += 1
                        metrics["device_distribution"][device_name] += 1
                        
                        # Check if this caused an overload
                        if latest_req.get("overloaded", False):
                            for device in local_devices:
                                if device.name == device_name:
                                    recovery_time = scheduler.device_cooldown[device.name]["recovery_until"]
                                    
                                    metrics["overload_events"].append({
                                        "time": current_time,
                                        "device": device_name,
                                        "recovery_time": recovery_time
                                    })
                except Exception as e:
                    logging.info(f"Error processing request result: {e}")
        
        # Get system status and record metrics
        status = scheduler.system_status()
        
        # Record metrics
        for device_status in status["devices"]:
            device_name = device_status["device"]
            metrics["power_consumption"][device_name].append(device_status["current_power"])
            total_fps = sum(device_status["models"].values())
            metrics["fps_allocation"][device_name].append(total_fps)
            metrics["utilization"][device_name].append(device_status["overall_utilization"])
            metrics["queue_sizes"][device_name].append(device_status["queue_size"])
        
        metrics["timestamps"].append(current_time)
        total_power = sum(device_status["current_power"] for device_status in status["devices"])
        total_fps = sum(sum(device_status["models"].values()) for device_status in status["devices"])
        
        metrics["total_power"].append(total_power)
        metrics["total_fps"].append(total_fps)
        
        # Calculate efficiency (power per FPS)
        if total_fps > 0:
            metrics["efficiency"].append(total_power / total_fps)
        else:
            metrics["efficiency"].append(0)
    
    logging.info(f"Completed {metrics['allocation_success']} allocations, {metrics['requests_queued']} queued")
    return metrics


# Scenario 2 (Modified): User decides device (initially Raspberry Pi), scheduler decides model
def run_scenario_user_decides_device_modified(workload, num_requests=100):
    # Initialize devices and scheduler
    local_devices = reset_devices()
    scheduler = AHPQueueScheduler(local_devices, models)
    
    # Find the Raspberry Pi device
    raspberry_pi = None
    other_devices = []
    for device in local_devices:
        if "Raspberry Pi" in device.name:
            raspberry_pi = device
        else:
            other_devices.append(device)
    
    if not raspberry_pi:
        print("Error: Raspberry Pi device not found")
        return None
        
    # Tracking structures for metrics
    metrics = {
        "timestamps": [],
        "power_consumption": {device.name: [] for device in local_devices},
        "fps_allocation": {device.name: [] for device in local_devices},
        "total_power": [],
        "total_fps": [],
        "efficiency": [],
        "utilization": {device.name: [] for device in local_devices},
        "model_distribution": {model.name: 0 for model in models},
        "device_distribution": {device.name: 0 for device in local_devices},
        "allocation_success": 0,
        "requests_queued": 0,
        "queue_sizes": {device.name: [] for device in local_devices},
        "overload_events": []
    }
    
    # Tracking active requests
    active_requests = {}  # Maps request_id -> (completion_time, device, model, fps)
    
    # Simulation time
    current_time = 0
    
    # Specific FPS values to use
    fps_values = [1, 10, 30, 60, 120, 240, 500, 600, 800, 1000, 800, 600, 500, 240]
    
    # Create proper request objects with durations
    requests = []
    for i, fps in enumerate(fps_values):
        requests.append({
            "id": i,
            "fps_request": fps,
            "duration": 5.0,  # Fixed duration of 5 seconds for each request
            "model_name": None  # Will be selected by scheduler
        })
    
    # Point at which we start switching to other devices
    switch_point = len(requests) // 3  # Switch after processing 1/3 of FPS values
    
    # Import time module for sleep functionality
    import time
    
    for i, request in enumerate(requests):
        fps_request = request["fps_request"]
        print(f"\nProcessing iteration {i+1}/{len(requests)} with FPS {fps_request}")
        
        # Update time
        current_time += 10  # 10 seconds between each iteration
        
        # Process completed requests
        completed_requests = []
        for req_id, (completion_time, device_name, model_name, fps) in list(active_requests.items()):
            if current_time >= completion_time:
                # Release resources
                success, message = scheduler.release_request(model_name, fps, device_name)
                completed_requests.append(req_id)
                print(f"Released request {req_id}: {message}")
        
        # Remove completed requests
        for req_id in completed_requests:
            if req_id in active_requests:
                del active_requests[req_id]
        
        # Process the new request - User decides device, scheduler decides model
        # Selection strategy:
        # First 1/3: Always choose Raspberry Pi
        # Remaining: Gradually shift to other devices
        if i < switch_point:
            # First phase: Always choose Raspberry Pi
            user_device = raspberry_pi
            print(f"Initial phase: Selected Raspberry Pi for FPS {fps_request}")
        else:
            # Second phase: Gradually shift to other devices
            # Calculate transition probability based on progress
            progress = (i - switch_point) / (len(requests) - switch_point)
            # Higher probability of choosing other devices as we progress
            prob_other_device = min(0.9, progress * 1.5)  # Cap at 90% to maintain some Pi requests
            
            if random.random() < prob_other_device:
                # Choose one of the other devices
                user_device = random.choice(other_devices)
                print(f"Transition phase: Selected {user_device.name} for FPS {fps_request} (prob={prob_other_device:.2f})")
            else:
                # Still choose Raspberry Pi sometimes
                user_device = raspberry_pi
                print(f"Transition phase: Still selected Raspberry Pi for FPS {fps_request} (prob={1-prob_other_device:.2f})")
                
        # Wait for 10 seconds after each iteration
        print(f"Sleeping for 10 seconds...")
        time.sleep(1)
        
        # Let scheduler select model based on FPS requirement
        # For very high FPS values, we need to be careful about model selection
        eligible_models = [model for model in models if model.max_fps >= fps_request]
        
        if eligible_models:
            selected_model = scheduler.select_model_using_ahp(fps_request)
            # Double-check if selected model can actually handle this FPS
            if selected_model.max_fps < fps_request:
                # Fall back to the model with highest max_fps
                selected_model = max(eligible_models, key=lambda m: m.max_fps)
        else:
            # If no model can handle this FPS directly, select the one with highest capacity
            selected_model = max(models, key=lambda m: m.max_fps)
            
        model_name = selected_model.name
        print(f"Selected model {model_name} (max FPS: {selected_model.max_fps}) for requested FPS: {fps_request}")
        
        # Store model name in request for future reference
        request["model_name"] = model_name
        
        # Check if user's device is overloaded
        if scheduler.device_cooldown[user_device.name]["overloaded"]:
            # Queue the request on user's device directly
            user_device.request_queue.put({
                "model_name": model_name,
                "required_fps": fps_request,
                "timestamp": current_time,
                "power_increase": 0  # We don't calculate this here
            })
            
            metrics["requests_queued"] += 1
        else:
            # Determine how much FPS we can allocate on user's device
            available_fps = user_device.max_fps[model_name] - user_device.current_fps[model_name]
            allocatable_fps = min(fps_request, available_fps)
            
            if allocatable_fps > 0:
                # Calculate power before allocation
                power_before = user_device.current_power
                
                # Allocate resources on user's device
                user_device.current_fps[model_name] += allocatable_fps
                update_device_power(user_device)
                
                # Calculate power increase
                power_increase = user_device.current_power - power_before
                
                # Check if this will overload the device
                utilization = sum(user_device.current_fps.values()) / sum(user_device.max_fps.values())
                will_overload = utilization > 0.9 or allocatable_fps < fps_request
                
                if will_overload:
                    # Mark as overloaded
                    recovery_time = current_time + random.uniform(2.0, 4.0)
                    scheduler.device_cooldown[user_device.name]["overloaded"] = True
                    scheduler.device_cooldown[user_device.name]["recovery_until"] = recovery_time
                    scheduler.device_cooldown[user_device.name]["workload_before_overload"] = user_device.current_fps.copy()
                    
                    metrics["overload_events"].append({
                        "time": current_time,
                        "device": user_device.name,
                        "recovery_time": recovery_time
                    })
                
                # Schedule completion
                completion_time = current_time + request["duration"]
                active_requests[f"req-{i}"] = (completion_time, user_device.name, model_name, allocatable_fps)
                
                # Update metrics
                metrics["allocation_success"] += 1
                metrics["model_distribution"][model_name] += 1
                metrics["device_distribution"][user_device.name] += 1
                
                # Add to request history for consistency
                scheduler.request_history.append({
                    "timestamp": len(scheduler.request_history),
                    "action": "allocate_request",
                    "model": model_name,
                    "requested_fps": fps_request,
                    "allocated_fps": allocatable_fps,
                    "device": user_device.name,
                    "power_increase": power_increase,
                    "total_device_power": user_device.current_power,
                    "overloaded": will_overload
                })
            else:
                # Queue the request
                user_device.request_queue.put({
                    "model_name": model_name,
                    "required_fps": fps_request,
                    "timestamp": current_time,
                    "power_increase": 0  # We don't calculate this here
                })
                
                metrics["requests_queued"] += 1
                
                # Add to request history for consistency
                scheduler.request_history.append({
                    "timestamp": len(scheduler.request_history),
                    "action": "queue_request_no_capacity",
                    "model": model_name,
                    "requested_fps": fps_request,
                    "device": user_device.name,
                    "queue_position": user_device.request_queue.qsize()
                })
        
        # Get system status and record metrics
        status = scheduler.system_status()
        
        # Record metrics
        for device_status in status["devices"]:
            device_name = device_status["device"]
            metrics["power_consumption"][device_name].append(device_status["current_power"])
            total_fps = sum(device_status["models"].values())
            metrics["fps_allocation"][device_name].append(total_fps)
            metrics["utilization"][device_name].append(device_status["overall_utilization"])
            metrics["queue_sizes"][device_name].append(device_status["queue_size"])
        
        metrics["timestamps"].append(current_time)
        total_power = sum(device_status["current_power"] for device_status in status["devices"])
        total_fps = sum(sum(device_status["models"].values()) for device_status in status["devices"])
        
        metrics["total_power"].append(total_power)
        metrics["total_fps"].append(total_fps)
        
        # Calculate efficiency (power per FPS)
        if total_fps > 0:
            metrics["efficiency"].append(total_power / total_fps)
        else:
            metrics["efficiency"].append(0)
    
    print(f"Completed {metrics['allocation_success']} allocations, {metrics['requests_queued']} queued")
    print(f"Device distribution: {metrics['device_distribution']}")
    return metrics

# Function to plot the modified scenario results
def plot_modified_scenario_results(metrics, num_requests=100):
    plt.figure(figsize=(15, 12))
    
    if timestamp not in os.listdir("log-images"):
        os.mkdir(f"log-images/{timestamp}")

    # 1. Power consumption by device over time
    plt.subplot(2, 2, 1)
    for device_name, power_values in metrics["power_consumption"].items():
        plt.plot(metrics["timestamps"], power_values, label=device_name)
    plt.title("Power Consumption by Device")
    plt.xlabel("Time")
    plt.ylabel("Power (W)")
    plt.legend()
    plt.grid(True)
    
    # 2. FPS allocation by device over time
    plt.subplot(2, 2, 2)
    for device_name, fps_values in metrics["fps_allocation"].items():
        plt.plot(metrics["timestamps"], fps_values, label=device_name)
    plt.title("FPS Allocation by Device")
    plt.xlabel("Time")
    plt.ylabel("Total FPS")
    plt.legend()
    plt.grid(True)
    
    # 3. Total power vs total FPS
    plt.subplot(2, 2, 3)
    plt.plot(metrics["timestamps"], metrics["total_power"], label="Total Power (W)")
    plt.plot(metrics["timestamps"], metrics["total_fps"], label="Total FPS")
    plt.title("Total System Power vs FPS")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    
    # 4. System efficiency (power per FPS)
    plt.subplot(2, 2, 4)
    plt.plot(metrics["timestamps"], metrics["efficiency"])
    plt.title("System Efficiency (Power per FPS)")
    plt.xlabel("Time")
    plt.ylabel("W/FPS")
    plt.grid(True)
    
    plt.tight_layout()
    img_name = f"log-images/{timestamp}/modified_scenario_results{num_requests}.png"
    plt.savefig(img_name)
    plt.close()
    
    # Additional plots for queue sizes and utilization
    plt.figure(figsize=(15, 6))
    
    # 1. Queue sizes
    plt.subplot(1, 2, 1)
    for device_name, queue_values in metrics["queue_sizes"].items():
        plt.plot(metrics["timestamps"], queue_values, label=device_name)
    plt.title("Request Queue Size by Device")
    plt.xlabel("Time")
    plt.ylabel("Queue Size")
    plt.legend()
    plt.grid(True)
    
    # 2. Device utilization
    plt.subplot(1, 2, 2)
    for device_name, util_values in metrics["utilization"].items():
        plt.plot(metrics["timestamps"], util_values, label=device_name)
    plt.title("Device Utilization (%)")
    plt.xlabel("Time")
    plt.ylabel("Utilization (%)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    img_name = f"log-images/{timestamp}/modified_scenario_queues_utilization{num_requests}.png"
    plt.savefig(img_name)
    plt.close()
    
    # Plot model distribution
    plt.figure(figsize=(10, 6))
    model_names = list(metrics["model_distribution"].keys())
    model_counts = list(metrics["model_distribution"].values())
    plt.bar(model_names, model_counts)
    plt.title("Model Distribution in Requests")
    plt.xlabel("Model")
    plt.ylabel("Count")
    img_name = f"log-images/{timestamp}/modified_scenario_model_distribution{num_requests}.png"
    plt.savefig(img_name)
    plt.close()
    
    # Plot device distribution
    plt.figure(figsize=(10, 6))
    device_names = list(metrics["device_distribution"].keys())
    device_counts = list(metrics["device_distribution"].values())
    plt.bar(device_names, device_counts)
    plt.title("Device Distribution in Requests")
    plt.xlabel("Device")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    img_name = f"log-images/{timestamp}/modified_scenario_device_distribution{num_requests}.png"
    plt.savefig(img_name)
    plt.close()

# Function to run the modified scenario and compare with original
def run_comparison():
    # Generate a workload
    # num_req=1000
    modified_metrics = run_scenario_user_decides_device_modified(None)  # Workload not needed, using predefined FPS values
    
    # Plot the results
    plot_modified_scenario_results(modified_metrics)
    
    # Generate a summary of the progressive transitions
    for i, (ts, pi_power) in enumerate(zip(modified_metrics["timestamps"], 
                                          modified_metrics["power_consumption"]["Device3 (Raspberry Pi 4)"])):
        # Find the corresponding FPS values for all devices at this timestamp
        fps_values = {dev: modified_metrics["fps_allocation"][dev][i] for dev in modified_metrics["fps_allocation"]}
        
        # Calculate total FPS and percentage on Pi
        total_fps = sum(fps_values.values())
        pi_percentage = (fps_values["Device3 (Raspberry Pi 4)"] / total_fps * 100) if total_fps > 0 else 0
        
        print(f"Timestamp {ts:.1f}:")
        print(f"  Raspberry Pi: {fps_values['Device3 (Raspberry Pi 4)']:.1f} FPS ({pi_percentage:.1f}% of total)")
        print(f"  Power: {pi_power:.2f}W")
        print(f"  Other devices: {sum(v for k, v in fps_values.items() if 'Raspberry Pi' not in k):.1f} FPS")
        print()
    
    # Device distribution summary
    print("\nDevice Distribution (Modified):")
    for device, count in modified_metrics['device_distribution'].items():
        total_count = sum(modified_metrics['device_distribution'].values())
        percentage = (count / total_count * 100) if total_count > 0 else 0
        print(f"  {device}: {count} ({percentage:.1f}%)")
    
    return modified_metrics