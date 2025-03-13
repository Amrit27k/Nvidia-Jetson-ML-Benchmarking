import logging
import os
from datetime import datetime
from dataclasses import dataclass, field
from queue import Queue
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from scheduler_sim_scenarios import AHPQueueScheduler, IMAGE_ACCURACY
# Define image accuracy data


# Model name mapping between code and accuracy data
MODEL_NAME_MAPPING = {
    "MobileNet": "MobileNetV3",
    "ResNext50": "ResNeXt50",
    "ResNet50": "ResNet50",
    "ResNet18": "ResNet18",
    "SqueezeNet": "SqueezeNet"
}

# Reverse mapping for display
REVERSE_MODEL_MAPPING = {v: k for k, v in MODEL_NAME_MAPPING.items()}

# Setup directory and logging
timestamp = str(datetime.now().strftime('%Y-%m-%dT%H-%M'))
version = "v1.0.4-accuracy-sim"
os.makedirs(f"log-images/{timestamp}-{version}", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    filename=f"log-images/{timestamp}-{version}/logs_inference_benchmark-accuracy.txt",
    filemode="a+",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define data structures
@dataclass
class Model:
    name: str
    max_fps: float  # Maximum FPS this model can achieve on any device

@dataclass
class Device:
    name: str
    
    # Power consumption coefficients (ax³ + bx² + cx + d)
    power_coefficients: dict  # Maps model_name -> [a, b, c, d]
    
    # Maximum FPS achievable for each model on this device
    max_fps: dict  # Maps model_name -> max_fps
    
    # Current allocation of FPS for each model
    current_fps: dict  # Maps model_name -> current_fps
    
    # Current power consumption
    current_power: float = 0.0
    
    # Request queue for this device
    request_queue: Queue = field(default_factory=Queue)


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
        name="D1",  # Jetson Orin Nano
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
        name="D2",  # Jetson Nano
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
        name="D3",  # Raspberry Pi 4
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

# Function to add trend line to a time series plot
def add_trend_line(x, y):
    # Handle empty or single data point case
    if len(x) <= 1 or len(y) <= 1:
        return
    
    # Convert to numpy arrays and handle NaN values
    x_np = np.array(x)
    y_np = np.array(y)
    mask = ~np.isnan(y_np)
    
    if np.sum(mask) <= 1:  # If there's only one valid point or none
        return
    
    # Use a moving average to show the overall trend
    window_size = max(3, len(y) // 8)  # Use a reasonable window size based on data length
    # Ensure window size is odd for centered moving average
    if window_size % 2 == 0:
        window_size += 1
        
    # Create the moving average
    y_smoothed = np.zeros_like(y_np)
    # Handle edge cases (beginning and end of array)
    half_window = window_size // 2
    
    # For the beginning of the array
    for i in range(half_window):
        y_smoothed[i] = np.mean(y_np[:i+half_window+1])
    
    # For the middle of the array (use proper centered window)
    for i in range(half_window, len(y_np) - half_window):
        y_smoothed[i] = np.mean(y_np[i-half_window:i+half_window+1])
    
    # For the end of the array
    for i in range(len(y_np) - half_window, len(y_np)):
        y_smoothed[i] = np.mean(y_np[i-half_window:])
    
    # Plot the trend line
    plt.plot(x_np, y_smoothed, "r--", linewidth=2)

# Calculate accuracy metrics for a scenario
def calculate_accuracy_metrics(model_selections, image_assignments):
    """
    Calculate accuracy metrics based on model selections and image assignments
    
    Args:
        model_selections: List of selected model names in execution order
        image_assignments: Dictionary mapping request index to image name
        
    Returns:
        Dictionary containing accuracy metrics
    """
    # Track image-specific accuracy
    image_models = {}  # Maps image -> list of models used
    image_accuracies = {}  # Maps image -> list of accuracy scores
    
    # Process each model selection
    for i, model_name in enumerate(model_selections):
        if i in image_assignments:
            image = image_assignments[i]
            mapped_model = MODEL_NAME_MAPPING.get(model_name, model_name)
            
            # Record model used for this image
            if image not in image_models:
                image_models[image] = []
            image_models[image].append(model_name)
            
            # Calculate accuracy for this prediction
            if mapped_model in IMAGE_ACCURACY[image]:
                accuracy = IMAGE_ACCURACY[image][mapped_model]
                
                if image not in image_accuracies:
                    image_accuracies[image] = []
                image_accuracies[image].append(accuracy)
    
    # Calculate average accuracy per image
    avg_image_accuracy = {}
    for image, accuracies in image_accuracies.items():
        avg_image_accuracy[image] = sum(accuracies) / len(accuracies)
    
    # Calculate overall average accuracy
    if avg_image_accuracy:
        total_accuracy = sum(avg_image_accuracy.values())
        avg_accuracy = total_accuracy / len(avg_image_accuracy)
    else:
        total_accuracy = 0
        avg_accuracy = 0
    
    return {
        "image_models": image_models,
        "image_accuracies": image_accuracies,
        "avg_image_accuracy": avg_image_accuracy,
        "total_accuracy": total_accuracy,
        "avg_accuracy": avg_accuracy
    }

# Scenario 1 with accuracy tracking: Scheduler decides both device and model
def run_scenario_scheduler_decides_all_modified():
    """
    Run the modified scenario 1 with accuracy tracking for images
    """
    logging.info("\nRunning Modified Scenario 1: Scheduler decides both device and model (with accuracy tracking)")
    print("\nRunning Modified Scenario 1: Scheduler decides both device and model (with accuracy tracking)")
    
    # Initialize devices and scheduler
    local_devices = reset_devices()
    scheduler = AHPQueueScheduler(local_devices, models)
    
    # Gaussian-distributed FPS values 
    fps_values = [1, 8, 23, 54, 110, 199, 318, 461, 621, 777, 915, 1026, 1109, 1170, 1217, 1254, 1284, 1310, 1334, 1360, 1391, 1437, 1500, 1437, 1391, 1360, 1334, 1310, 1284, 1254, 1217, 1170, 1109, 1026, 915, 777, 621, 461, 381, 199, 110, 54, 23, 8, 1]
    
    # Get list of images
    images = list(IMAGE_ACCURACY.keys())
    
    # Assign images to requests in round-robin fashion
    image_assignments = {}
    for i in range(len(fps_values)):
        image_assignments[i] = images[i % len(images)]
    
    # Create proper request objects with durations
    requests = []
    for i, fps in enumerate(fps_values):
        requests.append({
            "id": i,
            "fps_request": fps,
            "duration": 1.0,  # Fixed duration of 1 second for each request
            "model_name": None,  # Will be selected by scheduler
            "image": image_assignments[i]  # Assigned image
        })
    
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
        "overload_events": [],
        "model_selections": [],  # Track the sequence of selected models
        "image_assignments": image_assignments  # Track which image was assigned to each request
    }
    
    # Tracking active requests
    active_requests = {}  # Maps request_id -> (completion_time, device, model, fps, image)
    
    # Simulation time
    current_time = 0
    
    for i, request in enumerate(requests):
        fps_request = request["fps_request"]
        image = request["image"]
        
        logging.info(f"\nProcessing iteration {i+1}/{len(requests)} with FPS {fps_request} for image {image}")
        print(f"Processing iteration {i+1}/{len(requests)} with FPS {fps_request} for image {image}")
        
        # Update time
        current_time += 1  # 1 second between each iteration
        
        # Process completed requests
        completed_requests = []
        for req_id, (completion_time, device_name, model_name, fps, req_image) in list(active_requests.items()):
            if current_time >= completion_time:
                # Release resources
                success, message = scheduler.release_request(model_name, fps, device_name)
                completed_requests.append(req_id)
                logging.info(f"Released request {req_id}: {message}")
        
        # Remove completed requests
        for req_id in completed_requests:
            if req_id in active_requests:
                del active_requests[req_id]
        
        # Process the new request - Scheduler decides both model and device
        success, message = scheduler.handle_request(model_name=None, required_fps=fps_request)
        
        if success:
            # Parse the allocation result
            if "queued" in message:
                metrics["requests_queued"] += 1
                logging.info(f"Request queued: {message}")
            else:
                # Find allocated FPS amount from the message
                try:
                    # Extract information from the most recent request in history
                    latest_req = scheduler.request_history[-1]
                    
                    if "action" in latest_req and latest_req["action"] == "allocate_request":
                        allocated_fps = latest_req["allocated_fps"]
                        device_name = latest_req["device"]
                        model_name = latest_req["model"]
                        
                        # Track model selection for this request
                        metrics["model_selections"].append(model_name)
                        
                        # Schedule completion
                        completion_time = current_time + request["duration"]
                        active_requests[f"req-{i}"] = (completion_time, device_name, model_name, allocated_fps, image)
                        
                        metrics["allocation_success"] += 1
                        metrics["model_distribution"][model_name] += 1
                        metrics["device_distribution"][device_name] += 1
                        
                        # Get accuracy for this model and image
                        mapped_model = MODEL_NAME_MAPPING.get(model_name, model_name)
                        accuracy = IMAGE_ACCURACY[image].get(mapped_model, 0)
                        
                        logging.info(f"Request scheduled to complete at time {completion_time:.1f}")
                        logging.info(f"Allocated {allocated_fps:.1f}/{fps_request:.1f} FPS of {model_name} to {device_name}")
                        logging.info(f"Accuracy for {image} using {model_name}: {accuracy:.2f}%")
                        print(f"Allocated {allocated_fps:.1f}/{fps_request:.1f} FPS of {model_name} to {device_name}")
                        print(f"Accuracy for {image} using {model_name}: {accuracy:.2f}%")
                        
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
                                    logging.info(f"Device {device_name} overloaded, recovery until {recovery_time-current_time:.1f} seconds from now")
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
            
        # Sleep for simulation purposes
        time.sleep(0.1)  # Reduced sleep time for faster simulation
    
    # Calculate accuracy metrics
    accuracy_metrics = calculate_accuracy_metrics(metrics["model_selections"], metrics["image_assignments"])
    metrics["accuracy"] = accuracy_metrics
    
    logging.info(f"Completed {metrics['allocation_success']} allocations, {metrics['requests_queued']} queued")
    logging.info(f"Average accuracy: {accuracy_metrics['avg_accuracy']:.2f}%")
    
    print(f"\nCompleted {metrics['allocation_success']} allocations, {metrics['requests_queued']} queued")
    print(f"Average accuracy: {accuracy_metrics['avg_accuracy']:.2f}%")
    
    # Print accuracy breakdown by image
    print("\nAccuracy by image:")
    for image, avg_acc in accuracy_metrics['avg_image_accuracy'].items():
        print(f"  {image}: {avg_acc:.2f}%")
    
    return metrics

# Scenario 2 with accuracy tracking: User decides device, scheduler decides model
def run_scenario_user_decides_device_modified():
    """
    Run modified scenario 2 with accuracy tracking: User decides device, scheduler decides model
    """
    logging.info("\nRunning Modified Scenario 2: User decides device, scheduler decides model (with accuracy tracking)")
    print("\nRunning Modified Scenario 2: User decides device, scheduler decides model (with accuracy tracking)")
    
    # Initialize devices and scheduler
    local_devices = reset_devices()
    scheduler = AHPQueueScheduler(local_devices, models)
    
    # Find the Raspberry Pi device
    raspberry_pi = None
    other_devices = []
    for device in local_devices:
        if "D2" in device.name:  # Jetson Nano for initial phase
            raspberry_pi = device
        else:
            other_devices.append(device)
    
    if not raspberry_pi:
        logging.error("Error: Jetson Nano (D2) device not found")
        print("Error: Jetson Nano (D2) device not found - cannot run this scenario")
        return None
    
    # Gaussian-distributed FPS values
    fps_values = [1, 8, 23, 54, 110, 199, 318, 461, 621, 777, 915, 1026, 1109, 1170, 1217, 1254, 1284, 1310, 1334, 1360, 1391, 1437, 1500, 1437, 1391, 1360, 1334, 1310, 1284, 1254, 1217, 1170, 1109, 1026, 915, 777, 621, 461, 381, 199, 110, 54, 23, 8, 1]
    
    # Get list of images
    images = list(IMAGE_ACCURACY.keys())
    
    # Assign images to requests in round-robin fashion
    image_assignments = {}
    for i in range(len(fps_values)):
        image_assignments[i] = images[i % len(images)]
    
    # Create proper request objects with durations
    requests = []
    for i, fps in enumerate(fps_values):
        requests.append({
            "id": i,
            "fps_request": fps,
            "duration": 1.0,  # Fixed duration of 1 second for each request
            "model_name": None,  # Will be selected by scheduler
            "image": image_assignments[i]  # Assigned image
        })
    
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
        "overload_events": [],
        "model_selections": [],  # Track the sequence of selected models
        "image_assignments": image_assignments  # Track which image was assigned to each request
    }
    
    # Tracking active requests
    active_requests = {}  # Maps request_id -> (completion_time, device, model, fps, image)
    
    # Simulation time
    current_time = 0
    
    # Point at which we start switching to other devices
    switch_point = len(requests) // 3  # Switch after processing 1/3 of FPS values
    
    for i, request in enumerate(requests):
        fps_request = request["fps_request"]
        image = request["image"]
        
        logging.info(f"\nProcessing iteration {i+1}/{len(requests)} with FPS {fps_request} for image {image}")
        print(f"Processing iteration {i+1}/{len(requests)} with FPS {fps_request} for image {image}")
        
        # Update time
        current_time += 1  # 1 second between each iteration
        
        # Process completed requests
        completed_requests = []
        for req_id, (completion_time, device_name, model_name, fps, req_image) in list(active_requests.items()):
            if current_time >= completion_time:
                # Release resources
                success, message = scheduler.release_request(model_name, fps, device_name)
                completed_requests.append(req_id)
                logging.info(f"Released request {req_id}: {message}")
        
        # Remove completed requests
        for req_id in completed_requests:
            if req_id in active_requests:
                del active_requests[req_id]
        
        # Selection strategy:
        # First 1/3: Always choose Raspberry Pi
        # Remaining: Gradually shift to other devices
        if i < switch_point:
            # First phase: Always choose D2
            user_device = raspberry_pi
            logging.info(f"Initial phase: Selected D2 (Jetson Nano) for FPS {fps_request}")
        else:
            # Second phase: Gradually shift to other devices
            # Calculate transition probability based on progress
            progress = (i - switch_point) / (len(requests) - switch_point)
            # Higher probability of choosing other devices as we progress
            prob_other_device = min(0.9, progress * 1.5)  # Cap at 90% to maintain some D2 requests
            
            if random.random() < prob_other_device:
                # Choose one of the other devices
                user_device = random.choice(other_devices)
                logging.info(f"Transition phase: Selected {user_device.name} for FPS {fps_request} (prob={prob_other_device:.2f})")
            else:
                # Still choose D2 sometimes
                user_device = raspberry_pi
                logging.info(f"Transition phase: Still selected D2 for FPS {fps_request} (prob={1-prob_other_device:.2f})")
        
        # Wait for 0.1 seconds after each iteration
        time.sleep(0.1)
        
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
        logging.info(f"Selected model {model_name} (max FPS: {selected_model.max_fps}) for requested FPS: {fps_request}")
        
        # Store model selection for accuracy tracking
        metrics["model_selections"].append(model_name)
        
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
                
                # Get accuracy for this model and image
                mapped_model = MODEL_NAME_MAPPING.get(model_name, model_name)
                accuracy = IMAGE_ACCURACY[image].get(mapped_model, 0)
                
                # Schedule completion
                completion_time = current_time + request["duration"]
                active_requests[f"req-{i}"] = (completion_time, user_device.name, model_name, allocatable_fps, image)
                
                # Update metrics
                metrics["allocation_success"] += 1
                metrics["model_distribution"][model_name] += 1
                metrics["device_distribution"][user_device.name] += 1
                
                logging.info(f"Allocated {allocatable_fps}/{fps_request} FPS of {model_name} to {user_device.name}")
                logging.info(f"Accuracy for {image} using {model_name}: {accuracy:.2f}%")
                print(f"Allocated {allocatable_fps}/{fps_request} FPS of {model_name} to {user_device.name}")
                print(f"Accuracy for {image} using {model_name}: {accuracy:.2f}%")
                
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
    
    # Calculate accuracy metrics
    accuracy_metrics = calculate_accuracy_metrics(metrics["model_selections"], metrics["image_assignments"])
    metrics["accuracy"] = accuracy_metrics
    
    logging.info(f"Completed {metrics['allocation_success']} allocations, {metrics['requests_queued']} queued")
    logging.info(f"Device distribution: {metrics['device_distribution']}")
    logging.info(f"Average accuracy: {accuracy_metrics['avg_accuracy']:.2f}%")
    
    print(f"\nCompleted {metrics['allocation_success']} allocations, {metrics['requests_queued']} queued")
    print(f"Device distribution: {metrics['device_distribution']}")
    print(f"Average accuracy: {accuracy_metrics['avg_accuracy']:.2f}%")
    
    # Print accuracy breakdown by image
    print("\nAccuracy by image:")
    for image, avg_acc in accuracy_metrics['avg_image_accuracy'].items():
        print(f"  {image}: {avg_acc:.2f}%")
    
    return metrics

# Scenario 3 with accuracy tracking: User decides model, scheduler decides device
def run_scenario_user_decides_model_modified():
    """
    Run modified scenario 3 with accuracy tracking: User decides model, scheduler decides device
    """
    logging.info("\nRunning Modified Scenario 3: User decides model, scheduler decides device (with accuracy tracking)")
    print("\nRunning Modified Scenario 3: User decides model, scheduler decides device (with accuracy tracking)")
    
    # Initialize devices and scheduler
    local_devices = reset_devices()
    scheduler = AHPQueueScheduler(local_devices, models)
    
    # Gaussian-distributed FPS values
    fps_values = [1, 8, 23, 54, 110, 199, 318, 461, 621, 777, 915, 1026, 1109, 1170, 1217, 1254, 1284, 1310, 1334, 1360, 1391, 1437, 1500, 1437, 1391, 1360, 1334, 1310, 1284, 1254, 1217, 1170, 1109, 1026, 915, 777, 621, 461, 381, 199, 110, 54, 23, 8, 1]
    
    # Get list of images
    images = list(IMAGE_ACCURACY.keys())
    
    # Assign images to requests in round-robin fashion
    image_assignments = {}
    for i in range(len(fps_values)):
        image_assignments[i] = images[i % len(images)]
    
    # For this scenario, we'll select the best model for each image from accuracy data
    requests = []
    for i, fps in enumerate(fps_values):
        image = image_assignments[i]
        
        # Find the best model for this image based on accuracy
        best_model_accuracy = 0
        best_model_name = None
        
        for model_name, mapped_name in MODEL_NAME_MAPPING.items():
            if mapped_name in IMAGE_ACCURACY[image]:
                accuracy = IMAGE_ACCURACY[image][mapped_name]
                if accuracy > best_model_accuracy:
                    best_model_accuracy = accuracy
                    best_model_name = model_name
        
        # If no model found (unlikely), use round-robin
        if best_model_name is None:
            best_model_name = models[i % len(models)].name
            
        requests.append({
            "id": i,
            "fps_request": fps,
            "duration": 1.0,  # Fixed duration of 1 second for each request
            "model_name": best_model_name,  # User decides model based on best accuracy
            "image": image,
            "expected_accuracy": best_model_accuracy
        })
    
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
        "overload_events": [],
        "model_selections": [],  # Track the sequence of selected models
        "image_assignments": image_assignments  # Track which image was assigned to each request
    }
    
    # Tracking active requests
    active_requests = {}  # Maps request_id -> (completion_time, device, model, fps, image)
    
    # Simulation time
    current_time = 0
    
    for i, request in enumerate(requests):
        fps_request = request["fps_request"]
        model_name = request["model_name"]
        image = request["image"]
        expected_accuracy = request.get("expected_accuracy", 0)
        
        logging.info(f"\nProcessing iteration {i+1}/{len(requests)} with FPS {fps_request}, Model: {model_name}, Image: {image}")
        logging.info(f"Expected accuracy: {expected_accuracy:.2f}%")
        print(f"Processing iteration {i+1}/{len(requests)} with FPS {fps_request}, Model: {model_name}, Image: {image}")
        print(f"Expected accuracy: {expected_accuracy:.2f}%")
        
        # Track model selection
        metrics["model_selections"].append(model_name)
        
        # Check if the model can actually handle this FPS
        model_obj = next((m for m in models if m.name == model_name), None)
        if model_obj and model_obj.max_fps < fps_request:
            logging.info(f"Warning: Selected model {model_name} has max FPS of {model_obj.max_fps}, which is less than requested {fps_request}")
        
        # Update time
        current_time += 1  # 1 second between each iteration
        
        # Process completed requests
        completed_requests = []
        for req_id, (completion_time, device_name, req_model_name, fps, req_image) in list(active_requests.items()):
            if current_time >= completion_time:
                # Release resources
                success, message = scheduler.release_request(req_model_name, fps, device_name)
                completed_requests.append(req_id)
                logging.info(f"Released request {req_id}: {message}")
        
        # Remove completed requests
        for req_id in completed_requests:
            if req_id in active_requests:
                del active_requests[req_id]
        
        # Process the new request - User decides model, scheduler decides device
        success, message = scheduler.handle_request(model_name=model_name, required_fps=fps_request)
        
        if success:
            # Parse the allocation result
            if "queued" in message:
                metrics["requests_queued"] += 1
                logging.info(f"Request queued: {message}")
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
                        active_requests[f"req-{i}"] = (completion_time, device_name, model_name, allocated_fps, image)
                        
                        metrics["allocation_success"] += 1
                        metrics["model_distribution"][model_name] += 1
                        metrics["device_distribution"][device_name] += 1
                        
                        # Get actual accuracy for this model and image
                        mapped_model = MODEL_NAME_MAPPING.get(model_name, model_name)
                        accuracy = IMAGE_ACCURACY[image].get(mapped_model, 0)
                        
                        logging.info(f"Request scheduled to complete at time {completion_time:.1f}")
                        logging.info(f"Allocated {allocated_fps:.1f}/{fps_request:.1f} FPS of {model_name} to {device_name}")
                        logging.info(f"Accuracy for {image} using {model_name}: {accuracy:.2f}%")
                        print(f"Allocated {allocated_fps:.1f}/{fps_request:.1f} FPS of {model_name} to {device_name}")
                        print(f"Accuracy for {image} using {model_name}: {accuracy:.2f}%")
                        
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
                                    logging.info(f"Device {device_name} overloaded, recovery until {recovery_time-current_time:.1f} seconds from now")
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
            
        # Wait for 0.1 seconds after each iteration
        time.sleep(0.1)
    
    # Calculate accuracy metrics
    accuracy_metrics = calculate_accuracy_metrics(metrics["model_selections"], metrics["image_assignments"])
    metrics["accuracy"] = accuracy_metrics
    
    logging.info(f"Completed {metrics['allocation_success']} allocations, {metrics['requests_queued']} queued")
    logging.info(f"Average accuracy: {accuracy_metrics['avg_accuracy']:.2f}%")
    
    print(f"\nCompleted {metrics['allocation_success']} allocations, {metrics['requests_queued']} queued")
    print(f"Average accuracy: {accuracy_metrics['avg_accuracy']:.2f}%")
    
    # Print accuracy breakdown by image
    print("\nAccuracy by image:")
    for image, avg_acc in accuracy_metrics['avg_image_accuracy'].items():
        print(f"  {image}: {avg_acc:.2f}%")
    
    return metrics

# Scenario 6 with accuracy tracking: Adaptive Model Switching based on FPS & accuracy
def run_scenario_adaptive_model_switching():
    """
    Run the adaptive model switching scenario with accuracy tracking.
    
    User decides model initially, but when FPS exceeds the model's max_fps or when a higher accuracy
    model is available, requests are scheduled to a different model using AHP scheduler.
    """
    logging.info("\nRunning Scenario 6: Adaptive Model Switching with Accuracy Tracking")
    print("\nRunning Scenario 6: Adaptive Model Switching with Accuracy Tracking")
    
    # Initialize devices and scheduler
    local_devices = reset_devices()
    scheduler = AHPQueueScheduler(local_devices, models)
    
    # Gaussian-distributed FPS values
    fps_values = [1, 8, 23, 54, 110, 199, 318, 461, 621, 777, 915, 1026, 1109, 1170, 1217, 1254, 1284, 1310, 1334, 1360, 1391, 1437, 1500, 1437, 1391, 1360, 1334, 1310, 1284, 1254, 1217, 1170, 1109, 1026, 915, 777, 621, 461, 381, 199, 110, 54, 23, 8, 1]
    
    # Get list of images
    images = list(IMAGE_ACCURACY.keys())
    
    # Assign images to requests in round-robin fashion
    image_assignments = {}
    for i in range(len(fps_values)):
        image_assignments[i] = images[i % len(images)]
    
    # Start with a balanced model (not the best, not the worst)
    # For each image, select a model that's not the top performer
    initial_model_for_image = {}
    for image in images:
        # Get model accuracies for this image
        model_accuracies = [(model_name, IMAGE_ACCURACY[image].get(mapped_name, 0)) 
                          for model_name, mapped_name in MODEL_NAME_MAPPING.items()]
        
        # Sort by accuracy (ascending)
        model_accuracies.sort(key=lambda x: x[1])
        
        # Select a model in the middle of the pack
        if len(model_accuracies) >= 3:
            initial_model_for_image[image] = model_accuracies[len(model_accuracies) // 2][0]
        else:
            initial_model_for_image[image] = model_accuracies[0][0]
    
    # Create proper request objects with durations and initial model preferences
    requests = []
    for i, fps in enumerate(fps_values):
        image = image_assignments[i]
        initial_model_name = initial_model_for_image.get(image, models[0].name)
        
        requests.append({
            "id": i,
            "fps_request": fps,
            "duration": 1.0,  # Fixed duration of 1 second for each request
            "model_name": initial_model_name,  # User's initial model preference
            "original_model": initial_model_name,  # Keep track of original selection
            "image": image
        })
    
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
        "overload_events": [],
        "model_switches": [],  # Track when model switches occur
        "model_selections": [],  # Track the sequence of selected models
        "image_assignments": image_assignments  # Track which image was assigned to each request
    }
    
    # Tracking active requests
    active_requests = {}  # Maps request_id -> (completion_time, device, model, fps, image)
    
    # Simulation time
    current_time = 0
    
    for i, request in enumerate(requests):
        fps_request = request["fps_request"]
        initial_model_name = request["model_name"]
        image = request["image"]
        
        logging.info(f"\nProcessing iteration {i+1}/{len(requests)} with FPS {fps_request} for image {image}")
        logging.info(f"User initially selected model: {initial_model_name}")
        print(f"Processing iteration {i+1}/{len(requests)} with FPS {fps_request} for image {image}")
        print(f"User initially selected model: {initial_model_name}")
        
        # Find the initial model object
        initial_model = next((m for m in models if m.name == initial_model_name), None)
        
        # Check if user's selected model can handle the FPS
        model_can_handle = False
        if initial_model:
            model_can_handle = initial_model.max_fps >= fps_request
            if not model_can_handle:
                logging.info(f"Warning: Selected model {initial_model_name} has max FPS of {initial_model.max_fps}, which is less than requested {fps_request}")
                logging.info("Switching to AHP model selection...")
                print(f"Warning: Selected model {initial_model_name} has max FPS of {initial_model.max_fps}, which is less than requested {fps_request}")
                print("Switching to AHP model selection...")
        
        # Get initial model accuracy for this image
        initial_mapped_model = MODEL_NAME_MAPPING.get(initial_model_name, initial_model_name)
        initial_accuracy = IMAGE_ACCURACY[image].get(initial_mapped_model, 0)
        
        # Check if better accuracy models are available
        better_models = []
        for model_name, mapped_name in MODEL_NAME_MAPPING.items():
            if mapped_name in IMAGE_ACCURACY[image]:
                model_accuracy = IMAGE_ACCURACY[image][mapped_name]
                model_obj = next((m for m in models if m.name == model_name), None)
                
                if model_obj and model_accuracy > initial_accuracy + 10:  # At least 10% better accuracy
                    if model_obj.max_fps >= fps_request:  # Can handle the FPS
                        better_models.append((model_name, model_accuracy, model_obj.max_fps))
        
        # Sort better models by accuracy (descending)
        better_models.sort(key=lambda x: x[1], reverse=True)
        
        # Update time
        current_time += 1  # 1 second between each iteration
        
        # Process completed requests
        completed_requests = []
        for req_id, (completion_time, device_name, model_name, fps, req_image) in list(active_requests.items()):
            if current_time >= completion_time:
                # Release resources
                success, message = scheduler.release_request(model_name, fps, device_name)
                completed_requests.append(req_id)
                logging.info(f"Released request {req_id}: {message}")
        
        # Remove completed requests
        for req_id in completed_requests:
            if req_id in active_requests:
                del active_requests[req_id]
        
        # Process the new request based on model selection criteria
        switch_reason = None
        final_model_name = initial_model_name
        
        if not model_can_handle:
            # Need to switch because FPS requirement exceeds model capacity
            switch_reason = "fps_exceeded"
            selected_model = scheduler.select_model_using_ahp(fps_request)
            final_model_name = selected_model.name
        elif better_models:
            # Switch to better accuracy model if available
            switch_reason = "better_accuracy"
            best_model_name = better_models[0][0]
            final_model_name = best_model_name
        
        # Track model selection
        metrics["model_selections"].append(final_model_name)
        
        # Track model switch if it occurred
        if final_model_name != initial_model_name:
            final_mapped_model = MODEL_NAME_MAPPING.get(final_model_name, final_model_name)
            final_accuracy = IMAGE_ACCURACY[image].get(final_mapped_model, 0)
            
            metrics["model_switches"].append({
                "time": current_time,
                "original_model": initial_model_name,
                "new_model": final_model_name,
                "fps_request": fps_request,
                "image": image,
                "original_accuracy": initial_accuracy,
                "new_accuracy": final_accuracy,
                "reason": switch_reason
            })
            
            logging.info(f"Model switched: {initial_model_name} -> {final_model_name}")
            logging.info(f"Original accuracy: {initial_accuracy:.2f}%, New accuracy: {final_accuracy:.2f}%")
            print(f"Model switched: {initial_model_name} -> {final_model_name}")
            print(f"Original accuracy: {initial_accuracy:.2f}%, New accuracy: {final_accuracy:.2f}%")
            print(f"Reason: {switch_reason}")
        
        # Now let scheduler handle the request with the final model
        success, message = scheduler.handle_request(model_name=final_model_name, required_fps=fps_request)
        
        if success:
            # Parse the allocation result
            if "queued" in message:
                metrics["requests_queued"] += 1
                logging.info(f"Request queued: {message}")
            else:
                # Find allocated FPS amount from the message
                try:
                    # Extract information from the most recent request in history
                    latest_req = scheduler.request_history[-1]
                    
                    if "action" in latest_req and latest_req["action"] == "allocate_request":
                        allocated_fps = latest_req["allocated_fps"]
                        device_name = latest_req["device"]
                        
                        # Get accuracy for this model and image
                        mapped_model = MODEL_NAME_MAPPING.get(final_model_name, final_model_name)
                        accuracy = IMAGE_ACCURACY[image].get(mapped_model, 0)
                        
                        # Schedule completion
                        completion_time = current_time + request["duration"]
                        active_requests[f"req-{i}"] = (completion_time, device_name, final_model_name, allocated_fps, image)
                        
                        metrics["allocation_success"] += 1
                        metrics["model_distribution"][final_model_name] += 1
                        metrics["device_distribution"][device_name] += 1
                        
                        logging.info(f"Request scheduled to complete at time {completion_time:.1f}")
                        logging.info(f"Allocated {allocated_fps:.1f}/{fps_request:.1f} FPS of {final_model_name} to {device_name}")
                        logging.info(f"Accuracy for {image} using {final_model_name}: {accuracy:.2f}%")
                        print(f"Allocated {allocated_fps:.1f}/{fps_request:.1f} FPS of {final_model_name} to {device_name}")
                        print(f"Accuracy for {image} using {final_model_name}: {accuracy:.2f}%")
                        
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
                                    logging.info(f"Device {device_name} overloaded, recovery until {recovery_time-current_time:.1f} seconds from now")
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
            
        # Wait for 0.1 seconds after each iteration
        time.sleep(0.1)
    
    # Calculate accuracy metrics
    accuracy_metrics = calculate_accuracy_metrics(metrics["model_selections"], metrics["image_assignments"])
    metrics["accuracy"] = accuracy_metrics
    
    # Print summary of model switching
    logging.info("\nModel Switching Summary:")
    print("\nModel Switching Summary:")
    if metrics["model_switches"]:
        for switch in metrics["model_switches"]:
            logging.info(f"Image {switch['image']}: {switch['original_model']} -> {switch['new_model']}, Accuracy: {switch['original_accuracy']:.2f}% -> {switch['new_accuracy']:.2f}%, Reason: {switch['reason']}")
            print(f"Image {switch['image']}: {switch['original_model']} -> {switch['new_model']}, Accuracy: {switch['original_accuracy']:.2f}% -> {switch['new_accuracy']:.2f}%, Reason: {switch['reason']}")
    else:
        logging.info("No model switches occurred")
        print("No model switches occurred")
        
    logging.info(f"\nCompleted {metrics['allocation_success']} allocations, {metrics['requests_queued']} queued")
    logging.info(f"Average accuracy: {accuracy_metrics['avg_accuracy']:.2f}%")
    
    print(f"\nCompleted {metrics['allocation_success']} allocations, {metrics['requests_queued']} queued")
    print(f"Average accuracy: {accuracy_metrics['avg_accuracy']:.2f}%")
    
    # Print accuracy breakdown by image
    print("\nAccuracy by image:")
    for image, avg_acc in accuracy_metrics['avg_image_accuracy'].items():
        print(f"  {image}: {avg_acc:.2f}%")
    
    return metrics

# Scenario 7 with accuracy tracking: Cross-device model consistency
def run_scenario_cross_device_model_consistency():
    """
    Run the cross-device model consistency scenario with accuracy tracking.
    
    User decides model and the scheduler tries to keep the same model
    across different devices for maximum accuracy consistency.
    """
    logging.info("\nRunning Scenario 7: Cross-Device Model Consistency with Accuracy Tracking")
    print("\nRunning Scenario 7: Cross-Device Model Consistency with Accuracy Tracking")
    
    # Initialize devices and scheduler
    local_devices = reset_devices()
    scheduler = AHPQueueScheduler(local_devices, models)
    
    # Gaussian-distributed FPS values
    fps_values = [1, 8, 23, 54, 110, 199, 318, 461, 621, 777, 915, 1026, 1109, 1170, 1217, 1254, 1284, 1310, 1334, 1360, 1391, 1437, 1500, 1437, 1391, 1360, 1334, 1310, 1284, 1254, 1217, 1170, 1109, 1026, 915, 777, 621, 461, 381, 199, 110, 54, 23, 8, 1]
    
    # Get list of images
    images = list(IMAGE_ACCURACY.keys())
    
    # Assign images to requests in round-robin fashion
    image_assignments = {}
    for i in range(len(fps_values)):
        image_assignments[i] = images[i % len(images)]
    
    # For each image, select the best model based on accuracy
    best_model_for_image = {}
    for image in images:
        # Find the best model for this image
        best_model_name = None
        best_accuracy = 0
        
        for model_name, mapped_name in MODEL_NAME_MAPPING.items():
            if mapped_name in IMAGE_ACCURACY[image]:
                accuracy = IMAGE_ACCURACY[image][mapped_name]
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model_name = model_name
        
        best_model_for_image[image] = best_model_name
    
    # Create proper request objects with durations
    requests = []
    for i, fps in enumerate(fps_values):
        image = image_assignments[i]
        model_name = best_model_for_image.get(image, models[0].name)
        
        requests.append({
            "id": i,
            "fps_request": fps,
            "duration": 1.0,  # Fixed duration of 1 second for each request
            "model_name": model_name,  # User selects the best model for this image
            "image": image
        })
    
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
        "overload_events": [],
        "device_switches": [],  # Track when device switches occur
        "model_selections": [],  # Track the sequence of selected models
        "image_assignments": image_assignments  # Track which image was assigned to each request
    }
    
    # Tracking active requests
    active_requests = {}  # Maps request_id -> (completion_time, device, model, fps, image)
    
    # Simulation time
    current_time = 0
    
    for i, request in enumerate(requests):
        fps_request = request["fps_request"]
        model_name = request["model_name"]
        image = request["image"]
        
        logging.info(f"\nProcessing iteration {i+1}/{len(requests)} with FPS {fps_request}, Model: {model_name}, Image: {image}")
        print(f"Processing iteration {i+1}/{len(requests)} with FPS {fps_request}, Model: {model_name}, Image: {image}")
        
        # Track model selection
        metrics["model_selections"].append(model_name)
        
        # Update time
        current_time += 1  # 1 second between each iteration
        
        # Process completed requests
        completed_requests = []
        for req_id, (completion_time, device_name, req_model_name, fps, req_image) in list(active_requests.items()):
            if current_time >= completion_time:
                # Release resources
                success, message = scheduler.release_request(req_model_name, fps, device_name)
                completed_requests.append(req_id)
                logging.info(f"Released request {req_id}: {message}")
        
        # Remove completed requests
        for req_id in completed_requests:
            if req_id in active_requests:
                del active_requests[req_id]
        
        # Find the best device for this model and FPS
        original_device = None
        device_suitable = False
        best_device = None
        max_available_fps = 0
        
        # First, check if any device can directly handle this model at the requested FPS
        for device in local_devices:
            # Skip overloaded devices
            if scheduler.device_cooldown[device.name]["overloaded"]:
                continue
                
            # Check if this device can handle the model
            if model_name in device.max_fps:
                # Get available FPS capacity for this model on this device
                available_fps = device.max_fps[model_name] - device.current_fps[model_name]
                
                if available_fps >= fps_request:
                    # This device can handle the request directly
                    best_device = device
                    device_suitable = True
                    break
                elif available_fps > max_available_fps:
                    # Keep track of the device with most available capacity
                    max_available_fps = available_fps
                    best_device = device
        
        # Get accuracy for this model and image
        mapped_model = MODEL_NAME_MAPPING.get(model_name, model_name)
        accuracy = IMAGE_ACCURACY[image].get(mapped_model, 0)
        
        if device_suitable:
            # Allocate to the best device
            logging.info(f"Found suitable device: {best_device.name} for {model_name} at {fps_request} FPS")
            logging.info(f"Expected accuracy: {accuracy:.2f}%")
            print(f"Found suitable device: {best_device.name} for {model_name} at {fps_request} FPS")
            print(f"Expected accuracy: {accuracy:.2f}%")
            
            # Calculate power before allocation
            power_before = best_device.current_power
            
            # Allocate resources on best device
            best_device.current_fps[model_name] += fps_request
            update_device_power(best_device)
            
            # Calculate power increase
            power_increase = best_device.current_power - power_before
            
            # Check if this will overload the device
            utilization = sum(best_device.current_fps.values()) / sum(best_device.max_fps.values())
            will_overload = utilization > 0.9
            
            if will_overload:
                # Mark as overloaded
                recovery_time = current_time + random.uniform(2.0, 4.0)
                scheduler.device_cooldown[best_device.name]["overloaded"] = True
                scheduler.device_cooldown[best_device.name]["recovery_until"] = recovery_time
                scheduler.device_cooldown[best_device.name]["workload_before_overload"] = best_device.current_fps.copy()
                
                metrics["overload_events"].append({
                    "time": current_time,
                    "device": best_device.name,
                    "recovery_time": recovery_time
                })
                
                logging.info(f"Device {best_device.name} overloaded, recovery until {recovery_time-current_time:.1f} seconds from now")
            
            # Schedule completion
            completion_time = current_time + request["duration"]
            active_requests[f"req-{i}"] = (completion_time, best_device.name, model_name, fps_request, image)
            
            # Update metrics
            metrics["allocation_success"] += 1
            metrics["model_distribution"][model_name] += 1
            metrics["device_distribution"][best_device.name] += 1
            
            # Add to request history for consistency
            scheduler.request_history.append({
                "timestamp": len(scheduler.request_history),
                "action": "allocate_request",
                "model": model_name,
                "requested_fps": fps_request,
                "allocated_fps": fps_request,
                "device": best_device.name,
                "power_increase": power_increase,
                "total_device_power": best_device.current_power,
                "overloaded": will_overload
            })
            
            logging.info(f"Request scheduled to complete at time {completion_time:.1f}")
            logging.info(f"Allocated {fps_request:.1f} FPS of {model_name} to {best_device.name}")
        else:
            # No device can handle the full FPS, allocate partial FPS to the device with most capacity
            if best_device and max_available_fps > 0:
                # Try to allocate what we can
                logging.info(f"No device can handle full request. Allocating {max_available_fps}/{fps_request} FPS to {best_device.name}")
                logging.info(f"Expected accuracy: {accuracy:.2f}%")
                print(f"No device can handle full request. Allocating {max_available_fps}/{fps_request} FPS to {best_device.name}")
                print(f"Expected accuracy: {accuracy:.2f}%")
                
                # Record a device switch event
                metrics["device_switches"].append({
                    "time": current_time,
                    "model": model_name,
                    "fps_request": fps_request,
                    "allocated_device": best_device.name,
                    "allocated_fps": max_available_fps,
                    "image": image,
                    "accuracy": accuracy
                })
                
                # Calculate power before allocation
                power_before = best_device.current_power
                
                # Allocate resources on best device
                best_device.current_fps[model_name] += max_available_fps
                update_device_power(best_device)
                
                # Calculate power increase
                power_increase = best_device.current_power - power_before
                
                # Check if this will overload the device
                utilization = sum(best_device.current_fps.values()) / sum(best_device.max_fps.values())
                will_overload = utilization > 0.9
                
                if will_overload:
                    # Mark as overloaded
                    recovery_time = current_time + random.uniform(2.0, 4.0)
                    scheduler.device_cooldown[best_device.name]["overloaded"] = True
                    scheduler.device_cooldown[best_device.name]["recovery_until"] = recovery_time
                    scheduler.device_cooldown[best_device.name]["workload_before_overload"] = best_device.current_fps.copy()
                    
                    metrics["overload_events"].append({
                        "time": current_time,
                        "device": best_device.name,
                        "recovery_time": recovery_time
                    })
                
                # Schedule completion
                completion_time = current_time + request["duration"]
                active_requests[f"req-{i}"] = (completion_time, best_device.name, model_name, max_available_fps, image)
                
                # Update metrics
                metrics["allocation_success"] += 1
                metrics["model_distribution"][model_name] += 1
                metrics["device_distribution"][best_device.name] += 1
                
                # Add to request history for consistency
                scheduler.request_history.append({
                    "timestamp": len(scheduler.request_history),
                    "action": "allocate_request",
                    "model": model_name,
                    "requested_fps": fps_request,
                    "allocated_fps": max_available_fps,
                    "device": best_device.name,
                    "power_increase": power_increase,
                    "total_device_power": best_device.current_power,
                    "overloaded": will_overload
                })
                
                logging.info(f"Request scheduled to complete at time {completion_time:.1f}")
                logging.info(f"Partially allocated {max_available_fps:.1f}/{fps_request:.1f} FPS of {model_name} to {best_device.name}")
            else:
                # Queue the request on the device with the least queue size
                target_device = min(local_devices, key=lambda d: d.request_queue.qsize())
                
                target_device.request_queue.put({
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
                    "device": target_device.name,
                    "queue_position": target_device.request_queue.qsize()
                })
                
                logging.info(f"Request queued on {target_device.name} (no device can handle model {model_name} at {fps_request} FPS)")
        
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
            
        # Wait for 0.1 seconds after each iteration
        time.sleep(0.1)
    
    # Calculate accuracy metrics
    accuracy_metrics = calculate_accuracy_metrics(metrics["model_selections"], metrics["image_assignments"])
    metrics["accuracy"] = accuracy_metrics
    
    # Print summary of device switching
    logging.info("\nDevice Switching Summary:")
    print("\nDevice Switching Summary:")
    if metrics["device_switches"]:
        for switch in metrics["device_switches"]:
            logging.info(f"Image {switch['image']}: Model {switch['model']} requested {switch['fps_request']} FPS, allocated {switch['allocated_fps']} FPS on {switch['allocated_device']}, Accuracy: {switch['accuracy']:.2f}%")
            print(f"Image {switch['image']}: Model {switch['model']} requested {switch['fps_request']} FPS, allocated {switch['allocated_fps']} FPS on {switch['allocated_device']}, Accuracy: {switch['accuracy']:.2f}%")
    else:
        logging.info("No device switches occurred")
        print("No device switches occurred")
        
    logging.info(f"\nCompleted {metrics['allocation_success']} allocations, {metrics['requests_queued']} queued")
    logging.info(f"Average accuracy: {accuracy_metrics['avg_accuracy']:.2f}%")
    
    print(f"\nCompleted {metrics['allocation_success']} allocations, {metrics['requests_queued']} queued")
    print(f"Average accuracy: {accuracy_metrics['avg_accuracy']:.2f}%")
    
    # Print accuracy breakdown by image
    print("\nAccuracy by image:")
    for image, avg_acc in accuracy_metrics['avg_image_accuracy'].items():
        print(f"  {image}: {avg_acc:.2f}%")
    
    return metrics

# Plot accuracy metrics
def plot_accuracy_metrics(metrics, scenario_name):
    """Generate plots visualizing accuracy metrics for a scenario with percentages for distributions"""
    if "accuracy" not in metrics:
        logging.error(f"Cannot plot accuracy for {scenario_name}: accuracy metrics not available")
        return
    
    accuracy_metrics = metrics["accuracy"]
    
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Accuracy by image
    plt.subplot(2, 1, 1)
    images = list(accuracy_metrics["avg_image_accuracy"].keys())
    image_accuracies = [accuracy_metrics["avg_image_accuracy"][img] for img in images]
    
    plt.bar(images, image_accuracies)
    plt.title(f"{scenario_name}: Accuracy by Image")
    plt.xlabel("Image")
    plt.ylabel("Average Accuracy (%)")
    # plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    
    # Plot 2: Model distribution as percentages
    plt.subplot(2, 1, 2)
    model_names = list(metrics["model_distribution"].keys())
    model_counts = [metrics["model_distribution"][model] for model in model_names]
    
    # Convert to percentages
    total_models = sum(model_counts)
    if total_models > 0:
        model_percentages = [count / total_models * 100 for count in model_counts]
    else:
        model_percentages = [0] * len(model_counts)
    
    plt.bar(model_names, model_percentages)
    plt.title(f"{scenario_name}: Model Distribution")
    plt.xlabel("Model")
    plt.ylabel("Request Percentage (%)")
    # plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    filename = f"log-images/{timestamp}-{version}/{scenario_name.lower().replace(' ', '_').replace(':', '')}_accuracy_1.png"
    plt.savefig(filename)

    plt.figure(figsize=(14, 10))
    # Plot 3: Model-Image Mapping Heatmap (percentages within each image)
    
    
    # Create matrix of model usage for each image (as percentages)
    image_model_matrix = np.zeros((len(images), len(model_names)))
    
    # Calculate raw counts first
    for i, image in enumerate(images):
        if image in accuracy_metrics["image_models"]:
            for model in accuracy_metrics["image_models"][image]:
                if model in model_names:
                    j = model_names.index(model)
                    image_model_matrix[i, j] += 1
    
    # Convert to percentages per image (row)
    for i in range(len(images)):
        row_sum = np.sum(image_model_matrix[i, :])
        if row_sum > 0:
            image_model_matrix[i, :] = (image_model_matrix[i, :] / row_sum) * 100
    
    plt.imshow(image_model_matrix, cmap='viridis')
    plt.colorbar(label='Percentage (%)')
    plt.title(f"{scenario_name}: Model Usage by Image")
    plt.xlabel("Model")
    plt.ylabel("Image")
    plt.xticks(np.arange(len(model_names)), model_names)
    plt.yticks(np.arange(len(images)), images)
    
    plt.tight_layout()
    filename = f"log-images/{timestamp}-{version}/{scenario_name.lower().replace(' ', '_').replace(':', '')}_accuracy_2.png"
    plt.savefig(filename)
    # Plot 4: Accuracy vs Power Efficiency
    plt.figure(figsize=(14, 10))
    # Calculate average efficiency (power per FPS)
    avg_efficiency = sum(metrics["efficiency"]) / len(metrics["efficiency"]) / 1000  # Convert to W/FPS
    
    # Create scatter plot of models by accuracy and power efficiency
    model_avg_accuracy = {}
    for model_name in model_names:
        # Calculate average accuracy for this model across all instances
        total_acc = 0
        count = 0
        for i, model in enumerate(metrics["model_selections"]):
            if model == model_name and i in metrics["image_assignments"]:
                image = metrics["image_assignments"][i]
                mapped_model = MODEL_NAME_MAPPING.get(model_name, model_name)
                accuracy = IMAGE_ACCURACY[image].get(mapped_model, 0)
                total_acc += accuracy
                count += 1
        
        if count > 0:
            model_avg_accuracy[model_name] = total_acc / count
    
    # Create scatter plot for each model
    x_values = []
    y_values = []
    labels = []
    sizes = []  # Proportional to percentage of usage
    
    # Calculate total models used
    total_model_usage = sum(metrics["model_distribution"].values())
    
    for model_name, avg_acc in model_avg_accuracy.items():
        # Use model percentage as size
        count = metrics["model_distribution"].get(model_name, 0)
        percentage = (count / total_model_usage * 100) if total_model_usage > 0 else 0
        
        if count > 0:
            x_values.append(avg_efficiency)
            y_values.append(avg_acc)
            labels.append(model_name)
            sizes.append(percentage * 10)  # Scale for visibility
    
    plt.scatter(x_values, y_values, s=sizes)
    
    for i, model in enumerate(labels):
        plt.annotate(f"{model} ({sizes[i]/10:.1f}%)", (x_values[i], y_values[i]))
    
    plt.title(f"{scenario_name}: Accuracy vs Efficiency")
    plt.xlabel("Efficiency (W/FPS)")
    plt.ylabel("Average Accuracy (%)")
    plt.grid(True)
    
    plt.tight_layout()
    filename = f"log-images/{timestamp}-{version}/{scenario_name.lower().replace(' ', '_').replace(':', '')}_accuracy_3.png"
    plt.savefig(filename)
    plt.close()

def plot_comparative_accuracy(all_metrics):
    """Generate plots comparing accuracy across scenarios with percentages for distributions"""
    plt.figure(figsize=(14, 10))
    scenarioMap = {"Scenario 1": "S1", "Scenario 2": "S2", "Scenario 3": "S3", "Scenario 4": "S4", "Scenario 5": "S5"}
    # Plot 1: Average accuracy by scenario
    plt.subplot(1, 2, 1)
    # scenario_names = list(all_metrics.keys())
    avg_accuracies = [metrics.get("accuracy", {}).get("avg_accuracy", 0) 
                     for metrics in all_metrics.values()]
    print(scenarioMap.values())
    plt.bar(scenarioMap.values(), avg_accuracies)
    plt.title("Average Accuracy by Scenario")
    plt.xlabel("Scenario")
    plt.ylabel("Average Accuracy (%)")
    plt.grid(True, axis='y')
    
    # Plot 2: Accuracy by image across scenarios
    plt.subplot(1, 2, 2)
    # Get list of all images
    all_images = set()
    for metrics in all_metrics.values():
        if "accuracy" in metrics:
            all_images.update(metrics["accuracy"].get("avg_image_accuracy", {}).keys())
    
    all_images = sorted(list(all_images))
    image_positions = np.arange(len(all_images))
    bar_width = 0.15
    
    for i, (scenario_name, metrics) in enumerate(all_metrics.items()):
        if "accuracy" in metrics:
            image_accuracies = [metrics["accuracy"]["avg_image_accuracy"].get(img, 0) for img in all_images]
            plt.bar(image_positions + i*bar_width, image_accuracies, bar_width, label=scenarioMap[scenario_name])
    
    plt.title("Accuracy by Image Across Scenarios")
    plt.xlabel("Image")
    plt.ylabel("Average Accuracy (%)")
    plt.xticks(image_positions + bar_width*2, all_images)
    plt.legend()
    plt.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig(f"log-images/{timestamp}-{version}/comparative_accuracy_1.png")
    
    plt.figure(figsize=(14, 10))
    # Plot 3: Model distribution across scenarios (as percentages)
    plt.subplot(1, 2, 1)
    # Get list of all models
    all_models = set()
    for metrics in all_metrics.values():
        all_models.update(metrics.get("model_distribution", {}).keys())
    
    all_models = sorted(list(all_models))
    model_positions = np.arange(len(all_models))
    
    for i, (scenario_name, metrics) in enumerate(all_metrics.items()):
        if "model_distribution" in metrics:
            # Get counts
            model_counts = [metrics["model_distribution"].get(model, 0) for model in all_models]
            
            # Convert to percentages
            total_count = sum(model_counts)
            if total_count > 0:
                model_percentages = [count / total_count * 100 for count in model_counts]
            else:
                model_percentages = [0] * len(model_counts)
                
            plt.bar(model_positions + i*bar_width, model_percentages, bar_width, label=scenarioMap[scenario_name])
    
    plt.title("Model Distribution Across Scenarios")
    plt.xlabel("Model")
    plt.ylabel("Request Percentage (%)")
    plt.xticks(model_positions + bar_width*2, all_models)
    plt.legend()
    plt.grid(True, axis='y')
    
    # Plot 4: Accuracy vs Efficiency trade-off
    plt.subplot(1, 2, 2)
    
    x_values = []  # Efficiency
    y_values = []  # Accuracy
    s_values = []  # Success rate (size)
    
    for scenario_name, metrics in all_metrics.items():
        if "efficiency" in metrics and "accuracy" in metrics:
            # Efficiency
            avg_efficiency = sum(metrics["efficiency"]) / len(metrics["efficiency"]) / 1000  # W/FPS
            
            # Accuracy
            avg_accuracy = metrics["accuracy"]["avg_accuracy"]
            
            # Success rate (for point size)
            total_requests = metrics["allocation_success"] + metrics["requests_queued"]
            success_rate = (metrics["allocation_success"] / total_requests * 100) if total_requests > 0 else 0
            
            x_values.append(avg_efficiency)
            y_values.append(avg_accuracy)
            s_values.append(success_rate * 10)  # Scale for visibility
    
    plt.scatter(x_values, y_values, s=s_values)
    
    for i, scenario_name in enumerate(all_metrics.keys()):
        plt.annotate(f"{scenarioMap[scenario_name]} ({s_values[i]/10:.1f}%)", (x_values[i], y_values[i]))
    
    plt.title("Accuracy vs Efficiency Trade-off")
    plt.xlabel("Efficiency (W/FPS)")
    plt.ylabel("Average Accuracy (%)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"log-images/{timestamp}-{version}/comparative_accuracy_2.png")
    plt.close()

# Reference heatmap showing the accuracy of each model for each image
def plot_model_image_accuracy_matrix():
    """Generate a heatmap showing the accuracy of each model for each image"""
    plt.figure(figsize=(12, 8))
    
    # Prepare data for heatmap
    images = list(IMAGE_ACCURACY.keys())
    models = list(MODEL_NAME_MAPPING.values())
    
    # Create accuracy matrix
    accuracy_matrix = np.zeros((len(images), len(models)))
    for i, image in enumerate(images):
        for j, model in enumerate(models):
            accuracy_matrix[i, j] = IMAGE_ACCURACY[image].get(model, 0)
    
    plt.imshow(accuracy_matrix, cmap='viridis')
    plt.colorbar(label='Accuracy (%)')
    plt.title('Model Accuracy by Image (Reference)')
    plt.xlabel('Model')
    plt.ylabel('Image')
    plt.xticks(np.arange(len(models)), [REVERSE_MODEL_MAPPING.get(m, m) for m in models])
    plt.yticks(np.arange(len(images)), images)
    
    # Add text labels to the heatmap cells
    for i in range(len(images)):
        for j in range(len(models)):
            plt.text(j, i, f"{accuracy_matrix[i, j]:.1f}%", 
                    ha="center", va="center", color="white" if accuracy_matrix[i, j] < 50 else "black")
    
    plt.tight_layout()
    plt.savefig(f"log-images/{timestamp}-{version}/model_image_accuracy_matrix.png")
    plt.close()

# Plot functions for all scenarios
def plot_scenario_results(metrics, scenario_name):
    """Plot scenario results with percentages instead of raw counts for distributions"""
    # Check if metrics is None
    if metrics is None:
        print(f"Cannot plot {scenario_name}: metrics is None")
        return
        
    # Standard performance metrics (unchanged)
    plt.figure(figsize=(15, 12))
    
    # 1. Power consumption by device over time
    plt.subplot(1, 2, 1)
    for device_name, power_values in metrics["power_consumption"].items():
        # Convert mW to W for visualization
        power_values_w = [p/1000 for p in power_values]
        plt.plot(metrics["timestamps"], power_values_w, label=device_name)
    
    # Add overall trend line for total power
    total_power_values = [p/1000 for p in metrics["total_power"]]  # Convert to W
    add_trend_line(metrics["timestamps"], total_power_values)
    
    plt.title(f"{scenario_name}: Power Consumption by Device")
    plt.xlabel("Time")
    plt.ylabel("Power (W)")
    plt.legend()
    plt.grid(True)
    
    # 2. FPS allocation by device over time
    plt.subplot(1, 2, 2)
    for device_name, fps_values in metrics["fps_allocation"].items():
        plt.plot(metrics["timestamps"], fps_values, label=device_name)
    
    # Add overall trend line for total FPS
    total_fps_values = metrics["total_fps"]
    add_trend_line(metrics["timestamps"], total_fps_values)
    
    plt.title(f"{scenario_name}: FPS Allocation by Device")
    plt.xlabel("Time")
    plt.ylabel("Total FPS")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    filename = f"log-images/{timestamp}-{version}/{scenario_name.lower().replace(' ', '_').replace(':', '')}_results_1.png"
    plt.savefig(filename)

    plt.figure(figsize=(14, 10))
    # 3. Total power vs total FPS
    plt.subplot(1, 2, 1)
    # Convert mW to W for visualization
    total_power_w = [p/1000 for p in metrics["total_power"]]
    plt.plot(metrics["timestamps"], total_power_w, label="Total Power (W)")
    plt.plot(metrics["timestamps"], metrics["total_fps"], label="Total FPS")
    
    # Add trend lines for both series
    add_trend_line(metrics["timestamps"], total_power_w)
    add_trend_line(metrics["timestamps"], metrics["total_fps"])
    
    plt.title(f"{scenario_name}: Total System Power vs FPS")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    
    # 4. System efficiency (power per FPS)
    plt.subplot(1, 2, 2)
    # Convert efficiency from mW/FPS to W/FPS
    efficiency_w = [e/1000 for e in metrics["efficiency"]]
    plt.plot(metrics["timestamps"], efficiency_w)
    
    # Add trend line for efficiency
    add_trend_line(metrics["timestamps"], efficiency_w)
    
    plt.title(f"{scenario_name}: System Efficiency (Power per FPS)")
    plt.xlabel("Time")
    plt.ylabel("W/FPS")
    plt.grid(True)
    
    plt.tight_layout()
    filename = f"log-images/{timestamp}-{version}/{scenario_name.lower().replace(' ', '_').replace(':', '')}_results_2.png"
    plt.savefig(filename)
    plt.close()
    
    # Additional plots for queue sizes and utilization
    plt.figure(figsize=(15, 6))
    
    # 1. Queue sizes
    plt.subplot(1, 2, 1)
    for device_name, queue_values in metrics["queue_sizes"].items():
        plt.plot(metrics["timestamps"], queue_values, label=device_name)
    
    # Calculate average queue size across all devices
    avg_queue_size = [sum(q) / len(q) if len(q) > 0 else 0 for q in zip(*metrics["queue_sizes"].values())]
    # add_trend_line(metrics["timestamps"], avg_queue_size)
    
    plt.title(f"{scenario_name}: Request Queue Size by Device")
    plt.xlabel("Time")
    plt.ylabel("Queue Size")
    plt.legend()
    plt.grid(True)
    
    # 2. Device utilization
    plt.subplot(1, 2, 2)
    for device_name, util_values in metrics["utilization"].items():
        plt.plot(metrics["timestamps"], util_values, label=device_name)
    
    # Calculate average utilization across all devices
    avg_utilization = [sum(u) / len(u) if len(u) > 0 else 0 for u in zip(*metrics["utilization"].values())]
    add_trend_line(metrics["timestamps"], avg_utilization)
    
    plt.title(f"{scenario_name}: Device Utilization (%)")
    plt.xlabel("Time")
    plt.ylabel("Utilization (%)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    filename = f"log-images/{timestamp}-{version}/{scenario_name.lower().replace(' ', '_').replace(':', '')}_queues_utilization.png"
    plt.savefig(filename)
    plt.close()
    
    # Plot model distribution as percentages
    plt.figure(figsize=(10, 6))
    model_names = list(metrics["model_distribution"].keys())
    model_counts = list(metrics["model_distribution"].values())
    
    # Convert counts to percentages
    total_model_allocations = sum(model_counts)
    model_percentages = [count / total_model_allocations * 100 if total_model_allocations > 0 else 0 
                        for count in model_counts]
    
    plt.bar(model_names, model_percentages)
    
    # Add a horizontal red dotted line for the average model percentage (should be 100%/num_models with uses)
    # used_models = [p for p in model_percentages if p > 0]
    # if used_models:
    #     avg_model_percentage = 100 / len(used_models)
    #     plt.axhline(y=avg_model_percentage, color='r', linestyle='--', linewidth=2)
    
    plt.title(f"{scenario_name}: Model Distribution in Requests")
    plt.xlabel("Model")
    plt.ylabel("Request Percentage (%)")
    plt.ylim(0, max(model_percentages) * 1.1)  # Add 10% margin at the top
    filename = f"log-images/{timestamp}-{version}/{scenario_name.lower().replace(' ', '_').replace(':', '')}_model_distribution.png"
    plt.savefig(filename)
    plt.close()
    
    # Plot device distribution as percentages
    plt.figure(figsize=(10, 6))
    device_names = list(metrics["device_distribution"].keys())
    device_counts = list(metrics["device_distribution"].values())
    
    # Convert counts to percentages
    total_device_allocations = sum(device_counts)
    device_percentages = [count / total_device_allocations * 100 if total_device_allocations > 0 else 0 
                         for count in device_counts]
    
    plt.bar(device_names, device_percentages)
    
    # Add a horizontal red dotted line for the average device percentage
    # used_devices = [p for p in device_percentages if p > 0]
    # if used_devices:
    #     avg_device_percentage = 100 / len(used_devices)
    #     plt.axhline(y=avg_device_percentage, color='r', linestyle='--', linewidth=2)
    
    plt.title(f"{scenario_name}: Device Distribution in Requests")
    plt.xlabel("Device")
    plt.ylabel("Request Percentage (%)")
    plt.ylim(0, max(device_percentages) * 1.1)  # Add 10% margin at the top
    # plt.xticks(rotation=45)
    plt.tight_layout()
    filename = f"log-images/{timestamp}-{version}/{scenario_name.lower().replace(' ', '_').replace(':', '')}_device_distribution.png"
    plt.savefig(filename)
    plt.close()


# Scenario 4: Accuracy-optimized scheduler
def run_scenario_accuracy_optimized():
    """
    Run a scenario where the scheduler heavily prioritizes accuracy over power efficiency.
    The scheduler will always choose the most accurate model for each image.
    """
    logging.info("\nRunning Scenario 4: Accuracy-Optimized Scheduler")
    print("\nRunning Scenario 4: Accuracy-Optimized Scheduler")
    print("This scheduler prioritizes model accuracy over power efficiency")
    
    # Initialize devices and scheduler
    local_devices = reset_devices()
    scheduler = AHPQueueScheduler(local_devices, models)
    
    # Gaussian-distributed FPS values
    fps_values = [1, 8, 23, 54, 110, 199, 318, 461, 621, 777, 915, 1026, 1109, 1170, 1217, 1254, 1284, 1310, 1334, 1360, 1391, 1437, 1500, 1437, 1391, 1360, 1334, 1310, 1284, 1254, 1217, 1170, 1109, 1026, 915, 777, 621, 461, 381, 199, 110, 54, 23, 8, 1]
    
    # Get list of images
    images = list(IMAGE_ACCURACY.keys())
    
    # Assign images to requests in round-robin fashion
    image_assignments = {}
    for i in range(len(fps_values)):
        image_assignments[i] = images[i % len(images)]
    
    # Create proper request objects with durations
    requests = []
    for i, fps in enumerate(fps_values):
        image = image_assignments[i]
        
        # Find the best model (highest accuracy) for this image
        best_model_name = None
        best_accuracy = -1
        
        for model_name, mapped_name in MODEL_NAME_MAPPING.items():
            if mapped_name in IMAGE_ACCURACY[image]:
                accuracy = IMAGE_ACCURACY[image][mapped_name]
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model_name = model_name
        
        # If no model found, use the default selection method
        if best_model_name is None:
            best_model_name = None  # Let scheduler decide
        
        requests.append({
            "id": i,
            "fps_request": fps,
            "duration": 1.0,  # Fixed duration of 1 second for each request
            "model_name": best_model_name,  # Pre-selected best accuracy model
            "image": image,
            "expected_accuracy": best_accuracy
        })
    
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
        "overload_events": [],
        "model_selections": [],  # Track the sequence of selected models
        "image_assignments": image_assignments  # Track which image was assigned to each request
    }
    
    # Tracking active requests
    active_requests = {}  # Maps request_id -> (completion_time, device, model, fps, image)
    
    # Simulation time
    current_time = 0
    
    for i, request in enumerate(requests):
        fps_request = request["fps_request"]
        model_name = request["model_name"]
        image = request["image"]
        expected_accuracy = request.get("expected_accuracy", 0)
        
        logging.info(f"\nProcessing iteration {i+1}/{len(requests)} with FPS {fps_request}, Model: {model_name}, Image: {image}")
        logging.info(f"Expected accuracy: {expected_accuracy:.2f}%")
        print(f"Processing iteration {i+1}/{len(requests)} with FPS {fps_request}, Image: {image}")
        print(f"Best model for this image: {model_name} (Expected accuracy: {expected_accuracy:.2f}%)")
        
        # Check if the model can handle the FPS requirement
        model_obj = next((m for m in models if m.name == model_name), None)
        if model_obj and fps_request > model_obj.max_fps:
            logging.info(f"Warning: Best accuracy model {model_name} can't handle {fps_request} FPS (max: {model_obj.max_fps})")
            print(f"Warning: Best accuracy model {model_name} can't handle {fps_request} FPS (max: {model_obj.max_fps})")
            print(f"Will try to use it at reduced FPS to maintain accuracy")
        
        # Update time
        current_time += 1  # 1 second between each iteration
        
        # Process completed requests
        completed_requests = []
        for req_id, (completion_time, device_name, req_model_name, fps, req_image) in list(active_requests.items()):
            if current_time >= completion_time:
                # Release resources
                success, message = scheduler.release_request(req_model_name, fps, device_name)
                completed_requests.append(req_id)
                logging.info(f"Released request {req_id}: {message}")
        
        # Remove completed requests
        for req_id in completed_requests:
            if req_id in active_requests:
                del active_requests[req_id]
        
        # Process the new request - Use accuracy-optimized model with scheduler deciding device
        success, message = scheduler.handle_request(model_name=model_name, required_fps=fps_request, image=image)
        
        if success:
            # Parse the allocation result
            if "queued" in message:
                metrics["requests_queued"] += 1
                logging.info(f"Request queued: {message}")
            else:
                # Find allocated FPS amount from the message
                try:
                    # Extract information from the most recent request in history
                    latest_req = scheduler.request_history[-1]
                    
                    if "action" in latest_req and latest_req["action"] == "allocate_request":
                        allocated_fps = latest_req["allocated_fps"]
                        device_name = latest_req["device"]
                        
                        # Track model selection
                        metrics["model_selections"].append(model_name)
                        
                        # Schedule completion
                        completion_time = current_time + request["duration"]
                        active_requests[f"req-{i}"] = (completion_time, device_name, model_name, allocated_fps, image)
                        
                        metrics["allocation_success"] += 1
                        metrics["model_distribution"][model_name] += 1
                        metrics["device_distribution"][device_name] += 1
                        
                        # Get actual accuracy for this model and image
                        mapped_model = MODEL_NAME_MAPPING.get(model_name, model_name)
                        accuracy = IMAGE_ACCURACY[image].get(mapped_model, 0)
                        
                        logging.info(f"Request scheduled to complete at time {completion_time:.1f}")
                        logging.info(f"Allocated {allocated_fps:.1f}/{fps_request:.1f} FPS of {model_name} to {device_name}")
                        logging.info(f"Accuracy for {image} using {model_name}: {accuracy:.2f}%")
                        print(f"Allocated {allocated_fps:.1f}/{fps_request:.1f} FPS of {model_name} to {device_name}")
                        print(f"Accuracy for {image} using {model_name}: {accuracy:.2f}%")
                        
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
                                    logging.info(f"Device {device_name} overloaded, recovery until {recovery_time-current_time:.1f} seconds from now")
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
            
        # Wait for 0.1 seconds after each iteration for simulation speed
        time.sleep(0.1)
    
    # Calculate accuracy metrics
    accuracy_metrics = calculate_accuracy_metrics(metrics["model_selections"], metrics["image_assignments"])
    metrics["accuracy"] = accuracy_metrics
    
    logging.info(f"Completed {metrics['allocation_success']} allocations, {metrics['requests_queued']} queued")
    logging.info(f"Average accuracy: {accuracy_metrics['avg_accuracy']:.2f}%")
    
    print(f"\nCompleted {metrics['allocation_success']} allocations, {metrics['requests_queued']} queued")
    print(f"Average accuracy: {accuracy_metrics['avg_accuracy']:.2f}%")
    
    # Print accuracy breakdown by image
    print("\nAccuracy by image:")
    for image, avg_acc in accuracy_metrics['avg_image_accuracy'].items():
        print(f"  {image}: {avg_acc:.2f}%")
    
    # Calculate power statistics
    avg_power = sum(metrics["total_power"]) / len(metrics["total_power"]) / 1000  # Convert to W
    avg_efficiency = sum(metrics["efficiency"]) / len(metrics["efficiency"]) / 1000  # Convert to W/FPS
    
    print(f"\nPower statistics:")
    print(f"  Average power consumption: {avg_power:.2f} W")
    print(f"  Average efficiency: {avg_efficiency:.4f} W/FPS")
    
    return metrics

# Scenario 5: Power-efficiency-optimized scheduler
def run_scenario_power_optimized():
    """
    Run a scenario where the scheduler heavily prioritizes power efficiency over accuracy.
    The scheduler will select models that minimize power consumption while meeting the FPS requirements.
    """
    logging.info("\nRunning Scenario 5: Power-Efficiency-Optimized Scheduler")
    print("\nRunning Scenario 5: Power-Efficiency-Optimized Scheduler")
    print("This scheduler prioritizes power efficiency over model accuracy")
    
    # Initialize devices and scheduler
    local_devices = reset_devices()
    scheduler = AHPQueueScheduler(local_devices, models)
    
    # Gaussian-distributed FPS values
    fps_values = [1, 8, 23, 54, 110, 199, 318, 461, 621, 777, 915, 1026, 1109, 1170, 1217, 1254, 1284, 1310, 1334, 1360, 1391, 1437, 1500, 1437, 1391, 1360, 1334, 1310, 1284, 1254, 1217, 1170, 1109, 1026, 915, 777, 621, 461, 381, 199, 110, 54, 23, 8, 1]
    
    # Get list of images
    images = list(IMAGE_ACCURACY.keys())
    
    # Assign images to requests in round-robin fashion
    image_assignments = {}
    for i in range(len(fps_values)):
        image_assignments[i] = images[i % len(images)]
    
    # Find the most power-efficient model for various FPS ranges
    # We'll pre-calculate this to simulate a power-aware scheduler's knowledge
    power_efficient_models = {}
    fps_ranges = [(0, 50), (51, 200), (201, 500), (501, 1000), (1001, float('inf'))]
    
    for fps_min, fps_max in fps_ranges:
        best_model = None
        best_efficiency = 0
        
        for model in models:
            # Calculate average power efficiency across all devices
            avg_efficiency = 0
            count = 0
            
            for device in local_devices:
                if model.name in device.max_fps and device.max_fps[model.name] >= fps_min:
                    # Calculate power at the middle of the range
                    test_fps = min(device.max_fps[model.name], (fps_min + fps_max) / 2)
                    if test_fps > 0:
                        power = calculate_power(test_fps, device.power_coefficients[model.name])
                        if power > 0:
                            efficiency = test_fps / power  # FPS per watt
                            avg_efficiency += efficiency
                            count += 1
            
            if count > 0:
                avg_efficiency /= count
                if best_model is None or avg_efficiency > best_efficiency:
                    best_model = model
                    best_efficiency = avg_efficiency
        
        # If we found a model for this range, store it
        if best_model:
            power_efficient_models[(fps_min, fps_max)] = best_model.name
    
    # Create proper request objects with durations
    requests = []
    for i, fps in enumerate(fps_values):
        image = image_assignments[i]
        
        # Find the most power-efficient model for this FPS range
        selected_model = None
        for (fps_min, fps_max), model_name in power_efficient_models.items():
            if fps_min <= fps <= fps_max:
                selected_model = model_name
                break
        
        # If no model found, use the default model
        if selected_model is None:
            selected_model = None  # Let scheduler decide
        
        requests.append({
            "id": i,
            "fps_request": fps,
            "duration": 1.0,  # Fixed duration of 1 second for each request
            "model_name": selected_model,  # Pre-selected power-efficient model
            "image": image
        })
    
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
        "overload_events": [],
        "model_selections": [],  # Track the sequence of selected models
        "image_assignments": image_assignments  # Track which image was assigned to each request
    }
    
    # Tracking active requests
    active_requests = {}  # Maps request_id -> (completion_time, device, model, fps, image)
    
    # Simulation time
    current_time = 0
    
    for i, request in enumerate(requests):
        fps_request = request["fps_request"]
        model_name = request["model_name"]
        image = request["image"]
        
        logging.info(f"\nProcessing iteration {i+1}/{len(requests)} with FPS {fps_request}, Model: {model_name}, Image: {image}")
        print(f"Processing iteration {i+1}/{len(requests)} with FPS {fps_request}, Image: {image}")
        print(f"Selected power-efficient model: {model_name}")
        
        # Update time
        current_time += 1  # 1 second between each iteration
        
        # Process completed requests
        completed_requests = []
        for req_id, (completion_time, device_name, req_model_name, fps, req_image) in list(active_requests.items()):
            if current_time >= completion_time:
                # Release resources
                success, message = scheduler.release_request(req_model_name, fps, device_name)
                completed_requests.append(req_id)
                logging.info(f"Released request {req_id}: {message}")
        
        # Remove completed requests
        for req_id in completed_requests:
            if req_id in active_requests:
                del active_requests[req_id]
        
        # Process the new request - Use power-efficiency-optimized model
        # For power-optimized scheduler, we also need to select the most power-efficient device
        # We'll implement a custom device selection that prioritizes efficiency
        
        success = False
        message = ""
        
        # If model_name is None, select most power-efficient model for this FPS
        if model_name is None:
            # Find most power-efficient model that can handle this FPS
            eligible_models = [model for model in models if model.max_fps >= fps_request]
            if eligible_models:
                best_efficiency = 0
                best_model = None
                
                for model in eligible_models:
                    avg_efficiency = 0
                    count = 0
                    
                    for device in local_devices:
                        if model.name in device.max_fps and device.max_fps[model.name] >= fps_request:
                            # Calculate power at requested FPS
                            power = calculate_power(fps_request, device.power_coefficients[model.name])
                            if power > 0:
                                efficiency = fps_request / power  # FPS per watt
                                avg_efficiency += efficiency
                                count += 1
                    
                    if count > 0:
                        avg_efficiency /= count
                        if best_model is None or avg_efficiency > best_efficiency:
                            best_model = model
                            best_efficiency = avg_efficiency
                
                if best_model:
                    model_name = best_model.name
                else:
                    # Fall back to default model selection
                    selected_model = scheduler.select_model_using_ahp(fps_request, image)
                    model_name = selected_model.name
            else:
                # Fall back to highest max_fps model
                highest_fps_model = max(models, key=lambda m: m.max_fps)
                model_name = highest_fps_model.name
        
        # Track the selected model
        metrics["model_selections"].append(model_name)
        
        # Now select most power-efficient device for this model and FPS
        best_device = None
        best_efficiency = 0
        allocatable_fps = 0
        
        for device in local_devices:
            if model_name in device.max_fps and not scheduler.device_cooldown[device.name]["overloaded"]:
                # Calculate available FPS
                available_fps = device.max_fps[model_name] - device.current_fps[model_name]
                fps_to_allocate = min(fps_request, available_fps)
                
                if fps_to_allocate > 0:
                    # Calculate power increase
                    power_before = device.current_power
                    original_fps = device.current_fps[model_name]
                    
                    device.current_fps[model_name] += fps_to_allocate
                    update_device_power(device)
                    power_after = device.current_power
                    
                    # Reset device state
                    device.current_fps[model_name] = original_fps
                    update_device_power(device)
                    
                    # Calculate efficiency
                    power_increase = power_after - power_before
                    if power_increase > 0:
                        efficiency = fps_to_allocate / power_increase
                        if best_device is None or efficiency > best_efficiency:
                            best_device = device
                            best_efficiency = efficiency
                            allocatable_fps = fps_to_allocate
        
        # If we found a device, allocate the request
        if best_device:
            # Calculate power before allocation
            power_before = best_device.current_power
            
            # Allocate resources on best device
            best_device.current_fps[model_name] += allocatable_fps
            update_device_power(best_device)
            
            # Calculate power increase
            power_increase = best_device.current_power - power_before
            
            # Check if this will overload the device
            utilization = sum(best_device.current_fps.values()) / sum(best_device.max_fps.values())
            will_overload = utilization > 0.9
            
            if will_overload:
                # Mark as overloaded
                recovery_time = current_time + random.uniform(2.0, 4.0)
                scheduler.device_cooldown[best_device.name]["overloaded"] = True
                scheduler.device_cooldown[best_device.name]["recovery_until"] = recovery_time
                scheduler.device_cooldown[best_device.name]["workload_before_overload"] = best_device.current_fps.copy()
                
                metrics["overload_events"].append({
                    "time": current_time,
                    "device": best_device.name,
                    "recovery_time": recovery_time
                })
            
            # Get actual accuracy for this model and image
            mapped_model = MODEL_NAME_MAPPING.get(model_name, model_name)
            accuracy = IMAGE_ACCURACY[image].get(mapped_model, 0)
            
            # Schedule completion
            completion_time = current_time + request["duration"]
            active_requests[f"req-{i}"] = (completion_time, best_device.name, model_name, allocatable_fps, image)
            
            # Update metrics
            metrics["allocation_success"] += 1
            metrics["model_distribution"][model_name] += 1
            metrics["device_distribution"][best_device.name] += 1
            
            # Add to request history for consistency
            scheduler.request_history.append({
                "timestamp": len(scheduler.request_history),
                "action": "allocate_request",
                "model": model_name,
                "requested_fps": fps_request,
                "allocated_fps": allocatable_fps,
                "device": best_device.name,
                "power_increase": power_increase,
                "total_device_power": best_device.current_power,
                "overloaded": will_overload
            })
            
            logging.info(f"Request scheduled to complete at time {completion_time:.1f}")
            logging.info(f"Allocated {allocatable_fps:.1f}/{fps_request:.1f} FPS of {model_name} to {best_device.name}")
            logging.info(f"Accuracy for {image} using {model_name}: {accuracy:.2f}%")
            print(f"Allocated {allocatable_fps:.1f}/{fps_request:.1f} FPS of {model_name} to {best_device.name}")
            print(f"Accuracy for {image} using {model_name}: {accuracy:.2f}%")
            print(f"Efficiency: {best_efficiency:.2f} FPS/W")
            
            success = True
        else:
            # Fall back to standard scheduler for this request
            success, message = scheduler.handle_request(model_name=model_name, required_fps=fps_request, image=image)
        
        # Process standard scheduler result if used
        if not best_device and success:
            # Parse the allocation result
            if "queued" in message:
                metrics["requests_queued"] += 1
                logging.info(f"Request queued: {message}")
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
                        active_requests[f"req-{i}"] = (completion_time, device_name, model_name, allocated_fps, image)
                        
                        metrics["allocation_success"] += 1
                        metrics["model_distribution"][model_name] += 1
                        metrics["device_distribution"][device_name] += 1
                        
                        # Get actual accuracy for this model and image
                        mapped_model = MODEL_NAME_MAPPING.get(model_name, model_name)
                        accuracy = IMAGE_ACCURACY[image].get(mapped_model, 0)
                        
                        logging.info(f"Request scheduled to complete at time {completion_time:.1f}")
                        logging.info(f"Allocated {allocated_fps:.1f}/{fps_request:.1f} FPS of {model_name} to {device_name}")
                        logging.info(f"Accuracy for {image} using {model_name}: {accuracy:.2f}%")
                        print(f"Allocated {allocated_fps:.1f}/{fps_request:.1f} FPS of {model_name} to {device_name}")
                        print(f"Accuracy for {image} using {model_name}: {accuracy:.2f}%")
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
            
        # Wait for 0.1 seconds after each iteration for simulation speed
        time.sleep(0.1)
    
    # Calculate accuracy metrics
    accuracy_metrics = calculate_accuracy_metrics(metrics["model_selections"], metrics["image_assignments"])
    metrics["accuracy"] = accuracy_metrics
    
    logging.info(f"Completed {metrics['allocation_success']} allocations, {metrics['requests_queued']} queued")
    logging.info(f"Average accuracy: {accuracy_metrics['avg_accuracy']:.2f}%")
    
    print(f"\nCompleted {metrics['allocation_success']} allocations, {metrics['requests_queued']} queued")
    print(f"Average accuracy: {accuracy_metrics['avg_accuracy']:.2f}%")
    
    # Calculate power statistics
    avg_power = sum(metrics["total_power"]) / len(metrics["total_power"]) / 1000  # Convert to W
    avg_efficiency = sum(metrics["efficiency"]) / len(metrics["efficiency"]) / 1000  # Convert to W/FPS
    
    print(f"\nPower statistics:")
    print(f"  Average power consumption: {avg_power:.2f} W")
    print(f"  Average efficiency: {avg_efficiency:.4f} W/FPS")
    
    return metrics

# Create comparative plots for all scenarios
def create_comparative_plots(all_metrics):
    """Generate comparative plots with percentages instead of raw counts"""
    # Check if there are any metrics to plot
    if not all_metrics:
        logging.error("No metrics available for comparative plots")
        print("No metrics available for comparative plots")
        return
    scenarioMap = {"Scenario 1": "S1", "Scenario 2": "S2", "Scenario 3": "S3", "Scenario 4": "S4", "Scenario 5": "S5"}    
    # 1. Compare power consumption over time
    plt.figure(figsize=(14, 10))
    
    # Total power consumption
    plt.subplot(1, 2, 1)
    for scenario_name, metrics in all_metrics.items():
        # Convert mW to W for visualization
        power_values_w = [p/1000 for p in metrics["total_power"]]
        plt.plot(metrics["timestamps"], power_values_w, label=scenarioMap[scenario_name])
    
    # Calculate average power consumption across all scenarios (in W)
    avg_power = []
    for i in range(len(next(iter(all_metrics.values()))["timestamps"])):
        scenario_power = []
        for metrics in all_metrics.values():
            if i < len(metrics["total_power"]):
                scenario_power.append(metrics["total_power"][i] / 1000)  # Convert to W
        avg_power.append(sum(scenario_power) / len(scenario_power) if scenario_power else 0)
    
    # Add trend line for overall average power
    add_trend_line(next(iter(all_metrics.values()))["timestamps"], avg_power)
    
    plt.title("Total Power Consumption Comparison")
    plt.xlabel("Time")
    plt.ylabel("Power (W)")
    plt.legend()
    plt.grid(True)
    
    # Total FPS
    plt.subplot(1, 2, 2)
    for scenario_name, metrics in all_metrics.items():
        plt.plot(metrics["timestamps"], metrics["total_fps"], label=scenarioMap[scenario_name])
    
    # Calculate average FPS across all scenarios
    avg_fps = []
    for i in range(len(next(iter(all_metrics.values()))["timestamps"])):
        scenario_fps = []
        for metrics in all_metrics.values():
            if i < len(metrics["total_fps"]):
                scenario_fps.append(metrics["total_fps"][i])
        avg_fps.append(sum(scenario_fps) / len(scenario_fps) if scenario_fps else 0)
    
    # Add trend line for overall average FPS
    add_trend_line(next(iter(all_metrics.values()))["timestamps"], avg_fps)
    
    plt.title("Total FPS Comparison")
    plt.xlabel("Time")
    plt.ylabel("FPS")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"log-images/{timestamp}-{version}/comparative_analysis_1.png")


    plt.figure(figsize=(14, 10))
    # Efficiency
    plt.subplot(1, 2, 1)
    for scenario_name, metrics in all_metrics.items():
        # Convert efficiency from mW/FPS to W/FPS
        efficiency_w = [e/1000 for e in metrics["efficiency"]]
        plt.plot(metrics["timestamps"], efficiency_w, label=scenarioMap[scenario_name])
    
    # Calculate average efficiency across all scenarios (in W/FPS)
    avg_efficiency = []
    for i in range(len(next(iter(all_metrics.values()))["timestamps"])):
        scenario_efficiency = []
        for metrics in all_metrics.values():
            if i < len(metrics["efficiency"]):
                if metrics["efficiency"][i] != float('inf') and not np.isnan(metrics["efficiency"][i]):
                    scenario_efficiency.append(metrics["efficiency"][i] / 1000)  # Convert to W/FPS
        avg_efficiency.append(sum(scenario_efficiency) / len(scenario_efficiency) if scenario_efficiency else 0)
    
    # Add trend line for overall average efficiency
    add_trend_line(next(iter(all_metrics.values()))["timestamps"], avg_efficiency)
    
    plt.title("Efficiency (Power per FPS) Comparison")
    plt.xlabel("Time")
    plt.ylabel("W/FPS")
    plt.legend()
    plt.grid(True)
    
    # Device distribution as percentages per scenario
    plt.subplot(1, 2, 2)
    bar_width = 0.15  # Reduced width to accommodate more scenarios
    device_names = list(next(iter(all_metrics.values()))["device_distribution"].keys())
    x = np.arange(len(device_names))
    
    # For each scenario, convert device counts to percentages
    scenario_device_percentages = {}
    
    for scenario_name, metrics in all_metrics.items():
        # Get device counts for this scenario
        device_counts = [metrics["device_distribution"].get(device, 0) for device in device_names]
        
        # Convert to percentages
        total_count = sum(device_counts)
        if total_count > 0:
            scenario_device_percentages[scenario_name] = [count / total_count * 100 for count in device_counts]
        else:
            scenario_device_percentages[scenario_name] = [0] * len(device_names)
    
    # Plot device distribution percentages by scenario
    for i, (scenario_name, percentages) in enumerate(scenario_device_percentages.items()):
        offset = (i - (len(scenario_device_percentages) - 1) / 2) * bar_width
        plt.bar(x + offset, percentages, bar_width, label=scenarioMap[scenario_name])
    
    plt.title("Device Distribution Comparison")
    plt.xlabel("Device")
    plt.ylabel("Request Percentage (%)")
    plt.xticks(x, device_names)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"log-images/{timestamp}-{version}/comparative_analysis_2.png")
    plt.close()


# Run all scenarios and compare with accuracy metrics
def run_all_scenarios_with_accuracy():
    """Run all scenarios with accuracy tracking and generate comparative visualizations"""
    # First, generate reference accuracy matrix
    plot_model_image_accuracy_matrix()
    
    # Set up containers for all metrics
    all_metrics = {}
    
    # Run Scenario 1 with accuracy tracking
    print("\n=== Running Scenario 1 with accuracy tracking: Scheduler decides both device and model ===")
    scenario1_metrics = run_scenario_scheduler_decides_all_modified()
    if scenario1_metrics is not None:
        all_metrics["Scenario 1"] = scenario1_metrics
        plot_scenario_results(scenario1_metrics, "Scenario 1: Scheduler Decides Both")
        plot_accuracy_metrics(scenario1_metrics, "Scenario 1: Scheduler Decides Both")
    time.sleep(5)
    # Run Scenario 2 with accuracy tracking
    print("\n=== Running Scenario 2 with accuracy tracking: User decides device, scheduler decides model ===")
    scenario2_metrics = run_scenario_user_decides_device_modified()
    if scenario2_metrics is not None:
        all_metrics["Scenario 2"] = scenario2_metrics
        plot_scenario_results(scenario2_metrics, "Scenario 2: User Decides Device")
        plot_accuracy_metrics(scenario2_metrics, "Scenario 2: User Decides Device")
    time.sleep(5)
    # Run Scenario 3 with accuracy tracking
    print("\n=== Running Scenario 3 with accuracy tracking: User decides model, scheduler decides device ===")
    scenario3_metrics = run_scenario_user_decides_model_modified()
    if scenario3_metrics is not None:
        all_metrics["Scenario 3"] = scenario3_metrics
        plot_scenario_results(scenario3_metrics, "Scenario 3: User Decides Model")
        plot_accuracy_metrics(scenario3_metrics, "Scenario 3: User Decides Model")
    time.sleep(5)
    

    # Run Scenario 4 with accuracy tracking
    print("\n=== Running Scenario 4 with accuracy tracking: Accuracy-Optimized Scheduler ===")
    scenario4_metrics = run_scenario_accuracy_optimized()
    if scenario4_metrics is not None:
        all_metrics["Scenario 4"] = scenario4_metrics
        plot_scenario_results(scenario4_metrics, "Scenario 4: Accuracy-Optimized Scheduler")
        plot_accuracy_metrics(scenario4_metrics, "Scenario 4: Accuracy-Optimized Scheduler")
    time.sleep(5)
    # Run Scenario 5 with power tracking
    print("\n=== Running Scenario 5 with accuracy tracking: Power-Efficiency-Optimized Scheduler ===")
    scenario5_metrics = run_scenario_power_optimized()
    if scenario5_metrics is not None:
        all_metrics["Scenario 5"] = scenario5_metrics
        plot_scenario_results(scenario5_metrics, "Scenario 5: Power-Efficiency-Optimized Scheduler")
        plot_accuracy_metrics(scenario5_metrics, "Scenario 5: Power-Efficiency-Optimized Scheduler")
    
    # Run Scenario 6 with accuracy tracking
    # print("\n=== Running Scenario 6 with accuracy tracking: Adaptive Model Switching ===")
    # scenario6_metrics = run_scenario_adaptive_model_switching()
    # if scenario6_metrics is not None:
    #     all_metrics["Scenario 6"] = scenario6_metrics
    #     plot_scenario_results(scenario6_metrics, "Scenario 6: Adaptive Model Switching")
    #     plot_accuracy_metrics(scenario6_metrics, "Scenario 6: Adaptive Model Switching")
    
    # # Run Scenario 7 with accuracy tracking
    # print("\n=== Running Scenario 7 with accuracy tracking: Cross-Device Model Consistency ===")
    # scenario7_metrics = run_scenario_cross_device_model_consistency()
    # if scenario7_metrics is not None:
    #     all_metrics["Scenario 7"] = scenario7_metrics
    #     plot_scenario_results(scenario7_metrics, "Scenario 7: Cross-Device Model Consistency")
    #     plot_accuracy_metrics(scenario7_metrics, "Scenario 7: Cross-Device Model Consistency")

    # Generate comparative visualization
    if len(all_metrics) > 1:
        create_comparative_plots(all_metrics)
        plot_comparative_accuracy(all_metrics)
    
    return all_metrics

# Add a function to generate a comprehensive summary report
def generate_summary_report(all_metrics):
    """Generate a comprehensive summary report for all scenarios with percentages"""
    print("\n========== SUMMARY REPORT ==========\n")
    print("Comparing all scenarios with accuracy metrics:\n")
    
    # Table header
    print(f"{'Scenario':<20} | {'Success %':<9} | {'Queued %':<9} | {'Avg Power(W)':<15} | {'Avg Eff(W/FPS)':<15} | {'Avg Acc(%)':<15}")
    print("-" * 95)
    
    # Table rows
    for scenario_name, metrics in all_metrics.items():
        # Calculate success and queued percentages
        total_requests = metrics["allocation_success"] + metrics["requests_queued"]
        success_pct = (metrics["allocation_success"] / total_requests * 100) if total_requests > 0 else 0
        queued_pct = (metrics["requests_queued"] / total_requests * 100) if total_requests > 0 else 0
        
        # Calculate average power and efficiency
        avg_power = sum(metrics["total_power"]) / len(metrics["total_power"]) / 1000  # Convert to W
        avg_efficiency = sum(metrics["efficiency"]) / len(metrics["efficiency"]) / 1000  # Convert to W/FPS
        
        # Get accuracy
        avg_accuracy = metrics["accuracy"]["avg_accuracy"]
        
        print(f"{scenario_name:<20} | {success_pct:<9.2f} | {queued_pct:<9.2f} | {avg_power:<15.2f} | {avg_efficiency:<15.4f} | {avg_accuracy:<15.2f}")
    
    print("\n")
    
    # Most accurate scenario
    most_accurate = max(all_metrics.items(), key=lambda x: x[1]["accuracy"]["avg_accuracy"])
    print(f"Most accurate scenario: {most_accurate[0]} ({most_accurate[1]['accuracy']['avg_accuracy']:.2f}%)")
    
    # Most power efficient scenario
    most_efficient = min(all_metrics.items(), key=lambda x: sum(x[1]["efficiency"]) / len(x[1]["efficiency"]))
    avg_eff = sum(most_efficient[1]["efficiency"]) / len(most_efficient[1]["efficiency"]) / 1000
    print(f"Most power efficient scenario: {most_efficient[0]} ({avg_eff:.4f} W/FPS)")
    
    # Best accuracy-efficiency trade-off calculation (unchanged)
    max_acc = max(x[1]["accuracy"]["avg_accuracy"] for x in all_metrics.items())
    min_acc = min(x[1]["accuracy"]["avg_accuracy"] for x in all_metrics.items())
    
    max_eff = max(sum(x[1]["efficiency"]) / len(x[1]["efficiency"]) for x in all_metrics.items())
    min_eff = min(sum(x[1]["efficiency"]) / len(x[1]["efficiency"]) for x in all_metrics.items())
    
    best_tradeoff = None
    best_score = -1
    
    for scenario_name, metrics in all_metrics.items():
        avg_acc = metrics["accuracy"]["avg_accuracy"]
        avg_eff = sum(metrics["efficiency"]) / len(metrics["efficiency"])
        
        # Normalize values to [0, 1]
        norm_acc = (avg_acc - min_acc) / (max_acc - min_acc) if max_acc > min_acc else 0
        norm_eff = (avg_eff - min_eff) / (max_eff - min_eff) if max_eff > min_eff else 0
        
        # Calculate weighted score (higher is better for accuracy, lower is better for efficiency)
        score = 0.7 * norm_acc + 0.3 * (1 - norm_eff)
        
        if score > best_score:
            best_score = score
            best_tradeoff = scenario_name
    
    print(f"Best accuracy-efficiency trade-off: {best_tradeoff}")
    
    # Accuracy breakdown by image
    print("\nAccuracy by image across scenarios:")
    
    # Get all images
    all_images = set()
    for metrics in all_metrics.values():
        all_images.update(metrics["accuracy"]["avg_image_accuracy"].keys())
    
    for image in sorted(all_images):
        print(f"\n  {image}:")
        for scenario_name, metrics in all_metrics.items():
            acc = metrics["accuracy"]["avg_image_accuracy"].get(image, 0)
            print(f"    {scenario_name}: {acc:.2f}%")
    
    # Model distribution (as percentages)
    print("\nModel distribution across scenarios (%):")
    
    # Get all models
    all_models = set()
    for metrics in all_metrics.values():
        all_models.update(metrics["model_distribution"].keys())
    
    for model in sorted(all_models):
        print(f"\n  {model}:")
        for scenario_name, metrics in all_metrics.items():
            # Calculate percentage
            count = metrics["model_distribution"].get(model, 0)
            total_count = sum(metrics["model_distribution"].values())
            percentage = (count / total_count * 100) if total_count > 0 else 0
            
            print(f"    {scenario_name}: {percentage:.2f}%")
    
    print("\n====================================\n")

# Run the main function if the script is executed directly
if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    print("Starting scheduler simulation with accuracy tracking")
    print("Each request will be assigned to an image in round-robin fashion")
    print("Accuracy will be tracked for each model-image pair")
    
    # Run all scenarios with accuracy tracking
    all_metrics = run_all_scenarios_with_accuracy()
    
    # Generate summary report
    generate_summary_report(all_metrics)
    
    print("\nSimulation complete!")
    print("Check the generated plots for visualization of accuracy results")