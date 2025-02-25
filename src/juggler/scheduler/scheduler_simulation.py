import numpy as np
import pandas as pd
from dataclasses import dataclass
import random
import matplotlib.pyplot as plt

# Define the structures to represent our system
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

# Function to calculate power consumption based on FPS
def calculate_power(fps, coefficients):
    a, b, c, d = coefficients
    power = a * (fps ** 3) + b * (fps ** 2) + c * fps + d
    return max(0, power)  # Power should never be negative

# Create sample models
models = [
    Model(name="ResNet50", max_fps=175),
    Model(name="MobileNet", max_fps=1000),
    Model(name="ResNet18", max_fps=425),
    Model(name="ResNext50", max_fps=110),
    Model(name="SqueezeNet", max_fps=950)
]

# Create sample devices with power equations for each model
# These are fictional equations for demonstration
devices = [
    Device(
        name="Device1 (Jetson Orin Nano)", 
        power_coefficients={
            "ResNet50": [6.072131e-04, -1.300612e-01, 4.557199e+01, 4285.86], 
            "MobileNet": [3.521553e-06, -8.418713e-03, 5.996142e+00, 4232.51], # 3.521553e-06x³ + -8.418713e-03x² + 5.996142e+00x + 4232.51
            "ResNet18": [-4.366841e-05, 7.913778e-03, 1.663735e+01, 4308.14], # -4.366841e-05x³ + 7.913778e-03x² + 1.663735e+01x + 4308.14
            "ResNext50": [-6.402410e-04, 7.870599e-02, 5.164135e+01, 4239.72], # -6.402410e-04x³ + 7.870599e-02x² + 5.164135e+01x + 4239.72
            "SqueezeNet": [1.073624e-05, -1.839516e-02, 1.024424e+01, 4160.02] # 1.073624e-05x³ + -1.839516e-02x² + 1.024424e+01x + 4160.02
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
            "ResNet50": [-2.057230e+01, 6.123255e+02, -4.061582e+03, 6333.14], #-2.057230e+01x³ + 6.123255e+02x² + -4.061582e+03x + 6333.14
            "MobileNet": [1.854938e-03, -6.409779e-01, 6.233325e+01, 1911.11], #1.854938e-03x³ + -6.409779e-01x² + 6.233325e+01x + 1911.11
            "ResNet18": [-1.888558e-02, 9.252537e-01, 1.157932e+02, 2803.937], #-1.888558e-02x³ + 9.252537e-01x² + 1.157932e+02x + 2803.94
            "ResNext50": [-1.988446e+02, 4.591937e+03, -2.792636e+04, 26012.155], #-1.988446e+02x³ + 4.591937e+03x² + -2.792636e+04x + 26012.15
            "SqueezeNet": [1.590855e-03, -5.632611e-01, 5.534153e+01, 2407.379] #1.590855e-03x³ + -5.632611e-01x² + 5.534153e+01x + 2407.38
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
        name="Device3 (Raspberry Pi 4 with hailo-8)", 
        power_coefficients={
            "ResNet50": [0, -0.312, 25.71, 4484.83], #-0.312x^2+25.71x+4484.43
            "MobileNet": [0, -0.04, 7.56, 4849.85], #-0.04x^2+7.56x+4849.85
            "ResNet18": [0, 0.0018, 0.15, 5107.62], #0.0018x^2+0.15x+5107.62
            "ResNext50": [0, -0.446, 31.81, 4724.72], #-0.446x^2+31.81x+4724.72
            "SqueezeNet": [0, -0.00071, 0.53, 5069.21] #-0.00071x^2+0.53x+5069.21
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

# Update each device's power consumption
def update_device_power(device):
    power = 0
    for model_name, fps in device.current_fps.items():
        if fps > 0:
            coefficients = device.power_coefficients[model_name]
            power += calculate_power(fps, coefficients)
    device.current_power = power
    return power

# Scheduler class
class Scheduler:
    def __init__(self, devices, models):
        self.devices = devices
        self.models = models
        self.request_history = []  # To track request allocations
        
    def get_optimal_device_for_model(self, model_name, required_fps):
        """Find the most power-efficient device for a given model and FPS requirement"""
        best_device = None
        min_power_increase = float('inf')
        
        for device in self.devices:
            # Check if device can handle the additional FPS
            if device.current_fps[model_name] + required_fps <= device.max_fps[model_name]:
                # Calculate current power
                current_power = update_device_power(device)
                
                # Calculate power with additional FPS
                original_fps = device.current_fps[model_name]
                device.current_fps[model_name] += required_fps
                new_power = update_device_power(device)
                
                # Reset to original state
                device.current_fps[model_name] = original_fps
                update_device_power(device)
                
                # Calculate power increase
                power_increase = new_power - current_power
                
                if power_increase < min_power_increase:
                    min_power_increase = power_increase
                    best_device = device
        
        return best_device, min_power_increase
    
    def handle_request(self, model_name, required_fps):
        """Process an incoming request"""
        if required_fps <= 0:
            
            return False, "Invalid FPS requirement"
        
        # Find the optimal device
        best_device, power_increase = self.get_optimal_device_for_model(model_name, required_fps)
        
        if best_device is None:
            
            return False, "No device can handle this request"
        
        # Allocate the request
        best_device.current_fps[model_name] += required_fps
        update_device_power(best_device)
        
        self.request_history.append({
            "timestamp": len(self.request_history),
            "model": model_name,
            "fps": required_fps,
            "device": best_device.name,
            "power_increase": power_increase,
            "total_device_power": best_device.current_power
        })
        
        return True, f"Allocated {required_fps} FPS of {model_name} to {best_device.name}"
    
    def release_request(self, model_name, fps_to_release, device_name):
        """Release resources from a completed request"""
        for device in self.devices:
            if device.name == device_name:
                if device.current_fps[model_name] >= fps_to_release:
                    device.current_fps[model_name] -= fps_to_release
                    update_device_power(device)
                    return True, f"Released {fps_to_release} FPS of {model_name} from {device.name}"
                else:
                    return False, "Invalid release request: FPS allocation doesn't match"
        
        return False, "Device not found"
    
    def system_status(self):
        """Get the current system status"""
        status = []
        for device in self.devices:
            update_device_power(device)
            status.append({
                "device": device.name,
                "current_power": device.current_power,
                "models": {model: fps for model, fps in device.current_fps.items() if fps > 0}
            })
        return status
    
    def load_balance(self):
        """Attempt to rebalance load across devices for better power efficiency"""
        changes_made = False
        
        # For each device and model with allocation
        for source_device in self.devices:
            for model_name, current_fps in source_device.current_fps.items():
                if current_fps <= 0:
                    continue
                
                # Try moving some workload to other devices
                fps_to_move = current_fps / 2  # Try moving half the workload
                
                if fps_to_move < 1:  # Don't bother with tiny workloads
                    continue
                    
                # Calculate current power
                original_power = update_device_power(source_device)
                
                # Simulate removing this portion of workload
                source_device.current_fps[model_name] -= fps_to_move
                new_source_power = update_device_power(source_device)
                source_power_decrease = original_power - new_source_power
                
                # Find best target device
                best_target = None
                min_power_increase = float('inf')
                
                for target_device in self.devices:
                    if target_device == source_device:
                        continue
                        
                    # Check if target can handle the additional FPS
                    if target_device.current_fps[model_name] + fps_to_move <= target_device.max_fps[model_name]:
                        # Calculate current target power
                        original_target_power = update_device_power(target_device)
                        
                        # Simulate adding workload
                        target_device.current_fps[model_name] += fps_to_move
                        new_target_power = update_device_power(target_device)
                        target_power_increase = new_target_power - original_target_power
                        
                        # Reset target
                        target_device.current_fps[model_name] -= fps_to_move
                        update_device_power(target_device)
                        
                        if target_power_increase < min_power_increase:
                            min_power_increase = target_power_increase
                            best_target = target_device
                
                # Reset source device to original state
                source_device.current_fps[model_name] += fps_to_move
                update_device_power(source_device)
                
                # If moving makes sense (saves power overall), do it
                if best_target is not None and min_power_increase < source_power_decrease:
                    source_device.current_fps[model_name] -= fps_to_move
                    best_target.current_fps[model_name] += fps_to_move
                    update_device_power(source_device)
                    update_device_power(best_target)
                    
                    self.request_history.append({
                        "timestamp": len(self.request_history),
                        "action": "load_balance",
                        "model": model_name,
                        "fps": fps_to_move,
                        "from_device": source_device.name,
                        "to_device": best_target.name,
                        "power_saving": source_power_decrease - min_power_increase
                    })
                    
                    changes_made = True
        
        return changes_made
    
    def simulate_random_workload(self, num_requests=100, min_fps=5, max_fps=1000):
        """Simulate a random workload of requests"""
        for i in range(num_requests):
            if i % 100 == 0:
                print(f"Processing request {i}...")
            # Choose a random model
            model_name = random.choice([model.name for model in self.models])
            
            # Generate random FPS requirement
            required_fps = random.randint(min_fps, max_fps)
            
            # Handle the request
            success, message = self.handle_request(model_name, required_fps)
            
            # Occasionally release some previous requests
            if i > 10 and random.random() < 0.3:
                # Pick a random previous allocation
                for j in range(3):  # Try up to 3 times to find a releasable request
                    if len(self.request_history) > 0:
                        past_request_idx = random.randint(0, len(self.request_history) - 1)
                        past_request = self.request_history[past_request_idx]
                        
                        if "device" in past_request and "action" not in past_request:
                            # Release it
                            fps_to_release = past_request["fps"] * random.uniform(0.5, 1.0)
                            self.release_request(past_request["model"], fps_to_release, past_request["device"])
                            break
            
            # Occasionally try to rebalance the load
            if i > 20 and i % 10 == 0:
                self.load_balance()
        
        # Get final system status
        return self.system_status()
    
    def visualize_history(self):
        """Create visualizations of the scheduling history"""
        if not self.request_history:
            return "No history to visualize"
        
        # Convert history to DataFrame
        history_df = pd.DataFrame(self.request_history)
        
        # Create power consumption over time plot
        plt.figure(figsize=(12, 8))
        
        # Filter for regular requests (not load balancing actions)
        request_df = history_df[~history_df.get('action', '').str.contains('load_balance', na=False)]
        
        # Aggregate data by device
        device_power = {}
        for device in self.devices:
            device_power[device.name] = []
        
        timestamps = sorted(request_df['timestamp'].unique())
        
        for ts in timestamps:
            # Get all requests up to this timestamp
            requests_so_far = request_df[request_df['timestamp'] <= ts]
            
            # Calculate cumulative power for each device
            for device in self.devices:
                device_requests = requests_so_far[requests_so_far['device'] == device.name]
                if not device_requests.empty:
                    latest_power = device_requests.iloc[-1]['total_device_power']
                    device_power[device.name].append(latest_power)
                else:
                    device_power[device.name].append(0)
        
        # Plot power consumption over time
        for device_name, power_values in device_power.items():
            # Make sure we have enough values (pad with zeros if needed)
            if len(power_values) < len(timestamps):
                power_values.extend([0] * (len(timestamps) - len(power_values)))
            plt.plot(timestamps[:len(power_values)], power_values, label=device_name)
        
        plt.title('Device Power Consumption Over Time')
        plt.xlabel('Request Timestamp')
        plt.ylabel('Power Consumption (mW)')
        plt.legend()
        plt.grid(False)
        plt.savefig("Scheduler-Device-Power-Consumption-Over-Time.png", format='png', dpi=300)
        # Create model allocation distribution
        plt.figure(figsize=(12, 8))
         
        # Count models on each device
        model_counts = {}
        for device in self.devices:
            model_counts[device.name] = {}
            for model in self.models:
                model_counts[device.name][model.name] = 0
        
        # Update counts based on final allocation
        for device in self.devices:
            for model_name, fps in device.current_fps.items():
                if fps > 0:
                    model_counts[device.name][model_name] = fps
        
        # Prepare data for stacked bar chart
        device_names = list(model_counts.keys())
        model_names = [model.name for model in self.models]
        
        data = {}
        for model_name in model_names:
            data[model_name] = []
            for device_name in device_names:
                data[model_name].append(model_counts[device_name][model_name])
        
        # Create stacked bar
        bottom = np.zeros(len(device_names))
        for model_name in model_names:
            plt.bar(device_names, data[model_name], bottom=bottom, label=model_name)
            bottom += np.array(data[model_name])
        
        plt.title('Model FPS Allocation by Device')
        plt.xlabel('Device')
        plt.ylabel('Allocated FPS')
        plt.legend()
        plt.savefig("Scheduler Model FPS Allocation by Device.png", format='png', dpi=300) 
        return "Visualizations generated"

# Run a simulation
def run_simulation():
    # Initialize the scheduler
    scheduler = Scheduler(devices, models)
    
    print("Starting simulation...")
    final_status = scheduler.simulate_random_workload(num_requests=1000)
    
    print("\nFinal System Status:")
    for device_status in final_status:
        print(f"\n{device_status['device']} - Power: {device_status['current_power']:.2f}mW")
        for model, fps in device_status['models'].items():
            print(f"  - {model}: {fps:.2f} FPS")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    scheduler.visualize_history()
    
    print("\nTotal number of events:", len(scheduler.request_history))
    
    # Calculate overall efficiency
    total_power = sum(device_status['current_power'] for device_status in final_status)
    total_fps = sum(sum(fps for fps in device_status['models'].values()) 
                    for device_status in final_status)
    
    if total_fps > 0:
        efficiency = total_power / total_fps
        print(f"\nOverall System Efficiency: {efficiency:.4f} mW/FPS")
    
    return scheduler

# Main execution
if __name__ == "__main__":
    scheduler = run_simulation()


# import numpy as np
# import pandas as pd
# from dataclasses import dataclass
# import random
# import matplotlib.pyplot as plt

# # Define the structures to represent our system
# @dataclass
# class Model:
#     name: str
#     max_fps: float

# @dataclass
# class Device:
#     name: str
#     power_coefficients: dict
#     max_fps: dict
#     current_fps: dict
#     current_power: float = 0.0

# # Function to calculate power consumption based on FPS
# def calculate_power(fps, coefficients):
#     a, b, c, d = coefficients
#     power = a * (fps ** 3) + b * (fps ** 2) + c * fps + d
#     return max(0, power)

# # Create sample models and devices (unchanged)
# models = [
#     Model(name="ResNet50", max_fps=175),
#     Model(name="MobileNet", max_fps=1000),
#     Model(name="ResNet18", max_fps=425),
#     Model(name="ResNext50", max_fps=110),
#     Model(name="SqueezeNet", max_fps=950)
# ]

# devices = [
#     Device(
#         name="Device1 (Jetson Orin Nano)",
#         power_coefficients={
#             "ResNet50": [6.072131e-04, -1.300612e-01, 4.557199e+01, 4285.86],
#             "MobileNet": [3.521553e-06, -8.418713e-03, 5.996142e+00, 4232.51],
#             "ResNet18": [-4.366841e-05, 7.913778e-03, 1.663735e+01, 4308.14],
#             "ResNext50": [-6.402410e-04, 7.870599e-02, 5.164135e+01, 4239.72],
#             "SqueezeNet": [1.073624e-05, -1.839516e-02, 1.024424e+01, 4160.02]
#         },
#         max_fps={
#             "ResNet50": 174.96,
#             "MobileNet": 967.68,
#             "ResNet18": 413.15,
#             "ResNext50": 108.93,
#             "SqueezeNet": 922.49
#         },
#         current_fps={model.name: 0 for model in models}
#     ),
#     Device(
#         name="Device2 (Jetson Nano)",
#         power_coefficients={
#             "ResNet50": [-2.057230e+01, 6.123255e+02, -4.061582e+03, 6333.14],
#             "MobileNet": [1.854938e-03, -6.409779e-01, 6.233325e+01, 1911.11],
#             "ResNet18": [-1.888558e-02, 9.252537e+01, 1.157932e+02, 2803.937],
#             "ResNext50": [-1.988446e+02, 4.591937e+03, -2.792636e+04, 26012.155],
#             "SqueezeNet": [1.590855e-03, -5.632611e-01, 5.534153e+01, 2407.379]
#         },
#         max_fps={
#             "ResNet50": 19.19,
#             "MobileNet": 201.96,
#             "ResNet18": 63.8,
#             "ResNext50": 12.3,
#             "SqueezeNet": 153.29
#         },
#         current_fps={model.name: 0 for model in models}
#     ),
#     Device(
#         name="Device3 (Raspberry Pi 4 with hailo-8)",
#         power_coefficients={
#             "ResNet50": [0, -0.312, 25.71, 4484.83],
#             "MobileNet": [0, -0.04, 7.56, 4849.85],
#             "ResNet18": [0, 0.0018, 0.15, 5107.62],
#             "ResNext50": [0, -0.446, 31.81, 4724.72],
#             "SqueezeNet": [0, -0.00071, 0.53, 5069.21]
#         },
#         max_fps={
#             "ResNet50": 44,
#             "MobileNet": 138,
#             "ResNet18": 318,
#             "ResNext50": 45,
#             "SqueezeNet": 689
#         },
#         current_fps={model.name: 0 for model in models}
#     )
# ]

# def update_device_power(device):
#     power = 0
#     for model_name, fps in device.current_fps.items():
#         if fps > 0:
#             coefficients = device.power_coefficients[model_name]
#             power += calculate_power(fps, coefficients)
#     device.current_power = power
#     return power

# # AHP Implementation
# class AHP:
#     def __init__(self, criteria, alternatives):
#         self.criteria = criteria
#         self.alternatives = alternatives
#         self.criteria_weights = None
#         self.decision_matrix = None
    
#     def pairwise_comparison(self, values):
#         n = len(values)
#         matrix = np.ones((n, n))
#         for i in range(n):
#             for j in range(i+1, n):
#                 matrix[i][j] = values[i] / values[j] if values[j] != 0 else 1
#                 matrix[j][i] = 1 / matrix[i][j]
#         return matrix
    
#     def normalize_matrix(self, matrix):
#         column_sums = np.sum(matrix, axis=0)
#         return matrix / column_sums
    
#     def calculate_weights(self, matrix):
#         return np.mean(self.normalize_matrix(matrix), axis=1)
    
#     def evaluate_models(self, devices, model_data):
#         power_eff_matrix = self.pairwise_comparison([1/min(d['power']) if d['power'] > 0 else 1 
#                                                     for d in model_data])
#         fps_matrix = self.pairwise_comparison([d['max_fps'] for d in model_data])
#         load_matrix = self.pairwise_comparison([1/(d['current_load'] + 1) for d in model_data])
        
#         self.criteria_weights = np.array([1/3, 1/3, 1/3])
#         power_weights = self.calculate_weights(power_eff_matrix)
#         fps_weights = self.calculate_weights(fps_matrix)
#         load_weights = self.calculate_weights(load_matrix)
        
#         self.decision_matrix = np.vstack([power_weights, fps_weights, load_weights])
#         final_scores = np.dot(self.criteria_weights, self.decision_matrix)
        
#         return dict(zip(self.alternatives, final_scores))

# # Modified Scheduler class
# class Scheduler:
#     def __init__(self, devices, models):
#         self.devices = devices
#         self.models = models
#         self.request_history = []
#         self.last_used_model = None
#         self.default_device = "Device1 (Jetson Orin Nano)"
        
#     def select_best_model(self, required_fps):
#         if not self.models:
#             return None
            
#         model_data = []
#         for model in self.models:
#             min_power = float('inf')
#             current_load = 0
#             for device in self.devices:
#                 if model.name in device.max_fps:
#                     coeffs = device.power_coefficients[model.name]
#                     power = calculate_power(required_fps, coeffs)
#                     min_power = min(min_power, power)
#                     current_load += device.current_fps.get(model.name, 0)
                    
#             model_data.append({
#                 'power': min_power,
#                 'max_fps': model.max_fps,
#                 'current_load': current_load
#             })
        
#         if not model_data or all(d['power'] == float('inf') for d in model_data):
#             return self.last_used_model if self.last_used_model else self.models[0].name
            
#         ahp = AHP(['power_efficiency', 'fps_capability', 'current_load'], 
#                  [m.name for m in self.models])
#         scores = ahp.evaluate_models(self.devices, model_data)
#         return max(scores.items(), key=lambda x: x[1])[0]
    
#     def get_optimal_device_for_model(self, model_name, required_fps):
#         best_device = None
#         min_power_increase = float('inf')
        
#         for device in self.devices:
#             if device.current_fps[model_name] + required_fps <= device.max_fps[model_name]:
#                 current_power = update_device_power(device)
#                 original_fps = device.current_fps[model_name]
#                 device.current_fps[model_name] += required_fps
#                 new_power = update_device_power(device)
#                 device.current_fps[model_name] = original_fps
#                 update_device_power(device)
#                 power_increase = new_power - current_power
                
#                 if power_increase < min_power_increase:
#                     min_power_increase = power_increase
#                     best_device = device
        
#         if best_device is None:
#             for device in self.devices:
#                 if device.name == self.default_device and model_name in device.max_fps:
#                     best_device = device
#                     min_power_increase = calculate_power(required_fps, 
#                                                        device.power_coefficients[model_name])
#                     break
        
#         return best_device, min_power_increase
    
#     def handle_request(self, model_name=None, required_fps=None):
#         if required_fps is None or required_fps <= 0:
#             return False, "Invalid FPS requirement"
            
#         if model_name is None or model_name not in [m.name for m in self.models]:
#             model_name = self.select_best_model(required_fps)
            
#         best_device, power_increase = self.get_optimal_device_for_model(model_name, required_fps)
        
#         if best_device is None:
#             return False, "No device can handle this request"
        
#         best_device.current_fps[model_name] += required_fps
#         update_device_power(best_device)
        
#         self.last_used_model = model_name
#         self.request_history.append({
#             "timestamp": len(self.request_history),
#             "model": model_name,
#             "fps": required_fps,
#             "device": best_device.name,
#             "power_increase": power_increase,
#             "total_device_power": best_device.current_power
#         })
        
#         return True, f"Allocated {required_fps} FPS of {model_name} to {best_device.name}"
    
#     def release_request(self, model_name, fps_to_release, device_name):
#         for device in self.devices:
#             if device.name == device_name:
#                 if device.current_fps[model_name] >= fps_to_release:
#                     device.current_fps[model_name] -= fps_to_release
#                     update_device_power(device)
#                     return True, f"Released {fps_to_release} FPS of {model_name} from {device.name}"
#                 else:
#                     return False, "Invalid release request: FPS allocation doesn't match"
#         return False, "Device not found"
    
#     def system_status(self):
#         status = []
#         for device in self.devices:
#             update_device_power(device)
#             status.append({
#                 "device": device.name,
#                 "current_power": device.current_power,
#                 "models": {model: fps for model, fps in device.current_fps.items() if fps > 0}
#             })
#         return status
    
#     def load_balance(self):
#         changes_made = False
#         for source_device in self.devices:
#             for model_name, current_fps in source_device.current_fps.items():
#                 if current_fps <= 0:
#                     continue
#                 fps_to_move = current_fps / 2
#                 if fps_to_move < 1:
#                     continue
                    
#                 original_power = update_device_power(source_device)
#                 source_device.current_fps[model_name] -= fps_to_move
#                 new_source_power = update_device_power(source_device)
#                 source_power_decrease = original_power - new_source_power
                
#                 best_target = None
#                 min_power_increase = float('inf')
                
#                 for target_device in self.devices:
#                     if target_device == source_device:
#                         continue
#                     if target_device.current_fps[model_name] + fps_to_move <= target_device.max_fps[model_name]:
#                         original_target_power = update_device_power(target_device)
#                         target_device.current_fps[model_name] += fps_to_move
#                         new_target_power = update_device_power(target_device)
#                         target_device.current_fps[model_name] -= fps_to_move
#                         update_device_power(target_device)
#                         target_power_increase = new_target_power - original_target_power
                        
#                         if target_power_increase < min_power_increase:
#                             min_power_increase = target_power_increase
#                             best_target = target_device
                
#                 source_device.current_fps[model_name] += fps_to_move
#                 update_device_power(source_device)
                
#                 if best_target is not None and min_power_increase < source_power_decrease:
#                     source_device.current_fps[model_name] -= fps_to_move
#                     best_target.current_fps[model_name] += fps_to_move
#                     update_device_power(source_device)
#                     update_device_power(best_target)
                    
#                     self.request_history.append({
#                         "timestamp": len(self.request_history),
#                         "action": "load_balance",
#                         "model": model_name,
#                         "fps": fps_to_move,
#                         "from_device": source_device.name,
#                         "to_device": best_target.name,
#                         "power_saving": source_power_decrease - min_power_increase
#                     })
#                     changes_made = True
#         return changes_made
    
#     def simulate_random_workload(self, num_requests=100, min_fps=5, max_fps=1000):
#         for i in range(num_requests):
#             if i % 100 == 0:
#                 print(f"Processing request {i}...")
            
#             if random.random() < 0.5:
#                 model_name = random.choice([model.name for model in self.models])
#             else:
#                 model_name = None
                
#             required_fps = random.randint(min_fps, max_fps)
#             success, message = self.handle_request(model_name, required_fps)
            
#             if i > 10 and random.random() < 0.3:
#                 for j in range(3):
#                     if len(self.request_history) > 0:
#                         past_request_idx = random.randint(0, len(self.request_history) - 1)
#                         past_request = self.request_history[past_request_idx]
#                         if "device" in past_request and "action" not in past_request:
#                             fps_to_release = past_request["fps"] * random.uniform(0.5, 1.0)
#                             self.release_request(past_request["model"], fps_to_release, past_request["device"])
#                             break
            
#             if i > 20 and i % 10 == 0:
#                 self.load_balance()
        
#         return self.system_status()
    
#     def visualize_history(self):
#         if not self.request_history:
#             return "No history to visualize"
        
#         history_df = pd.DataFrame(self.request_history)
#         plt.figure(figsize=(12, 8))
#         request_df = history_df[~history_df.get('action', '').str.contains('load_balance', na=False)]
        
#         device_power = {device.name: [] for device in self.devices}
#         timestamps = sorted(request_df['timestamp'].unique())
        
#         for ts in timestamps:
#             requests_so_far = request_df[request_df['timestamp'] <= ts]
#             for device in self.devices:
#                 device_requests = requests_so_far[requests_so_far['device'] == device.name]
#                 if not device_requests.empty:
#                     latest_power = device_requests.iloc[-1]['total_device_power']
#                     device_power[device.name].append(latest_power)
#                 else:
#                     device_power[device.name].append(0)
        
#         for device_name, power_values in device_power.items():
#             if len(power_values) < len(timestamps):
#                 power_values.extend([0] * (len(timestamps) - len(power_values)))
#             plt.plot(timestamps[:len(power_values)], power_values, label=device_name)
        
#         plt.title('Device Power Consumption Over Time')
#         plt.xlabel('Request Timestamp')
#         plt.ylabel('Power Consumption (mW)')
#         plt.legend()
#         plt.grid(False)
#         plt.savefig("Scheduler-Device-Power-Consumption-Over-Time.png", format='png', dpi=300)
        
#         plt.figure(figsize=(12, 8))
#         model_counts = {device.name: {model.name: 0 for model in self.models} 
#                        for device in self.devices}
        
#         for device in self.devices:
#             for model_name, fps in device.current_fps.items():
#                 if fps > 0:
#                     model_counts[device.name][model_name] = fps
        
#         device_names = list(model_counts.keys())
#         model_names = [model.name for model in self.models]
#         data = {model_name: [model_counts[device_name][model_name] 
#                 for device_name in device_names] for model_name in model_names}
        
#         bottom = np.zeros(len(device_names))
#         for model_name in model_names:
#             plt.bar(device_names, data[model_name], bottom=bottom, label=model_name)
#             bottom += np.array(data[model_name])
        
#         plt.title('Model FPS Allocation by Device')
#         plt.xlabel('Device')
#         plt.ylabel('Allocated FPS')
#         plt.legend()
#         plt.savefig("Scheduler-Model-FPS-Allocation-by-Device.png", format='png', dpi=300)
#         return "Visualizations generated"

# def run_simulation():
#     scheduler = Scheduler(devices, models)
#     print("Starting simulation...")
#     final_status = scheduler.simulate_random_workload(num_requests=1000)
    
#     print("\nFinal System Status:")
#     for device_status in final_status:
#         print(f"\n{device_status['device']} - Power: {device_status['current_power']:.2f}mW")
#         for model, fps in device_status['models'].items():
#             print(f"  - {model}: {fps:.2f} FPS")
    
#     print("\nGenerating visualizations...")
#     scheduler.visualize_history()
    
#     print("\nTotal number of events:", len(scheduler.request_history))
    
#     total_power = sum(device_status['current_power'] for device_status in final_status)
#     total_fps = sum(sum(fps for fps in device_status['models'].values()) 
#                    for device_status in final_status)
    
#     if total_fps > 0:
#         efficiency = total_power / total_fps
#         print(f"\nOverall System Efficiency: {efficiency:.4f} mW/FPS")
    
#     return scheduler

# if __name__ == "__main__":
#     scheduler = run_simulation()