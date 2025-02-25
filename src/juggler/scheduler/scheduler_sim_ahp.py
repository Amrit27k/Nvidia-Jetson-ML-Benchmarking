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
        name="Device3 (Raspberry Pi 4 with hailo-8)", 
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
        
    def calculate_consistency_ratio(self, matrix):
        """Calculate consistency ratio to check if comparisons are consistent"""
        n = len(matrix)
        
        # Random Index (RI) values for different matrix sizes
        ri_values = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
        
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        
        # Find the maximum eigenvalue
        max_eigenvalue = max(eigenvalues.real)
        
        # Calculate Consistency Index (CI)
        ci = (max_eigenvalue - n) / (n - 1)
        
        # Calculate Consistency Ratio (CR)
        if n <= 10:
            cr = ci / ri_values[n]
            return cr
        else:
            return None  # RI values not defined for n > 10
    
    def create_pairwise_matrix(self, criteria, comparison_func):
        """Create a pairwise comparison matrix based on the provided comparison function"""
        n = len(criteria)
        matrix = np.ones((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                # Apply the comparison function to get the relative importance
                comparison = comparison_func(criteria[i], criteria[j])
                matrix[i, j] = comparison
                matrix[j, i] = 1 / comparison
                
        return matrix
    
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

# Enhanced Scheduler class with AHP
class AHPScheduler:
    def __init__(self, devices, models):
        self.devices = devices
        self.models = models
        self.request_history = []  # To track request allocations
        self.ahp = AHPDecisionMaker()
        self.last_used_device = None
        self.last_used_model = None
        
    def select_model_using_ahp(self, required_fps):
        """Use AHP to select the best model based on criteria"""
        # Define criteria for model selection
        criteria = ["performance", "power_efficiency", "availability"]
        
        # Define criteria weights (can be adjusted based on priorities)
        # Example: performance (40%), power_efficiency (40%), availability (20%)
        criteria_weights = np.array([0.4, 0.4, 0.2])
        
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
            # Higher availability = more available FPS across devices
            available_fps = 0
            for device in self.devices:
                if model.name in device.max_fps:
                    available_fps += (device.max_fps[model.name] - device.current_fps[model.name])
            
            return available_fps / len(self.devices)
        
        scoring_funcs = [score_performance, score_power_efficiency, score_availability]
        
        # Evaluate models
        alternatives = [model for model in self.models]
        scores = self.ahp.evaluate_alternatives(alternatives, criteria, criteria_weights, scoring_funcs)
        
        # Find models that can satisfy the required FPS
        viable_models = []
        for i, model in enumerate(alternatives):
            for device in self.devices:
                if (model.name in device.max_fps and 
                    device.max_fps[model.name] >= required_fps and
                    device.current_fps[model.name] + required_fps <= device.max_fps[model.name]):
                    viable_models.append((model, scores[i]))
                    break
        
        if viable_models:
            # Sort by AHP score (descending)
            viable_models.sort(key=lambda x: x[1], reverse=True)
            return viable_models[0][0]
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
        # Example: power_efficiency (50%), load_balance (30%), performance_headroom (20%)
        criteria_weights = np.array([0.6, 0.2, 0.2])
        
        # Get viable devices (those that can handle the request)
        viable_devices = []
        for device in self.devices:
            if (model_name in device.max_fps and 
                device.max_fps[model_name] >= required_fps and
                device.current_fps[model_name] + required_fps <= device.max_fps[model_name]):
                viable_devices.append(device)
        
        if not viable_devices:
            if self.last_used_device:
                return self.last_used_device, 0
            return None, float('inf')
        
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
        
        return best_device, power_increase
    
    def handle_request(self, model_name=None, required_fps=0):
        """Process an incoming request, using AHP to select model and device if not specified"""
        if required_fps <= 0:
            return False, "Invalid FPS requirement"
        
        # If model not specified, select one using AHP
        if model_name is None:
            selected_model = self.select_model_using_ahp(required_fps)
            model_name = selected_model.name
            self.last_used_model = selected_model
        
        # Select the optimal device using AHP
        best_device, power_increase = self.select_device_using_ahp(model_name, required_fps)
        
        if best_device is None:
            return False, "No device can handle this request"
        
        self.last_used_device = best_device
        
        # Allocate the request
        best_device.current_fps[model_name] += required_fps
        update_device_power(best_device)
        
        self.request_history.append({
            "timestamp": len(self.request_history),
            "action": "allocate_request",
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
    
    def _release_random_requests(self):
        """Helper method to release random requests"""
        # Try up to 3 times to release some workload
        for _ in range(3):
            if len(self.request_history) > 0:
                # Filter for allocation requests
                allocation_requests = [req for req in self.request_history 
                                    if req.get("action") == 'allocate_request' 
                                    and 'device' in req 
                                    and 'model' in req]
                
                if not allocation_requests:
                    continue
                    
                # Pick a random allocation
                past_request = random.choice(allocation_requests)
                
                # Verify the device still has this allocation
                for device in self.devices:
                    if (device.name == past_request['device'] and 
                        device.current_fps[past_request['model']] > 0):
                        
                        # Release between 60-100% of the allocation
                        fps_to_release = min(
                            past_request['fps'] * random.uniform(0.6, 1.0),
                            device.current_fps[past_request['model']]
                        )
                        
                        if fps_to_release > 0:
                            self.release_request(
                                past_request['model'], 
                                fps_to_release, 
                                past_request['device']
                            )
                            return True
        
        return False

    def simulate_random_workload(self, num_requests=100, min_fps=5, max_fps=1000):
        """Simulate a random workload of requests"""
        for i in range(num_requests):
            if i % 100 == 0:
                print(f"Processing request {i}...")
        
            # Calculate current total FPS across all devices
            current_total_fps = 0
            for device in self.devices:
                current_total_fps += sum(device.current_fps.values())
            
            # Determine available FPS capacity
            available_fps = max_fps - current_total_fps
            
            if available_fps <= 0 and random.random() < 0.7:
                # System is at or above max capacity, focus on releasing
                self._release_random_requests()
                continue
            
            # When below min_fps threshold, always add new requests
            # Otherwise add with decreasing probability as we approach max_fps
            if current_total_fps < min_fps or random.random() < (available_fps / max_fps):
                # Generate random FPS requirement within available capacity
                required_fps = min(random.randint(5, 500), available_fps)
                
                # Skip if we can't allocate a meaningful amount
                if required_fps < 5:
                    continue
                    
                # Choose approach: let AHP select everything or select random model
                if random.random() < 0.5:  # 50% chance to let AHP select everything
                    success, message = self.handle_request(required_fps=500)
                else:  # 50% chance to select random model but AHP selects device
                    model_name = random.choice([model.name for model in self.models])
                    success, message = self.handle_request(model_name=model_name, required_fps=500)
            
            # Release requests based on current system load
            release_probability = max(0.1, min(0.8, current_total_fps / max_fps))
            if random.random() < release_probability:
                self._release_random_requests()
            
            # Try to rebalance the load occasionally
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
        request_df = history_df[~history_df["action"].astype(str).str.contains("load_balance", na=False)]
        
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
        plt.savefig("AHP-Scheduler-Device-Power-Consumption-Over-Time.png", format='png', dpi=300)
        
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
        plt.savefig("AHP-Scheduler-Model-FPS-Allocation-by-Device.png", format='png', dpi=300)
        
        # Create AHP decision criteria visualization
        plt.figure(figsize=(10, 6))
        model_criteria = ["Performance", "Power Efficiency", "Availability"]
        device_criteria = ["Power Efficiency", "Load Balance", "Performance Headroom"]
        
        # Create pie charts for criteria weights
        plt.subplot(1, 2, 1)
        plt.pie([0.4, 0.4, 0.2], labels=model_criteria, autopct='%1.1f%%', startangle=90)
        plt.title('AHP Model Selection Criteria')
        
        plt.subplot(1, 2, 2)
        plt.pie([0.5, 0.3, 0.2], labels=device_criteria, autopct='%1.1f%%', startangle=90)
        plt.title('AHP Device Selection Criteria')
        
        plt.tight_layout()
        plt.savefig("AHP-Decision-Criteria.png", format='png', dpi=300)
        
        return "Visualizations generated"
    
    def visualize_power_vs_fps(self):
        """Create a graph showing the relationship between total power consumption and overall FPS"""
        if not self.request_history:
            return "No history to visualize"
        
        # Convert history to DataFrame
        history_df = pd.DataFrame(self.request_history)
        
        # Filter for regular requests (not load balancing actions)
        request_df = history_df[~history_df["action"].astype(str).str.contains("load_balance", na=False)]
        
        # Track total power and FPS over time
        timestamps = sorted(request_df['timestamp'].unique())
        total_power_over_time = []
        total_fps_over_time = []
        device_models_fps = {device.name: {model.name: [] for model in self.models} for device in self.devices}
        
        # For each timestamp, calculate the total power and total FPS
        for ts in timestamps:
            # Get all requests up to this timestamp
            requests_so_far = request_df[request_df['timestamp'] <= ts]
            
            # Calculate total power and FPS at this point
            total_power = 0
            total_fps = 0
            
            # Reset tracking for this timestamp
            for device in self.devices:
                for model_name in device.current_fps:
                    device.current_fps[model_name] = 0
            
            # Apply all allocations and releases up to this point
            for _, req in requests_so_far.iterrows():
                if req.get("action") == 'allocate_request':
                    # Find the device
                    for device in self.devices:
                        if device.name == req['device']:
                            device.current_fps[req['model']] += req['fps']
                            break
                elif req.get("action") == 'release_request':
                    # Find the device
                    for device in self.devices:
                        if device.name == req['device']:
                            device.current_fps[req['model']] = max(0, device.current_fps[req['model']] - req['fps'])
                            break
            
            # Calculate power and FPS
            for device in self.devices:
                update_device_power(device)
                total_power += device.current_power
                device_fps = sum(device.current_fps.values())
                total_fps += device_fps
                
                # Track FPS for each model on each device
                for model_name, fps in device.current_fps.items():
                    device_models_fps[device.name][model_name].append(fps)
            
            total_power_over_time.append(total_power)
            total_fps_over_time.append(total_fps)
        
        # Create the power vs. FPS graph
        plt.figure(figsize=(12, 8))
        
        # Plot the main power vs. FPS relationship
        sc = plt.scatter(total_fps_over_time, total_power_over_time, 
                        c=range(len(timestamps)), cmap='viridis', 
                        alpha=0.7, s=50)
        
        # Add colorbar for time progression
        cbar = plt.colorbar(sc)
        cbar.set_label('Time Progression')
        
        # Add annotations for points with significant changes
        power_changes = np.diff(total_power_over_time)
        fps_changes = np.diff(total_fps_over_time)
        
        # Calculate the magnitude of changes
        change_magnitudes = np.sqrt(power_changes**2 + fps_changes**2)
        
        # Find significant change points (top 5% of changes)
        threshold = np.percentile(change_magnitudes, 95)
        significant_points = np.where(change_magnitudes > threshold)[0]
        
        # Annotate significant points
        for i in significant_points:
            plt.annotate(f"t={timestamps[i+1]}", 
                        (total_fps_over_time[i+1], total_power_over_time[i+1]),
                        textcoords="offset points", 
                        xytext=(0, 10), 
                        ha='center',
                        arrowprops=dict(arrowstyle='->', color='red'))
        
        plt.title('Power Consumption vs. FPS')
        plt.xlabel('Total FPS Across All Devices')
        plt.ylabel('Total Power Consumption (mW)')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add a trend line
        z = np.polyfit(total_fps_over_time, total_power_over_time, 1)
        p = np.poly1d(z)
        plt.plot(total_fps_over_time, p(total_fps_over_time), "r--", alpha=0.8, 
                label=f"Trend: {z[0]:.2f} mW/FPS")
        
        # Add efficiency lines for reference
        x_range = np.linspace(0, max(total_fps_over_time) * 1.1, 100)
        for efficiency in [5, 10, 15, 20]:
            plt.plot(x_range, efficiency * x_range, 'k:', alpha=0.3, 
                    label=f"{efficiency} mW/FPS" if efficiency == 5 else "")
        
        plt.legend()
        plt.tight_layout()
        plt.savefig("AHP-Scheduler-Power-vs-FPS.png", format='png', dpi=300)
        
        # Create a device-specific power vs. FPS graph to identify inefficient device usage
        plt.figure(figsize=(15, 10))
        
        for i, device in enumerate(self.devices):
            device_power_over_time = []
            device_fps_over_time = []
            
            for ts in timestamps:
                # Get all requests up to this timestamp for this device
                requests_so_far = request_df[(request_df['timestamp'] <= ts) & 
                                            (request_df['device'] == device.name)]
                
                if not requests_so_far.empty:
                    latest_power = requests_so_far.iloc[-1]['total_device_power']
                    device_power_over_time.append(latest_power)
                    
                    # Calculate total FPS for this device at this timestamp
                    device_fps = 0
                    for model_name in device.current_fps:
                        model_requests = requests_so_far[requests_so_far['model'] == model_name]
                        if not model_requests.empty:
                            # Sum allocations and subtract releases
                            model_fps = sum(model_requests[model_requests.get("action") == 'allocate_request']['fps']) - \
                                        sum(model_requests[model_requests.get("action") == 'release_request'].get('fps', 0))
                            device_fps += max(0, model_fps)
                    
                    device_fps_over_time.append(device_fps)
                else:
                    device_power_over_time.append(0)
                    device_fps_over_time.append(0)
            
            # Create subplot for each device
            plt.subplot(len(self.devices), 1, i+1)
            
            # Plot power vs. FPS for this device
            sc = plt.scatter(device_fps_over_time, device_power_over_time, 
                            c=range(len(device_power_over_time)), cmap='viridis', 
                            alpha=0.7, s=30, label=device.name)
            
            # Add trend line for this device
            if len(device_fps_over_time) > 1 and max(device_fps_over_time) > 0:
                z = np.polyfit(device_fps_over_time, device_power_over_time, 1)
                p = np.poly1d(z)
                plt.plot(device_fps_over_time, p(device_fps_over_time), "r--", alpha=0.8, 
                        label=f"Trend: {z[0]:.2f} mW/FPS")
            
            plt.title(f'{device.name} - Power vs. FPS')
            plt.xlabel('FPS')
            plt.ylabel('Power (mW)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig("AHP-Scheduler-Device-Power-vs-FPS.png", format='png', dpi=300)
        
        # Create model efficiency comparison graph
        plt.figure(figsize=(12, 8))
        
        model_colors = plt.cm.tab10(np.linspace(0, 1, len(self.models)))
        
        for device in self.devices:
            # Get power curve for each model on this device
            for i, model in enumerate(self.models):
                fps_values = np.linspace(0, device.max_fps[model.name], 100)
                power_values = []
                
                for fps in fps_values:
                    # Calculate power for this FPS value
                    power = calculate_power(fps, device.power_coefficients[model.name])
                    power_values.append(power)
                
                # Calculate efficiency (Power/FPS)
                efficiency_values = [p/f if f > 0 else 0 for p, f in zip(power_values, fps_values)]
                
                # Find optimal FPS (lowest power per FPS)
                valid_indices = [i for i, f in enumerate(fps_values) if f > 0]
                if valid_indices:
                    optimal_idx = min(valid_indices, key=lambda i: efficiency_values[i])
                    optimal_fps = fps_values[optimal_idx]
                    optimal_power = power_values[optimal_idx]
                    optimal_efficiency = efficiency_values[optimal_idx]
                    
                    plt.plot(fps_values, efficiency_values, 
                            label=f"{model.name} on {device.name}", 
                            color=model_colors[i], alpha=0.7)
                    
                    # Mark optimal point
                    plt.scatter([optimal_fps], [optimal_efficiency], 
                                color=model_colors[i], s=100, edgecolor='black', zorder=10)
                    
                    plt.annotate(f"Optimal: {optimal_fps:.1f} FPS\n{optimal_efficiency:.2f} mW/FPS",
                                (optimal_fps, optimal_efficiency),
                                textcoords="offset points", 
                                xytext=(5, 5), 
                                ha='left')
        
        plt.title('Model Efficiency Curves (Power per FPS)')
        plt.xlabel('FPS')
        plt.ylabel('Efficiency (mW/FPS)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("AHP-Scheduler-Model-Efficiency-Curves.png", format='png', dpi=300)
        
        return "Power vs. FPS visualizations generated"

# Run a simulation with AHP Scheduler
def run_ahp_simulation():
    # Initialize the AHP scheduler
    scheduler = AHPScheduler(devices, models)
    
    print("Starting AHP simulation...")
    final_status = scheduler.simulate_random_workload(num_requests=1000)
    
    print("\nFinal System Status:")
    for device_status in final_status:
        print(f"\n{device_status['device']} - Power: {device_status['current_power']:.2f}mW")
        for model, fps in device_status['models'].items():
            print(f"  - {model}: {fps:.2f} FPS")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    scheduler.visualize_history()
    scheduler.visualize_power_vs_fps()
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
    ahp_scheduler = run_ahp_simulation()