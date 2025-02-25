import time
import matplotlib.pyplot as plt
from typing import List, Tuple
import numpy as np

def benchmark_channels(
    image_paths: List[str],
    device_type: str,
    model_name: str = 'squeezenet',
    channel_range: range = range(1, 11),
    iterations: int = 10
) -> List[Tuple[int, float]]:
    """
    Benchmark different numbers of channels to find optimal performance.
    
    Args:
        image_paths: List of paths to test images
        device_type: Type of device to test
        model_name: Name of the model to use
        channel_range: Range of channel numbers to test
        iterations: Number of iterations per channel count
    
    Returns:
        List of tuples containing (channel_count, throughput)
    """
    results = []
    
    for num_channels in channel_range:
        print(f"\nTesting with {num_channels} channels...")
        
        # Run multiple iterations to get average performance
        total_time = 0
        total_images = 0
        
        for i in range(iterations):
            start_time = time.time()
            
            # Process batch with current channel count
            batch_results, _ = process_batch(
                image_paths=image_paths,
                device_type=device_type,
                model_name=model_name,
                batch_size=len(image_paths),
                num_iterations=1,  # Single iteration per test
                num_channels=num_channels
            )
            
            iteration_time = time.time() - start_time
            total_time += iteration_time
            total_images += len(batch_results)
            
            print(f"Iteration {i+1}: Processed {len(batch_results)} images in {iteration_time:.2f} seconds")
        
        # Calculate average throughput (images per second)
        avg_throughput = total_images / total_time
        results.append((num_channels, avg_throughput))
        
        print(f"Average throughput with {num_channels} channels: {avg_throughput:.2f} images/second")
    
    return results

def plot_results(results: List[Tuple[int, float]], save_path: str = None):
    """
    Plot the benchmarking results.
    
    Args:
        results: List of (channel_count, throughput) tuples
        save_path: Optional path to save the plot
    """
    channels, throughputs = zip(*results)
    
    plt.figure(figsize=(10, 6))
    plt.plot(channels, throughputs, 'bo-')
    plt.grid(True)
    plt.xlabel('Number of Channels')
    plt.ylabel('Throughput (images/second)')
    plt.title('RabbitMQ Channel Count vs Throughput')
    
    # Find optimal point
    optimal_channels = channels[np.argmax(throughputs)]
    max_throughput = max(throughputs)
    
    plt.annotate(f'Optimal: {optimal_channels} channels\n{max_throughput:.2f} imgs/sec',
                xy=(optimal_channels, max_throughput),
                xytext=(5, 5), textcoords='offset points')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def find_optimal_channels(image_paths: List[str], device_type: str, model_name: str = 'squeezenet'):
    """
    Find the optimal number of channels for your specific setup.
    
    Args:
        image_paths: List of paths to test images
        device_type: Type of device to test
        model_name: Name of the model to use
    """
    # Test range of channel counts
    results = benchmark_channels(
        image_paths=image_paths,
        device_type=device_type,
        model_name=model_name,
        channel_range=range(1, 11),  # Test 1-10 channels
        iterations=3  # Number of iterations per channel count
    )
    
    # Plot results
    plot_results(results, 'channel_benchmark_results.png')
    
    # Find optimal channel count
    optimal_channels, max_throughput = max(results, key=lambda x: x[1])
    
    print("\nBenchmark Results:")
    print(f"Optimal number of channels: {optimal_channels}")
    print(f"Maximum throughput: {max_throughput:.2f} images/second")
    
    return optimal_channels

# Example usage
if __name__ == "__main__":
    image_paths = [
        "E:/NCL/NCL-Intern/Jetson_Benchmarking/images/cat_1.jpg",
        "E:/NCL/NCL-Intern/Jetson_Benchmarking/images/cat_2.jpg",
        "E:/NCL/NCL-Intern/Jetson_Benchmarking/images/cat_3.jpg",
        "E:/NCL/NCL-Intern/Jetson_Benchmarking/images/dog_0.jpg",
        "E:/NCL/NCL-Intern/Jetson_Benchmarking/images/dog_1.jpg",
        "E:/NCL/NCL-Intern/Jetson_Benchmarking/images/dog_2.jpg",
    ]
    
    optimal_channels = find_optimal_channels(
        image_paths=image_paths,
        device_type='raspberry_pi',
        model_name='squeezenet'
    )