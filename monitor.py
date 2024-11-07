import ast
import argparse
import json
parser = argparse.ArgumentParser()

parser.add_argument('--file', dest='file_name', type=str, help='Add file_name')
parser.add_argument('--start', dest='start_time', type=str, help='Add start_time')
parser.add_argument('--end', dest='end_time', type=str, help='Add end_time')
args = parser.parse_args()


# Function to read logs from a file and convert them into a list of dictionaries
def read_logs_and_cal_avg(file_path):
    # Find the model name (assuming it's between underscores and before the file extension)
    model_name = str(file_path.split('_')[2].split('.')[0])
    device = str(file_path.split('_')[0].split('/')[3])
    total_cpu_temp = 0
    total_gpu_temp = 0
    power_cpu = 0
    power_gpu = 0
    power_tot = 0
    log_count = 0
    ram = 0
    # Open the file in read mode
    with open(file_path, 'r') as file:
        for line in file:
            # Remove any extra spaces or newlines
            line = line.strip()
            if line:
                # Convert the string representation of dictionary to an actual dictionary
                log_entry = ast.literal_eval(line)
                timeval = log_entry['time']
                if timeval >= args.start_time and timeval <= args.end_time:
                    print(log_entry['time']) 
                    total_cpu_temp += log_entry['Temp CPU']
                    total_gpu_temp += log_entry['Temp GPU']
                    power_cpu += log_entry['Power POM_5V_CPU']
                    power_gpu += log_entry['Power POM_5V_GPU']
                    power_tot += log_entry['Power TOT']
                    ram += log_entry['RAM']
                    log_count += 1

    avg_cpu_temp = total_cpu_temp / log_count if log_count > 0 else 0
    avg_gpu_temp = total_gpu_temp / log_count if log_count > 0 else 0
    avg_power_cpu = power_cpu / log_count if log_count > 0 else 0
    avg_power_gpu = power_cpu / log_count if log_count > 0 else 0
    avg_power_tot = power_tot / log_count if log_count > 0 else 0
    avg_ram = ram / log_count if log_count > 0 else 0
    print("")    
    print(f"Average CPU Temperature: {avg_cpu_temp:.2f}°C")
    print(f"Average GPU Temperature: {avg_gpu_temp:.2f}°C")
    print(f"Average Power CPU: {avg_power_cpu:.2f}mW")
    print(f"Average Power GPU: {avg_power_gpu:.2f}mW")
    print(f"Average Power TOT: {avg_power_tot:.2f}mW")
    print(f"Average RAM: {avg_ram:.4f}")

    # Specify the output JSON file path
    output_file_path = 'model_metrics.json'
    result = [{'Model': model_name,'Device': device,'Avg_CPU_temp':avg_cpu_temp, 'Avg_GPU_temp': avg_gpu_temp, 'Avg_Power_CPU': avg_power_cpu, 'Avg_Power_GPU': avg_power_gpu, 'Avg_CPU_RAM': avg_ram}]
    
    try:
        with open(output_file_path, 'r') as json_file:
            # Load existing data
            existing_data = json.load(json_file)
    except FileNotFoundError:
        # If file does not exist, start with an empty list
        existing_data = []

    # Ensure existing data is a list (in case the JSON structure is unexpected)
    if isinstance(existing_data, list):
        # Append new data to the list
        existing_data.extend(result)
    else:
        print("Error: Expected a list in the JSON file.")
        existing_data = result  # Start a new list if format is not as expected

    # Write the updated list back to the file
    with open(output_file_path, 'w') as json_file:
        json.dump(existing_data, json_file, indent=4)

    return result

# Path to the log file
file_path = f'./model-logs/resnet-18/{args.file_name}'
# Read the logs and calculate avg
status = read_logs_and_cal_avg(file_path)
print("")
print(status)