import ast
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--file', dest='file_name', type=str, help='Add file_name')
parser.add_argument('--start', dest='start_time', type=str, help='Add start_time')
parser.add_argument('--end', dest='end_time', type=str, help='Add end_time')
args = parser.parse_args()


# Function to read logs from a file and convert them into a list of dictionaries
def read_logs_and_cal_avg(file_path):
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
    return {'Avg_CPU_temp':avg_cpu_temp, 'Avg_GPU_temp': avg_gpu_temp, 'Avg_Power_CPU': avg_power_cpu, 'Avg_Power_GPU': avg_power_gpu, 'Avg_CPU_RAM': avg_ram}

# Path to the log file
file_path = f'./model-logs/resnext50-32x4d/{args.file_name}'
# Read the logs and calculate avg
status = read_logs_and_cal_avg(file_path)
print("")
print(status)