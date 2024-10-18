import ast
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--start', dest='start_time', type=str, help='Add start_time')
parser.add_argument('--end', dest='end_time', type=str, help='Add end_time')
args = parser.parse_args()
# Function to read logs from a file and convert them into a list of dictionaries
def read_logs_and_cal_avg(file_path):
    logs = []
    total_cpu_temp = 0
    total_gpu_temp = 0
    log_count = 0
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
                    
                    log_count += 1
    avg_cpu_temp = total_cpu_temp / log_count if log_count > 0 else 0
    avg_gpu_temp = total_gpu_temp / log_count if log_count > 0 else 0
    return avg_cpu_temp, avg_gpu_temp

# Path to the log file
file_path = 'resnet18-1.txt'

# Read the logs and calculate avg
avg_cpu_temp, avg_gpu_temp = read_logs_and_cal_avg(file_path)

print(f"Average CPU Temperature: {avg_cpu_temp:.2f}°C")
print(f"Average GPU Temperature: {avg_gpu_temp:.2f}°C")