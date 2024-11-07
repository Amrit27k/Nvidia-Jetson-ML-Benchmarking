import ast
import re
import csv
import datetime

# Function to convert timedelta to human-readable format
def format_timedelta(td_str):
    # Convert the string to a timedelta object
    td_obj = eval(td_str)  # Assumes input is a safe trusted source
    total_seconds = int(td_obj.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def exe_conversion(input_file, output_file):

    # Define a pattern to extract each image's details
    image_data_pattern = re.compile(
        r"Loading image: (?P<load_time>[^\n]+)\n"
        r"Image transformation: (?P<transform_time>[^\n]+)\n"
        r"During prediction: (?P<predict_time>[^\n]+)\n"
        r"Image: (?P<image_name>[^\n]+)\n"
        r"Resolution: (?P<resolution>[^\n]+)\n"
        r"Inference time: (?P<inference_time>[^\n]+) seconds\n"
        r"Accuracy: (?P<accuracy>[^\n]+)%\n"
        r"Argmax Pred class:(?P<argmax_class>[^\n]+)\n"
        r"TopK Predicted class:(?P<topk_class>[^\n]+)\n"
        r"Latency: (?P<latency>[^\n]+) seconds"
    )

    # Read data from the input file and parse it
    with open(input_file, 'r') as file:
        text_data = file.read()

    # Extract all image data using the defined regex pattern
    image_data_matches = image_data_pattern.finditer(text_data)

    # Prepare to write to CSV
    with open(output_file, 'a', newline='') as csvfile:
        fieldnames = [
            'Load Time', 'Transform Time', 'Predict Time', 'Image Name',
            'Resolution', 'Inference Time', 'Accuracy', 'Argmax Class', 'TopK Class', 'Latency'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csvfile.write('\n\n')
        writer.writeheader()

        # Write each match to the CSV
        for match in image_data_matches:
            # Parse match into a dictionary
            row = {
                'Load Time': match.group('load_time'),
                'Transform Time': match.group('transform_time'),
                'Predict Time': match.group('predict_time'),
                'Image Name': match.group('image_name'),
                'Resolution': match.group('resolution'),
                'Inference Time': float(match.group('inference_time')),
                'Accuracy': float(match.group('accuracy')),
                'Argmax Class': match.group('argmax_class').strip(),
                'TopK Class': match.group('topk_class').strip(),
                'Latency': float(match.group('latency'))
            }
            writer.writerow(row)


def logs_conversion(input_file, csv_file):
    
    # Read and parse the data from the text file
    data = []
    with open(input_file, 'r') as file:
        for line in file:
            # Parse each line as a dictionary
            entry = ast.literal_eval(line.strip())
            
            # Convert datetime and timedelta strings to objects
            entry['time'] = eval(entry['time']).isoformat()  # Convert datetime to ISO string
            entry['uptime'] = format_timedelta(entry['uptime'])  # Convert timedelta to human-readable
            
            # Add the dictionary to our data list
            data.append(entry)

    # Write data to CSV
    with open(csv_file, mode='a', newline='') as file:
        # Extract headers from the keys of the first dictionary
        fieldnames = data[0].keys()
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        file.write('\n\n')
        # Write headers and rows
        writer.writeheader()
        writer.writerows(data)

input_file = './model-logs/mobilenetv3/gpu_logs_mobilenet_max.txt'
output_file = 'output_logs_gpu.csv'
# exe_conversion(input_file, output_file)
logs_conversion(input_file, output_file)
print(f"Data successfully saved to {output_file}")
