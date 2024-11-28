import ast
import re
import csv
import datetime # don't remove, using internally.

# Function to convert timedelta to human-readable format
def format_timedelta(td_str):
    # Convert the string to a timedelta object
    td_obj = eval(td_str)  # Assumes input is a safe trusted source
    total_seconds = int(td_obj.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def exe_conversion(input_file, output_file):

    # Define patterns to capture loop run count
    loop_count_pattern = re.compile(r"Loop run count: (?P<loop_count>\d+)")
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
    )

    # Read data from the input file and parse it
    with open(input_file, 'r') as file:
        text_data = file.read()

    # Extract all image data using the defined regex pattern
    # Find all loop counts and image data
    loop_count_matches = list(loop_count_pattern.finditer(text_data))
    image_data_matches = list(image_data_pattern.finditer(text_data))
    model_name = str(input_file.split('_')[3].split('.')[0])
    device = str(input_file.split('_')[0].split('/')[3])
    # Prepare to write to CSV
    with open(output_file, 'a', newline='') as csvfile:
        fieldnames = [
            'Loop Run Count', 'Model', 'Device', 'Load Time', 'Transform Time', 'Predict Time', 'Image Name',
            'Resolution', 'Inference Time', 'Accuracy', 'Argmax Class', 'TopK Class'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csvfile.write('\n\n')
        writer.writeheader()

        # Initialize loop count index
        loop_index = 0
        # Write each match to the CSV
        for match in image_data_matches:
            # Update loop count if the current match position has moved past the current loop count position
            if loop_index < len(loop_count_matches) - 1 and match.start() > loop_count_matches[loop_index + 1].start():
                loop_index += 1
            current_loop_count = loop_count_matches[loop_index].group('loop_count')
            # Parse match into a dictionary
            row = {
                'Loop Run Count': int(current_loop_count),
                'Model': model_name,
                'Device': device,
                'Load Time': match.group('load_time'),
                'Transform Time': match.group('transform_time'),
                'Predict Time': match.group('predict_time'),
                'Image Name': match.group('image_name'),
                'Resolution': match.group('resolution'),
                'Inference Time': float(match.group('inference_time')),
                'Accuracy': float(match.group('accuracy')),
                'Argmax Class': match.group('argmax_class').strip(),
                'TopK Class': match.group('topk_class').strip() 
            }
            writer.writerow(row)


def jtop_logs_conversion(input_file, csv_file):
    
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

file_sequence = ["mobilenetv3","resnet18","resnet50","resnext50x32","squeezenet"]
# for file in file_sequence:
input_file = f'./orin-model-logs/resnet50/gpu_logs_exe_resnet50_max2.txt'
output_file = './orin-model-logs/output_logs_exe_GPU_all-1.csv'
exe_conversion(input_file, output_file)
# jtop_logs_conversion(input_file, output_file)
print(f"Data successfully saved to {output_file}")
