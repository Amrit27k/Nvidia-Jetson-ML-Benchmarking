import subprocess

# Variables
container_name = "0fa71d1e529f"
script_path_in_container = "/home/model-run.py"

try:
    # Construct the docker exec command
    command = [
        "docker", "exec", container_name, "python3", script_path_in_container
    ]

    # Execute the command
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Print the output and errors (if any)
    if result.stdout:
        print("Script Output:\n", result.stdout.decode())
    if result.stderr:
        print("Script Errors:\n", result.stderr.decode())

except Exception as e:
    print(f"An error occurred: {e}")