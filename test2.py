from datetime import datetime

class Solution:
    
    # Helper method to parse the timestamp from the log line
    def parse_time(self, log_line):
        timestamp_str = log_line.split(' ')[0]
        return datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S%z")

    # Method to calculate the total duration during which only the RED LED is on
    def calculate_red_only_duration(self, log_lines):
        red_on = False
        green_on = False
        blue_on = False
        red_only_start = None
        total_duration = 0
        last_timestamp = None
        
        # Loop through all log lines
        for log_line in log_lines:
            timestamp = self.parse_time(log_line)
            last_timestamp = timestamp  # Keep track of the last timestamp
            
            if "RED on" in log_line:
                # Start counting if only RED is on
                if not green_on and not blue_on:
                    red_only_start = timestamp
                red_on = True
            elif "RED off" in log_line:
                # Calculate the duration if RED was on alone
                if red_on and red_only_start:
                    duration = (timestamp - red_only_start).total_seconds()
                    total_duration += int(duration)
                    red_only_start = None
                red_on = False
            elif "GREEN on" in log_line:
                # If RED was on alone, calculate the duration until now
                if red_on and red_only_start:
                    duration = (timestamp - red_only_start).total_seconds()
                    total_duration += int(duration)
                    red_only_start = None
                green_on = True
            elif "GREEN off" in log_line:
                green_on = False
                # If RED is still on and BLUE is off, start counting again
                if red_on and not blue_on:
                    red_only_start = timestamp
            elif "BLUE on" in log_line:
                # If RED was on alone, calculate the duration until now
                if red_on and red_only_start:
                    duration = (timestamp - red_only_start).total_seconds()
                    total_duration += int(duration)
                    red_only_start = None
                blue_on = True
            elif "BLUE off" in log_line:
                blue_on = False
                # If RED is still on and GREEN is off, start counting again
                if red_on and not green_on:
                    red_only_start = timestamp

        # Handle the case where the log ends while RED is still on and the others are off
        if red_on and red_only_start:
            duration = (last_timestamp - red_only_start).total_seconds()
            total_duration += int(duration)
        
        return total_duration

# Example usage:
if __name__ == "__main__":
    # Example input log
    log_input = """2022-09-19T22:32:54+0000 INFO RED on
    2022-09-19T22:32:58+0000 INFO RED off
    2022-09-19T22:33:00+0000 INFO RED on
    2022-09-19T22:33:17+0000 INFO BLUE on
    2022-09-19T22:33:33+0000 INFO something else
    2022-09-19T22:33:49+0000 INFO GREEN on
    2022-09-19T22:33:49+0000 INFO something else
    2022-09-19T22:34:01+0000 INFO something else happened
    2022-09-19T22:34:55+0000 INFO something else happened
    2022-09-19T22:36:49+0000 INFO something else again
    2022-09-19T22:38:17+0000 INFO BLUE off
    2022-09-19T22:39:49+0000 INFO RED off
    2022-09-19T22:41:01+0000 INFO BLUE on
    2022-09-19T22:42:07+0000 INFO GREEN off"""

    lo

    # Convert the input into a list of log lines
    log_lines = log_input.splitlines()

    # Create an instance of the Solution class and calculate the duration
    solution = Solution()
    result = solution.calculate_red_only_duration(log_lines)

    # Print the result
    print(result)
