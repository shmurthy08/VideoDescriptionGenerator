# Define the filename
filename = 'container.txt'

# List to store the values
container_name_values = []
container_name = input("Enter the container name: ")

# Open and read the file
with open(filename, 'r') as file:
    # Read each line in the file
    for line in file:
        # Check if the line contains "container_name"
        if container_name in line.lower():
            # Split the line by space and store the first value
            first_value = line.split()[0]
            # Append the first value to the list
            container_name_values.append(first_value)

# Print the list of first values
print(container_name_values)
print(len(container_name_values))
# Store the list into a variable (optional)
container_name_values_variable = container_name_values


input("Press Enter to continue or CNTRL+C to exit...")
import os
for val in container_name_values:
    os.system('docker stop ' + val)
    os.system('docker rm ' + val)