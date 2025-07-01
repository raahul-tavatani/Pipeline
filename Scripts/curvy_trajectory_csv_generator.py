import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Function to generate a curvy trajectory using a sine wave
def generate_trajectory(x_range, y_range, frequency, num_points):
    x_values = np.linspace(x_range[0], x_range[1], num_points)
    y_values = y_range[1] * np.sin(frequency * x_values / x_range[1])
    angles = np.degrees(np.arctan2(np.gradient(y_values), np.gradient(x_values)))
    return list(zip(x_values, y_values, angles))

# Function to save the trajectory to a CSV file
def save_trajectory_to_csv(points, file_name):
    with open(file_name, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['x', 'y', 'angle'])
        csvwriter.writerows(points)

# Function to plot the trajectory and vehicle
def plot_trajectory_with_vehicle(points):
    fig, ax = plt.subplots()
    
    # Plot the trajectory points
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]
    ax.plot(x_coords, y_coords, marker='o', linestyle='-', color='b')
    
    # Plot the vehicle as a small rectangle at each point
    vehicle_length = 2
    vehicle_width = 1
    for x, y, angle in points:
        rect = patches.Rectangle(
            (x - vehicle_length / 2, y - vehicle_width / 2),
            vehicle_length,
            vehicle_width,
            angle=angle,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(rect)
    
    ax.set_aspect('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Trajectory with Vehicle Representation')
    plt.show()

# Parameters for the trajectory
x_range = (0, 90)
y_range = (-50, 50)
frequency = 7
num_points = 15

# Generate the trajectory
trajectory_points = generate_trajectory(x_range, y_range, frequency, num_points)

# Save the trajectory to a CSV file
save_trajectory_to_csv(trajectory_points, 'curvy_trajectory.csv')

# Plot the trajectory with vehicle representation
plot_trajectory_with_vehicle(trajectory_points)

