import numpy as np
import open3d as o3d
import carla
import time
import matplotlib.pyplot as plt
from queue import Queue, Empty
import os
import json 

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
client.load_world('Lidar_Testing_Ground')


# def draw_axes(world, location=carla.Location(0, -150, 0.5), axis_length=2.0, duration=0):
#     debug = world.debug
#     debug.draw_line(location, location + carla.Location(x=axis_length), 0.1, carla.Color(255, 0, 0), duration)
#     debug.draw_line(location, location + carla.Location(y=axis_length), 0.1, carla.Color(0, 255, 0), duration)
#     debug.draw_line(location, location + carla.Location(z=axis_length), 0.1, carla.Color(0, 0, 255), duration)


def spawn_lidar(world, blueprint_library, location):
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('horizontal_fov', '360')
    lidar_bp.set_attribute('sensor_tick', '0.1')
    lidar_bp.set_attribute('range', '60')
    lidar_bp.set_attribute('channels', '64')
    lidar_bp.set_attribute('points_per_second', '1300000')
    lidar_bp.set_attribute('rotation_frequency', '10')
    lidar_bp.set_attribute('upper_fov', '7')
    lidar_bp.set_attribute('lower_fov', '-16')
    transform = carla.Transform(location, carla.Rotation(pitch=0, yaw=0, roll=0))
    lidar = world.spawn_actor(lidar_bp, transform)
    if lidar is not None:
        print(" LiDAR sensor spawned successfully.")
    else:
        print(" Failed to spawn LiDAR sensor.")
    return lidar



def spawn_vehicle_at(world, blueprint_library,lidar_location, x, y, z=0, yaw=0):
    vehicle_bp = blueprint_library.filter("vehicle")[0]
    location = carla.Location(
        x=lidar_location.x + x,
        y=lidar_location.y + y,
        z=0
    )
    rotation = carla.Rotation(pitch=0, yaw=yaw, roll=0)
    transform = carla.Transform(location, rotation)

    vehicle = world.spawn_actor(vehicle_bp, transform)
    if vehicle:
        print(f" Vehicle spawned at ({x}, {y}, {z}) with yaw={yaw}")
        return vehicle
    else:
        print(f" Failed to spawn vehicle at ({x}, {y}, {z})")
        return None



def save_bounding_boxes_json(vehicles, save_dir, lidar_location):
    bounding_boxes = []

    for vehicle in vehicles:
        bb = vehicle.bounding_box
        vehicle_transform = vehicle.get_transform()

        # Convert vehicle world position to LiDAR-relative
        center_world = vehicle_transform.location
        center_lidar = {
            "x": center_world.x - lidar_location.x,
            "y": center_world.y - lidar_location.y,
            "z": center_world.z - lidar_location.z
        }

        # Bounding box center (offset from vehicle center in vehicle local frame)
        bb_offset = {
            "x": bb.location.x,
            "y": bb.location.y,
            "z": bb.location.z
        }

        bb_dict = {
            "id": vehicle.id,
            "type_id": vehicle.type_id,
            "center_lidar": center_lidar,
            "extent": {
                "x": bb.extent.x,
                "y": bb.extent.y,
                "z": bb.extent.z
            },
            "rotation": {
                "pitch": vehicle_transform.rotation.pitch,
                "yaw": vehicle_transform.rotation.yaw,
                "roll": vehicle_transform.rotation.roll
            },
            "bounding_box_offset": bb_offset
        }

        bounding_boxes.append(bb_dict)

    json_path = os.path.join(save_dir, "Ground_truth.json")
    with open(json_path, 'w') as f:
        json.dump(bounding_boxes, f, indent=4)
    print(f" Saved LiDAR-relative bounding boxes to: {json_path}")


def lidar_callback(point_cloud, sensor_queue, save_path="C:\\Pipeline\\saved_data\\single_frame.pcd"):
    points = np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape(-1, 4)
    xyz_points = points[:, :3]
    print(f"📡 LiDAR callback: received {xyz_points.shape[0]} points")


    if xyz_points.shape[0] > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_points)
        abs_path = os.path.abspath(save_path)
        o3d.io.write_point_cloud(abs_path, pcd)
        print(f" Saved point cloud to: {abs_path}")
    else:
        print("No points to save.")

    sensor_queue.put(point_cloud)


def visualize_pcd(pcd_path="C:\\Pipeline\\saved_data\\single_frame.pcd"):
    abs_path = os.path.abspath(pcd_path)
    if not os.path.exists(abs_path):
        print(f"File not found: {abs_path}")
        return

    print(f" Visualizing: {abs_path}")
    pcd = o3d.io.read_point_cloud(abs_path)
    points = np.asarray(pcd.points)
    if len(points) == 0:
        print("No points found in point cloud.")
        return

    distances = np.linalg.norm(points, axis=1)
    distances_normalized = (distances - distances.min()) / (distances.max() - distances.min() + 1e-8)
    cmap = plt.get_cmap("viridis")
    colors = cmap(distances_normalized)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

def main():
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    original_settings = world.get_settings()
    lidar = None
    vehicles = []

    try:
        # Apply synchronous simulation settings
        settings = carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=0.1)
        world.apply_settings(settings)

        sensor_queue = Queue()
        lidar_location = carla.Location(x=-150, y=0, z=1.7)

        # Spawn LiDAR at the origin
        lidar = spawn_lidar(world, blueprint_library, lidar_location)
        if lidar is None:
            print("Sensor creation failed. Exiting.")
            return

        # Prepare save directory
        save_dir = os.path.abspath("C:\\Pipeline\\saved_data")
        os.makedirs(save_dir, exist_ok=True)
        pcd_path = os.path.join(save_dir, "single_frame.pcd")

        # Attach LiDAR callback
        lidar.listen(lambda data: lidar_callback(data, sensor_queue, save_path=pcd_path))

        # Define 4 spawn points around the LiDAR
        vehicle_positions = [
            (10, 5),     # RT
            #(-15, 15),    # LT
            #(-10, 0),     # LB
            (10, -5)     # Back
        ]

        # Spawn vehicles at those locations
        for x, y in vehicle_positions:
            vehicle = spawn_vehicle_at(world, blueprint_library,lidar_location, x, y)
            if vehicle:
                vehicles.append(vehicle)

        # Let the LiDAR collect some data
        print("⏳ Collecting LiDAR frames...")
        for _ in range(5):
            world.tick()
            time.sleep(0.1)

        # Save point cloud and bounding boxes
        try:
            _ = sensor_queue.get(timeout=3.0)
            print("LiDAR frame received and saved.")
            save_bounding_boxes_json(vehicles, save_dir, lidar_location)
        except Empty:
            print("Timeout: No data received from LiDAR sensor.")

    finally:
        print("🧹 Cleaning up...")
        if lidar:
            lidar.stop()
            lidar.destroy()
        for v in vehicles:
            v.destroy()
        world.apply_settings(original_settings)

    #visualize_pcd(pcd_path)



if __name__ == "__main__":
    main()
