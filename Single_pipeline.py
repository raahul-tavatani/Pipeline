#Single_pipeline.py

import numpy as np
import open3d as o3d
import carla
import time
import matplotlib.pyplot as plt
from queue import Queue, Empty
import os
import json
import argparse

def spawn_lidar(world, blueprint_library, location):
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('horizontal_fov', '120')
    lidar_bp.set_attribute('sensor_tick', '0.1')
    lidar_bp.set_attribute('range', '150')
    lidar_bp.set_attribute('channels', '64')
    lidar_bp.set_attribute('points_per_second', '5000000')
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

def spawn_vehicle_at(world, blueprint_library, lidar_location, x, y, yaw, z=0.4): # z = 0.4 because of the ground
    vehicle_bps = blueprint_library.filter("vehicle.*")
    if not vehicle_bps:
        raise RuntimeError("No vehicle blueprints found.")
    vehicle_bp = vehicle_bps[0]

    location = carla.Location(
        x=lidar_location.x + x,
        y=lidar_location.y + y,
        z= z
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

def save_bounding_boxes_json(vehicles, save_dir, lidar_location, json_filename):
    bounding_boxes = []

    for vehicle in vehicles:
        bb = vehicle.bounding_box
        vehicle_transform = vehicle.get_transform()

        center_world = vehicle_transform.location
        center_lidar = {
            "x": center_world.x - lidar_location.x,
            "y": center_world.y - lidar_location.y,
            "z": center_world.z - lidar_location.z
        }

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

    json_path = os.path.join(save_dir, json_filename)
    with open(json_path, 'w') as f:
        json.dump(bounding_boxes, f, indent=4)
    print(f" Saved LiDAR-relative bounding boxes to: {json_path}")

def lidar_callback(point_cloud, sensor_queue, save_path="C:\\Pipeline\\saved_data\\single_frame.pcd"):
    points = np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape(-1, 4)
    xyz_points = points[:, :3]
    print(f" LiDAR callback: received {xyz_points.shape[0]} points")

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
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description="Carla LiDAR script with arg parser")
    parser.add_argument('--host', type=str, default='localhost', help='Carla server host')
    parser.add_argument('--port', type=int, default=2000, help='Carla server port')
    parser.add_argument('--x_dist_1', type=float, default=15, help='X distance for vehicle 1')
    parser.add_argument('--y_dist_1', type=float, default=-15, help='Y distance for vehicle 1')
    parser.add_argument('--x_dist_2', type=float, help='X distance for vehicle 2')
    parser.add_argument('--y_dist_2', type=float, help='Y distance for vehicle 2')
    parser.add_argument('--yaw_1', type=float, default=0, help='Yaw of vehicle 1')
    parser.add_argument('--yaw_2', type=float, default=0, help='Yaw of vehicle 2')
    parser.add_argument('--save_tag', type=str, default='default_run', help='Tag to name save files')
    parser.add_argument('--test_type', type=str, default='Radial_tests', help='Test type folder name')

    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    client.load_world('Proving_Ground')

    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    original_settings = world.get_settings()
    lidar = None
    vehicles = []
    sensor_queue = Queue()

    try:
        settings = carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=0.1)
        world.apply_settings(settings)

        lidar_location = carla.Location(x=-150, y=0, z=2.1) #Lidar spawns at 1.7, 0.4 m is added because of the world is at (0,0,0.4)

        base_save_dir = r"C:\Pipeline\saved_data"
        save_dir = os.path.join(base_save_dir, args.test_type)
        os.makedirs(save_dir, exist_ok=True)

        # Create ONLY 'pcd' and 'json' directories (no 'json' folder)
        pcd_dir = os.path.join(save_dir, "pcd")
        json_dir = os.path.join(save_dir, "json")
        os.makedirs(pcd_dir, exist_ok=True)
        os.makedirs(json_dir, exist_ok=True)

        pcd_path = os.path.join(pcd_dir, f"{args.save_tag}.pcd")
        json_filename = f"{args.save_tag}.json"

        # Spawn LiDAR sensor
        lidar = spawn_lidar(world, blueprint_library, lidar_location)
        if lidar is None:
            print("Sensor creation failed. Exiting.")
            return

        def one_shot_callback(data):
            lidar_callback(data, sensor_queue, save_path=pcd_path)
            # lidar.stop()  # Optional stop after first frame

        lidar.listen(one_shot_callback)

        vehicle_positions = []
        if args.x_dist_1 is not None and args.y_dist_1 is not None:
            vehicle_positions.append((args.x_dist_1, args.y_dist_1, args.yaw_1))
        if args.x_dist_2 is not None and args.y_dist_2 is not None:
            vehicle_positions.append((args.x_dist_2, args.y_dist_2, args.yaw_2))

        for x, y, yaw in vehicle_positions:
            vehicle = spawn_vehicle_at(world, blueprint_library, lidar_location, x, y, yaw)
            if vehicle:
                vehicles.append(vehicle)

        print("Collecting one LiDAR frame...")
        world.tick()
        time.sleep(0.2)

        try:
            _ = sensor_queue.get(timeout=3.0)
            print("LiDAR frame received and saved.")
            # Save JSON in 'json' folder
            save_bounding_boxes_json(vehicles, json_dir, lidar_location, json_filename)
        except Empty:
            print("Timeout: No data received from LiDAR sensor.")

    finally:
        print("Cleaning up...")
        if lidar:
            lidar.stop()
            lidar.destroy()
        for v in vehicles:
            v.destroy()
        world.apply_settings(original_settings)

    # To visualize point cloud, uncomment below
    # visualize_pcd(pcd_path)


if __name__ == "__main__":
    main()
