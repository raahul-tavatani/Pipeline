import subprocess
import time
import os
import sys
import psutil

CARLA_PATH = r"C:\carla\Build\UE4Carla\0.9.15-305-g6fc02a550-dirty\WindowsNoEditor\CarlaUE4.exe"
TEST_DIR = r"C:\Pipeline\Test_cases"
CONVERT_SCRIPT = r"C:\Pipeline\Scripts\convert_pcd_to_kitti_bin.py"


def launch_carla(port):
    return subprocess.Popen([
        CARLA_PATH,
        f"-carla-rpc-port={port}",
        "-windowed", "-ResX=800", "-ResY=600"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def kill_process_on_port(port):
    for conn in psutil.net_connections(kind='inet'):
        if conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
            try:
                psutil.Process(conn.pid).kill()
                print(f"Killed process on port {port}")
            except Exception as e:
                print(f"Could not kill process on port {port}: {e}")

def run_test_scripts():
    print("Starting test scripts in parallel...")

    long_proc = subprocess.Popen([sys.executable, os.path.join(TEST_DIR, "Longitudinal_case.py")])
    azim_proc = subprocess.Popen([sys.executable, os.path.join(TEST_DIR, "Azimutal_case.py")])
    traj_proc = subprocess.Popen([sys.executable, os.path.join(TEST_DIR, "trajectory_case.py")])

    # Wait for all to finish
    long_proc.wait()
    azim_proc.wait()
    traj_proc.wait()

    print("Test scripts completed.")

def run_pcd_to_bin_conversion():
    if not os.path.isfile(CONVERT_SCRIPT):
        print(f"Conversion script not found: {CONVERT_SCRIPT}")
        return

    print("Running PCD to BIN conversion script...")
    proc = subprocess.run([sys.executable, CONVERT_SCRIPT])
    if proc.returncode == 0:
        print("PCD to BIN conversion completed successfully.")
    else:
        print("PCD to BIN conversion failed.")

def main():
    print("Launching Carla servers...")
    carla_2000 = launch_carla(2000)
    carla_3000 = launch_carla(3000)
    carla_4000 = launch_carla(4000)

    print("⏳ Waiting for Carla to initialize...")
    time.sleep(20)

    try:
        run_test_scripts()
        run_pcd_to_bin_conversion()
    finally:
        print("Shutting down Carla servers...")
        kill_process_on_port(2000)
        kill_process_on_port(3000)
        kill_process_on_port(4000)

        print("🎉 All done.")

if __name__ == "__main__":
    main()
