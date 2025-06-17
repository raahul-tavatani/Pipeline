import subprocess
import time
import os
import signal
import psutil

CARLA_PATH = r"C:\carla\Build\UE4Carla\0.9.15-305-g6fc02a550-dirty\WindowsNoEditor\CarlaUE4.exe"
TEST_DIR = r"C:\Pipeline\Test_cases"

# Carla processes
carla_2000 = None
carla_3000 = None

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
                print(f"✅ Killed process on port {port}")
            except Exception as e:
                print(f"⚠️  Could not kill process on port {port}: {e}")

def main():
    global carla_2000, carla_3000

    print("🚀 Launching Carla servers...")
    carla_2000 = launch_carla(2000)
    carla_3000 = launch_carla(3000)

    print("⏳ Waiting for Carla to initialize...")
    time.sleep(20)

    print("▶️ Starting test scripts in parallel...")

    long_proc = subprocess.Popen(["python", os.path.join(TEST_DIR, "Longitudinal_case.py")])
    azim_proc = subprocess.Popen(["python", os.path.join(TEST_DIR, "Azimutal_case.py")])

    long_proc.wait()
    azim_proc.wait()

    print("✅ Test scripts completed.")
    print("🧹 Shutting down Carla servers...")

    # Kill Carla processes by port
    kill_process_on_port(2000)
    kill_process_on_port(3000)

    print("🎉 All done.")

if __name__ == "__main__":
    main()
