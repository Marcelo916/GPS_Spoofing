import sys
from pathlib import Path
import airsim
import time
import os
import shutil
import csv
import numpy as np
import random

# === Setup & connect ===
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# === Takeoff & climb ===
print("Taking off…")
client.takeoffAsync().join()
time.sleep(2)
target_altitude = -30
print(f"Climbing to {target_altitude} m")
client.moveToZAsync(target_altitude, 1).join()
time.sleep(1)

# === Start recording ===
client.startRecording()
print("Recording started")

# === Logging setup ===
ROOT_DIR     = Path(__file__).resolve().parent
target_data  = ROOT_DIR / "raw_data"
txt_log_path = target_data / "airsim_rec.txt"
images_dir   = target_data / "images"
lidar_dir    = target_data / "lidar"
os.makedirs(target_data, exist_ok=True)
os.makedirs(images_dir,   exist_ok=True)
os.makedirs(lidar_dir, exist_ok=True)
if txt_log_path.exists():
    txt_log_path.unlink()

# === Flight & logging parameters ===
sample_interval = 0.1         # 10 Hz
total_time      = 90 * 60     # 90 minutes
steps           = int(total_time / sample_interval)
speed           = 3           # m/s
step_dist       = speed * sample_interval

# === Pattern definitions ===
def straight(i, st):
    return st["x"] + step_dist, st["y"], st["z"]

def circle(i, st):
    θ = 2*np.pi * (i % int(60/sample_interval)) / (60/sample_interval)
    return st["cx"] + 10*np.cos(θ), st["cy"] + 10*np.sin(θ), st["z"]

def figure_eight(i, st):
    t = 2*np.pi * (i % int(120/sample_interval)) / (120/sample_interval)
    return st["cx"] + 10*np.sin(t), st["cy"] + 10*np.sin(t)*np.cos(t), st["z"]

def vertical_wiggle(i, st):
    t = 2*np.pi * (i % int(30/sample_interval)) / (30/sample_interval)
    return st["x"], st["y"], st["z_base"] + 5*np.sin(t)

patterns = [
    ("straight",      straight,      {"x":0.0,"y":0.0,"z":target_altitude}),
    ("circle",        circle,        {"cx":0.0,"cy":0.0,"z":target_altitude}),
    ("figure_eight",  figure_eight,  {"cx":0.0,"cy":0.0,"z":target_altitude}),
    ("vertical_wiggle", vertical_wiggle, {"x":0.0,"y":0.0,"z_base":target_altitude})
]

# === Open CSV ===
with open(txt_log_path, "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow([
        "TimeStamp","POS_X","POS_Y","POS_Z",
        "VEL_X","VEL_Y","VEL_Z",
        "LiDARFile"
    ])

    segment_length = int((5*60) / sample_interval)
    state = {"x":0.0,"y":0.0,"z":target_altitude,
             "cx":0.0,"cy":0.0,"z_base":target_altitude}
    name, func, params = random.choice(patterns)
    state.update(params)
    print(f"Pattern → {name}")

    for i in range(steps):
        if i and i % segment_length == 0:
            name, func, params = random.choice(patterns)
            state.update(params)
            print(f"→ Switching to pattern: {name}")

        x, y, z = func(i, state)
        state["x"], state["y"], state["z"] = x, y, z

        client.moveToPositionAsync(x, y, z, speed).join()
        s = client.getMultirotorState()
        p = s.kinematics_estimated.position
        v = s.kinematics_estimated.linear_velocity
        # get LiDAR
        lidar_data = client.getLidarData("LidarSensor1", "Drone1")
        pts = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1,3)
        # sample or pad to 1024
        if len(pts) > 1024:
            idx = np.random.choice(len(pts), 1024, replace=False)
            pts = pts[idx]
        else:
            pad = np.zeros((1024 - len(pts),3),dtype=np.float32)
            pts = np.vstack([pts, pad])
        lidar_file = lidar_dir / f"lidar_{i:06d}.npy"
        np.save(lidar_file, pts)

        writer.writerow([
            int(time.time()*1e6),
            p.x_val, p.y_val, p.z_val,
            v.x_val, v.y_val, v.z_val,
            lidar_file.name
        ])

        if i % int(1/sample_interval) == 0:
            print(f"Step {i+1}/{steps} recorded")
        time.sleep(sample_interval)

# === Stop & land ===
print("Stopping recording…")
client.stopRecording()
print("Landing…")
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)

# === Cleanup session folder ===
airsim_data_dir = Path("C:/Users/KELLYC40/Documents/AirSim")
time.sleep(3)
if airsim_data_dir.exists():
    sessions = sorted(
        [d for d in airsim_data_dir.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime, reverse=True
    )
    if sessions:
        shutil.rmtree(sessions[0])
        print(f"🧹 Deleted session folder: {sessions[0].name}")

print(f"\n✅ Flight log: {txt_log_path.resolve()}")
