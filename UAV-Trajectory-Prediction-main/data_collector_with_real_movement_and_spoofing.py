
import airsim
import time
import random
import numpy as np
from datetime import datetime
from pathlib import Path

# === SETUP ===
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()
time.sleep(2)

# === OUTPUT FILE ===
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
output_file = Path(f"airsim_spoofed_full_{timestamp}.txt")
output_file.parent.mkdir(parents=True, exist_ok=True)
print(f"📄 Logging to {output_file}")

with open(output_file, "w") as f:
    f.write("POS_X,POS_Y,POS_Z,VEL_X,VEL_Y,VEL_Z\n")

    print("🚀 Starting random motion and spoofing injection")

    for i in range(240):
        # Create random direction and normalize it
        speed = random.uniform(1, 4)
        dx, dy, dz = random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-0.3, 0.3)
        length = (dx**2 + dy**2 + dz**2) ** 0.5
        dx, dy, dz = (dx/length, dy/length, dz/length) if length != 0 else (1, 0, 0)
        vx, vy, vz = speed * dx, speed * dy, speed * dz

        # Move the drone
        client.moveByVelocityAsync(vx, vy, vz, 1).join()

        # Get real drone state
        kinematics = client.getMultirotorState().kinematics_estimated
        pos = kinematics.position
        vel = kinematics.linear_velocity

        pos_x, pos_y, pos_z = pos.x_val, pos.y_val, pos.z_val
        vel_x, vel_y, vel_z = vel.x_val, vel.y_val, vel.z_val

        # Inject spoofing into the logged data (not actual flight)
        if 20 <= i < 25:
            pos_x += random.uniform(5, 10)
            pos_y += random.uniform(5, 10)
        if 40 <= i < 45:
            vel_x += random.uniform(-5, 5)
            vel_y += random.uniform(-5, 5)
        if 100 <= i < 105:
            vel_x += random.uniform(-5, 5)
            vel_y += random.uniform(-5, 5)
        if 210 <= i < 215:
            pos_x += random.uniform(5, 10)
            pos_y += random.uniform(5, 10)

        # Write spoofed data to file
        f.write(f"{pos_x:.3f},{pos_y:.3f},{pos_z:.3f},{vel_x:.3f},{vel_y:.3f},{vel_z:.3f}\n")

        time.sleep(0.1)

# === LANDING ===
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
print("✅ Data collection complete.")
