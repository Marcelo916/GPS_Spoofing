
import airsim
import time
import random
import numpy as np
from datetime import datetime
from pathlib import Path

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()
time.sleep(2)

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
output_file = Path(f"airsim_spoofed_labeled_{timestamp}.txt")
output_file.parent.mkdir(parents=True, exist_ok=True)
print(f"📄 Logging to {output_file}")

with open(output_file, "w") as f:
    f.write("TIME,POS_X,POS_Y,POS_Z,VEL_X,VEL_Y,VEL_Z,LABEL,SPOOF_TYPE\n")

    for i in range(240):
        speed = random.uniform(1, 4)
        dx, dy, dz = random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-0.3, 0.3)
        length = (dx**2 + dy**2 + dz**2) ** 0.5
        dx, dy, dz = (dx/length, dy/length, dz/length) if length != 0 else (1, 0, 0)
        vx, vy, vz = speed * dx, speed * dy, speed * dz

        client.moveByVelocityAsync(vx, vy, vz, 1).join()

        kinematics = client.getMultirotorState().kinematics_estimated
        pos = kinematics.position
        vel = kinematics.linear_velocity

        pos_x, pos_y, pos_z = pos.x_val, pos.y_val, pos.z_val
        vel_x, vel_y, vel_z = vel.x_val, vel.y_val, vel.z_val
        label = 0
        spoof_type = "none"

        # Inject spoofing into logged data
        if 20 <= i < 25 or 210 <= i < 215:
            pos_x += random.uniform(5, 10)
            pos_y += random.uniform(5, 10)
            label = 1
            spoof_type = "pos"
        elif 40 <= i < 45 or 100 <= i < 105:
            vel_x += random.uniform(-5, 5)
            vel_y += random.uniform(-5, 5)
            label = 1
            spoof_type = "vel"

        current_time = time.time()
        f.write(f"{current_time:.3f},{pos_x:.3f},{pos_y:.3f},{pos_z:.3f},{vel_x:.3f},{vel_y:.3f},{vel_z:.3f},{label},{spoof_type}\n")
        time.sleep(0.1)

client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
print("✅ Data collection complete.")
