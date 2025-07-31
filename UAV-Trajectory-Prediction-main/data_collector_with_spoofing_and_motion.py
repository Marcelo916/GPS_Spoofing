
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
time.sleep(1)

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
output_file = Path(f"airsim_spoofed_motion_{timestamp}.txt")
output_file.parent.mkdir(parents=True, exist_ok=True)

print(f"📄 Logging to {output_file}")

with open(output_file, "w") as f:
    f.write("POS_X,POS_Y,POS_Z,VEL_X,VEL_Y,VEL_Z\n")

    for i in range(300):
        # Generate a small random movement direction
        dx, dy, dz = random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-0.2, 0.2)
        speed = random.uniform(1, 3)
        vx, vy, vz = speed * dx, speed * dy, speed * dz

        # Send movement command for 0.1s
        client.moveByVelocityAsync(vx, vy, vz, 0.1, drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom, yaw_mode=airsim.YawMode(False, 0))

        # Get current state
        kinematics = client.getMultirotorState().kinematics_estimated
        pos = kinematics.position
        vel = kinematics.linear_velocity

        pos_x, pos_y, pos_z = pos.x_val, pos.y_val, pos.z_val
        vel_x, vel_y, vel_z = vel.x_val, vel.y_val, vel.z_val

        # Inject spoofing for logging only
        if 100 <= i < 120:
            pos_x += random.uniform(5, 10)
            pos_y += random.uniform(5, 10)
        elif 200 <= i < 220:
            vel_x += random.uniform(-5, 5)
            vel_y += random.uniform(-5, 5)

        # Save spoofed values to file
        f.write(f"{pos_x:.3f},{pos_y:.3f},{pos_z:.3f},{vel_x:.3f},{vel_y:.3f},{vel_z:.3f}\n")
        time.sleep(0.1)

client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
print("✅ Data collection with motion complete.")
