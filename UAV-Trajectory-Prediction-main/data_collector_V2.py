import airsim
import time
import random # This line was added by Me
import numpy as np
import cv2

# === SETUP ===
client = airsim.MultirotorClient()
client.confirmConnection()
print("✅ Connected to AirSim")

client.enableApiControl(True)
print("🔧 API control enabled")

client.armDisarm(True)
print("🔫 Drone armed")

# === TEST LIDAR ===
lidar_data = client.getLidarData(vehicle_name="Drone1")
print(f"Lidar points: {len(lidar_data.point_cloud) // 3}")

# === TEST DEPTH CAMERA ===
response = client.simGetImage("0", airsim.ImageType.DepthPerspective)

if response is not None:
    img1d = np.frombuffer(response, dtype=np.uint8)
    img_bgr = cv2.imdecode(img1d, cv2.IMREAD_UNCHANGED)

    if img_bgr is not None:
        print("📷 Depth image shape:", img_bgr.shape)
        cv2.imshow("Depth View", img_bgr)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
    else:
        print("⚠️ Could not decode depth image")
else:
    print("⚠️ No depth image received")

print("🛫 Taking off...")
client.takeoffAsync().join()
time.sleep(2)

# Go to target altitude
target_altitude = -2.5
print(f"📍 Moving to altitude {target_altitude}m")
client.moveToZAsync(target_altitude, 1).join()
time.sleep(1)

# === MAKE DEPTH WINDOW BIGGER (Step 1 of debug tools) ===
cv2.namedWindow("Depth Patch", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Depth Patch", 600, 400)

# === START RECORDING ===
client.startRecording()
print("🎥 Recording started")

depth_stuck_counter = 0
depth_stuck_threshold = 5  # number of times in a row before escape

# === PHASE 1: RANDOM FLIGHT ===
print("🚀 Phase 1: Free random motion")
for i in range(60):
    speed = random.uniform(1, 4)

    dx = random.uniform(-1, 1)
    dy = random.uniform(-1, 1)
    dz = random.uniform(-0.3, 0.3)

    length = (dx**2 + dy**2 + dz**2) ** 0.5
    if length == 0:
        dx, dy, dz = 1, 0, 0
    else:
        dx /= length
        dy /= length
        dz /= length

    vx = speed * dx
    vy = speed * dy
    vz = speed * dz

    # === DEPTH CAMERA OBSTACLE DETECTION ===
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)
    ])
    if responses:
        img_float = responses[0].image_data_float
        height = responses[0].height
        width = responses[0].width
        depth_img = np.array(img_float, dtype=np.float32).reshape(height, width)

        # Focus on a center patch (60×80 window)
        center_patch = depth_img[60:120, 80:160]
        center_patch = np.clip(center_patch, 0, 20)  # Limit values to [0, 20] meters
        avg_depth = np.mean(center_patch)

        # Optional: visualize center patch for debugging
        print(f"Center patch stats: min={np.min(center_patch):.2f}, max={np.max(center_patch):.2f}")
        norm_patch = center_patch - np.min(center_patch)
        norm_patch /= (np.max(norm_patch) + 1e-5)  # Avoid divide-by-zero

        gray_patch = (norm_patch * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(gray_patch, cv2.COLORMAP_JET)

        cv2.imshow("Depth Patch", heatmap)
        cv2.waitKey(1)

        # 🔍 Visual feedback to debug average depth and stuck counter
        print(f"🔍 Center patch avg depth: {avg_depth:.2f}m — Stuck counter: {depth_stuck_counter}")

        # Enhanced depth obstacle detection
        close_pixels = np.sum(center_patch < 1.0)
        if close_pixels > 10 or np.min(center_patch) < 0.5:
            print("🚨 Close object pixels detected! [via pixel check]")
            depth_stuck_counter += 1

        if avg_depth < 3.0:
            print(f"⚠️ Depth camera detected obstacle at step {i + 1} (avg distance: {avg_depth:.2f}m)")
            depth_stuck_counter += 1

            if depth_stuck_counter >= depth_stuck_threshold:
                print("🚨 Stuck too long! Forcing escape maneuver ⬆️")
                client.moveByVelocityAsync(1, -1, -1, 1).join()  # Move up to escape
                depth_stuck_counter = 0
            else:
                client.hoverAsync().join()
                time.sleep(1)
                continue
        else:
            depth_stuck_counter = 0



    # === LIDAR OBSTACLE DETECTION + AVOIDANCE ===
    lidar_data = client.getLidarData(vehicle_name="Drone1")
    points = lidar_data.point_cloud

    danger_front = False
    path_left_clear = True
    path_up_clear = True
    path_back_clear = True
    path_upfront_blocked = False  # new

    if points:
        for j in range(0, len(points), 3):
            x = points[j]
            y = points[j + 1]
            z = points[j + 2]

            # FRONT — expanded closer range
            if 0.2 <= x <= 5 and -2 <= y <= 2 and -2 <= z <= 2:
                danger_front = True

            # LEFT
            if 0.2 <= x <= 3 and -5 <= y <= -2 and -2 <= z <= 2:
                path_left_clear = False

            # ABOVE
            if 0.5 <= x <= 3 and -2 <= y <= 2 and -5 <= z <= -2.5:
                path_up_clear = False

            # UP+FRONT (for roofs)
            if 0.5 <= x <= 3 and -2 <= y <= 2 and -5 <= z <= -1:
                path_upfront_blocked = True

            # BACK
            if -5 <= x <= -0.2 and -2 <= y <= 2 and -2 <= z <= 2:
                path_back_clear = False


    if danger_front or path_upfront_blocked:
        print(f"🚨 Obstacle detected (front or above-front) at step {i + 1}")
        if path_left_clear:
            print("↩️ Dodging LEFT")
            client.moveByVelocityAsync(0, -2, 0, 1).join()
        elif path_up_clear:
            print("⬆️ Dodging UP")
            client.moveByVelocityAsync(0, 0, -2, 1).join()
        elif path_back_clear:
            print("↩️ Emergency reverse")
            client.moveByVelocityAsync(-2, 0, 0, 1).join()
        else:
            print("🛑 All paths blocked — hovering")
            client.hoverAsync().join()
            time.sleep(2)
            break
    else:
        client.moveByVelocityAsync(vx, vy, vz, 1).join()
        print(f"Free Step {i + 1}/60 → Speed: {speed:.2f}, vx: {vx:.2f}, vy: {vy:.2f}, vz: {vz:.2f}")
        time.sleep(0.1)


# === PHASE 2: STRUCTURED SQUARE LOOP ===
print("🔁 Phase 2: Square path")

side_length = 10  # meters
speed = 2         # m/s
duration = side_length / speed

# Move in a square: +X → +Y → -X → -Y
square_directions = [
    ( speed,  0),  # +X
    ( 0,  speed),  # +Y
    (-speed,  0),  # -X
    ( 0, -speed)   # -Y
]

for i, (vx, vy) in enumerate(square_directions * 2):  # do two full loops
    client.moveByVelocityAsync(vx, vy, 0, duration).join()
    print(f"Square Leg {i + 1}/8 → vx: {vx:.2f}, vy: {vy:.2f}")
    time.sleep(0.2)

# === STOP ===
client.stopRecording()
print("🛑 Recording stopped")

print("🛬 Landing...")
client.landAsync().join()

client.armDisarm(False)
print("🔒 Drone disarmed")

client.enableApiControl(False)
print("🔌 API control released")