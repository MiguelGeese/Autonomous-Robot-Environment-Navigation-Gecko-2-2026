# OS and Math imports
import os
import time
import math
import numpy as np
import cv2
import subprocess
import csv
import matplotlib.pyplot as plt

# Robohat imports
try:
    from robohatlib.Robohat import Robohat
    from testlib import TestConfig
except ImportError:
    print("Failed to import Robohatlib or TestConfig.")
    exit()

# Configuration
ACTIVE_PINS = [0, 1, 2, 3, 4, 5, 6, 7]

# Camera HSV threshold for the orange/red cone
LOWER_ORANGE = np.array([2, 80, 50])
UPPER_ORANGE = np.array([25, 255, 255])

# Brain Loader

def load_brain(mode: str) -> np.ndarray:
    npy_path = f"gecko_brain_{mode.lower()}.npy"
    csv_path = f"gecko_brain_{mode.lower()}.csv"
    
    if os.path.exists(npy_path):
        print(f"Loaded {npy_path}")
        return np.load(npy_path)
    elif os.path.exists(csv_path):
        print(f"Loaded {csv_path}")
        return np.loadtxt(csv_path, delimiter=",", skiprows=1)
    else: 
        raise FileNotFoundError(f"Could not find {npy_path} or {csv_path}!")

# Main loop

def main():
    print("STARTING GECKO AUTONOMOUS HUNTER FSM")

    
    # Load the trained brains
    print("Loading Neural Networks...")
    try:
        brains = {
            'FORWARD': load_brain('FORWARD'),
            'LEFT':    load_brain('LEFT'),
            'RIGHT':   load_brain('RIGHT')
        }
    except Exception as e:
        print(e)
        return

    # Wake up the camera
    print("Waking up camera...")
    cmd = [
        "rpicam-vid",
        "-t", "0",            # Stream forever
        "--inline",           # Force inline headers
        "--width", "160",     # Width for speed
        "--height", "120",    # Height for speed
        "--framerate", "30",  # 30 fps
        "--codec", "yuv420",  # Raw uncompressed data
        "-o", "-"             # Output to stdout
    ]
    
    try:
        camera_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except Exception as e:
         print(f"Failed to start rpicam-vid. {e}")
         return

    time.sleep(2.0)
    
    # Check if camera died on startup
    if camera_process.poll() is not None:
        print("CRITICAL: Camera process died on startup. Check ribbon cable!")
        return

    # Wake up hardware
    print("Initializing Robohat...")
    robohat = Robohat(
        TestConfig.SERVOASSEMBLY_1_CONFIG, 
        TestConfig.SERVOASSEMBLY_2_CONFIG, 
        TestConfig.TOPBOARD_ID_SWITCH
    )
    robohat.init(TestConfig.SERVOBOARD_1_DATAS_LIST, TestConfig.SERVOBOARD_2_DATAS_LIST)
    robohat.do_buzzer_beep()
    robohat.start_servo_drivers()
    robohat.set_servo_direct_mode(True) 

    # Send limbs to neutral
    for pin in ACTIVE_PINS:
        robohat.set_servo_single_angle(pin, 90.0)
    time.sleep(1.0)

    # FSM Variables
    frame_bytes = 160 * 120 * 3 // 2 
    current_state = 'SEARCH'
    search_direction = 'RIGHT'
    vision_memory = 0
    steering_mask = np.array([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
    
    print("\nRunning, PRESS CTRL+C TO FORCE QUIT.")
    start_time = time.time()

    # Flush the tinted startup frames from the camera
    for _ in range(15):
        camera_process.stdout.read(frame_bytes)

    try:
        thesis_data = []
        print("\n>>> Running, PRESS CTRL+C TO FORCE QUIT")
        start_time = time.time()
        last_frame_time = start_time
        
        while True:
            # Vision processing
            raw_data = camera_process.stdout.read(frame_bytes)
            if len(raw_data) != frame_bytes:
                print("Camera buffer empty or corrupted")
                time.sleep(0.1)
                continue
                
            yuv_frame = np.frombuffer(raw_data, dtype=np.uint8).reshape((120 * 3 // 2, 160))
            frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)
            
            sections = np.array_split(mask, 3, axis=1)
            left_val = cv2.countNonZero(sections[0]) / sections[0].size
            center_val = cv2.countNonZero(sections[1]) / sections[1].size
            right_val = cv2.countNonZero(sections[2]) / sections[2].size
            total_orange = (left_val + center_val + right_val) / 3.0
            
            # FSM Logic
            if total_orange > 0.25:
                current_state = 'STOP'
            elif total_orange > 0.02: 
                vision_memory = 15 
                if center_val > left_val and center_val > right_val:
                    current_state = 'FORWARD'
                elif left_val > right_val:
                    current_state = 'LEFT'
                    search_direction = 'LEFT' 
                else:
                    current_state = 'RIGHT'
                    search_direction = 'RIGHT'
            else:
                vision_memory = max(0, vision_memory - 1)
                if vision_memory == 0:
                    current_state = 'SEARCH'
                    
            print(f"Current state is: {current_state} (Orange: {total_orange:.2f})")

            # Data collection
            current_time = time.time()
            loop_duration = current_time - last_frame_time
            last_frame_time = current_time
            
            thesis_data.append([
                round(current_time - start_time, 3), # Seconds since start
                round(loop_duration, 4),             # Time taken for this one frame
                round(total_orange, 3),              # Total size of cone
                round(left_val, 3),                  # Left area
                round(center_val, 3),                # Center area
                round(right_val, 3),                 # Right area
                current_state                        # FSM Decision
            ])

            # Break if stopped
            if current_state == 'STOP':
                print("\n[!] TARGET ACQUIRED! STOPPING.")
                break

            # Motion generation
            if current_state == 'SEARCH':
                active_weights = brains[search_direction]
                steering_signal = 1.0 if search_direction == 'RIGHT' else -1.0
            else:
                active_weights = brains[current_state]
                if current_state == 'FORWARD': steering_signal = 0.0
                elif current_state == 'LEFT': steering_signal = -1.0
                elif current_state == 'RIGHT': steering_signal = 1.0

            elapsed = time.time() - start_time
            cycle_time = elapsed % 2.0
            
            if cycle_time < 1.0:
                base_gait = active_weights[:, 0]
                steering = active_weights[:, 2] * steering_signal * steering_mask
            else:
                base_gait = active_weights[:, 1]
                steering = active_weights[:, 3] * steering_signal * steering_mask
                
            commands_rads = np.clip(base_gait + steering, -1.5, 1.5)

            # Execute on hardware
            for i, rad_val in enumerate(commands_rads):
                physical_pin = ACTIVE_PINS[i]
                degree_val = (rad_val * (180.0 / math.pi)) + 90.0
                safe_angle = np.clip(degree_val, 20.0, 160.0)
                robohat.set_servo_single_angle(physical_pin, safe_angle)
                
    except KeyboardInterrupt:
        print("\nForce Quit detected")
        
    finally:
        print("Relaxing servos and powering down")

        # WRITE DATA TO CSV
        if 'thesis_data' in locals() and len(thesis_data) > 0:
            print(f"Saving {len(thesis_data)} frames of data to CSV...")
            base_filename = f"gecko_thesis_log_{int(time.time())}"
            csv_filename = f"{base_filename}.csv"
            
            with open(csv_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Time_Seconds", "Frame_Delta_Sec", "Total_Orange", "Left_Orange", "Center_Orange", "Right_Orange", "FSM_State"])
                writer.writerows(thesis_data)
            print(f"Data saved to {csv_filename}")
            
            # GENERATE PDF GRAPH
            print("Generating PDF approach graph")
            try:
                times = [row[0] for row in thesis_data]
                orange_vals = [row[2] for row in thesis_data]
                
                # Create plot
                plt.figure(figsize=(10, 5))
                plt.plot(times, orange_vals, label="Target Size (Distance Proxy)", color='darkorange', linewidth=2)
                
                # Format the graph
                plt.title("Gecko Autonomy: Approach Trajectory over Time")
                plt.xlabel("Time (Seconds)")
                plt.ylabel("Cone Size (% of Vision)")
                plt.ylim(0, 0.25)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                
                # Save as PDF
                pdf_filename = f"{base_filename}.pdf"
                plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
                plt.close()
                print(f"Graph safely saved to {pdf_filename}")
            except Exception as e:
                print(f"[!] Failed to generate PDF: {e}")

        for pin in ACTIVE_PINS:
            robohat.set_servo_single_angle(pin, 90.0)
        time.sleep(0.5)
        
        robohat.stop_servo_drivers()
        robohat.exit_program()
        
        if 'camera_process' in locals() and camera_process.poll() is None:
            camera_process.kill()
            
        print("Hardware powered down")

if __name__ == "__main__":
    main()