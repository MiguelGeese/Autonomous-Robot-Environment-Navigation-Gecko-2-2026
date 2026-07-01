# OS imports
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Imports
import mujoco
import numpy as np
import csv
from datetime import datetime

# ARIEL specific imports
from ariel.simulation.environments import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.utils.video_recorder import VideoRecorder

# Brain Loader
def load_brain(mode: str) -> np.ndarray:

    # Get path to brain
    npy_path = f"gecko_brain_{mode.lower()}.npy"
    csv_path = f"gecko_brain_{mode.lower()}.csv"
    
    if os.path.exists(npy_path):
        print(f"Loaded {npy_path}")
        return np.load(npy_path)
    elif os.path.exists(csv_path):
        print(f"Loaded {csv_path}")
        return np.loadtxt(csv_path, delimiter=",", skiprows=1)
    else: 
        raise FileNotFoundError(f"Could not find {npy_path} or {csv_path} in the current directory")

# Alpha calculation
def calculate_alpha(data: mujoco.MjData, target_position: np.ndarray, threshold: float = 1e-8) -> float:
    robot_position = data.geom("robot1_core").xpos[:2]
    vector_to_target = target_position[:2] - robot_position

    dist = np.linalg.norm(vector_to_target)
    if dist < threshold: return 0.0
    to_target = vector_to_target / dist

    xmat = data.geom("robot1_core").xmat.reshape(3, 3)
    world_forward_2d = (xmat @ [0.0, -1.0, 0.0])[:2]
    wf_norm = np.linalg.norm(world_forward_2d)

    if wf_norm < threshold: return 0.0
    world_forward_2d /= wf_norm

    dot = np.dot(world_forward_2d, to_target)
    cross = world_forward_2d[0] * to_target[1] - world_forward_2d[1] * to_target[0]
    return float(np.arctan2(cross, dot))

# Scenario recorder
def record_scenario(scenario_name: str, target_loc: list, brains: dict, videos_dir: str, telemetry_dir: str):
    print(f"Setting up scenario: {scenario_name} | Target: {target_loc}")
    
    # Setup world
    world = SimpleFlatWorld()
    box = mujoco.MjSpec()
    cube = box.worldbody.add_body(name="cube")
    cube.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.05, 0.05, 0.05), rgba=(0.8, 0.2, 0.2, 1.0), name="target_cube")
    
    # Spawn robot and box
    robot = gecko()
    world.spawn(robot.spec, position=[0, 0, 0.2])
    world.spawn(box, position=[target_loc[0], target_loc[1], 0.2])

    # Add camera to gecko
    for body in world.spec.bodies:
        if body.name == "robot1_core":
            body.add_camera(name="gecko_vision", pos=[0.0, -0.15, 0.05], xyaxes=[-1.0, 0.0, 0.0, 0.0, -0.3, 0.95], fovy=120.0)
            break

    # Setup video recording
    video_model = world.spec.compile()
    video_data = mujoco.MjData(video_model)
    
    video_model.vis.global_.offwidth = 640
    video_model.vis.global_.offheight = 480
    video_renderer = mujoco.Renderer(video_model, height=480, width=640)
    video = VideoRecorder(file_name=f"gecko_fsm_{scenario_name}", output_folder=videos_dir)
    
    # Setup vision
    eye_renderer = mujoco.Renderer(video_model, height=64, width=128)
    box_id = video_model.geom("robot2_target_cube").id
    
    # Initialize recording parameters
    step = 0
    frame_count = 0
    fps = 30
    runtime = 240.0 
    
    # Set default state
    current_state = 'SEARCH'
    search_direction = 'RIGHT' 
    vision_memory = 0 

    fsm_thesis_data = [["Time_sec", "X_Pos", "Y_Pos", "Active_State", "Raw_Pixels_Seen", "Has_Vision_Memory", "True_Alpha"]]
    mujoco.mj_forward(video_model, video_data)

    # Start recording
    while video_data.time < runtime:

        # Get alpha and current distance to the box (For theoretical knowledge)
        true_alpha = calculate_alpha(video_data, np.array([target_loc[0], target_loc[1], 0.2]))
        current_dist = float(np.linalg.norm(video_data.geom_xpos[box_id][0:2] - video_data.qpos[0:2]))
        
        # Update vision
        eye_renderer.update_scene(video_data, camera="gecko_vision")
        pixels = eye_renderer.render().astype(np.int32)
        
        # Extract seen colours
        r = pixels[:, :, 0]
        g = pixels[:, :, 1]
        b = pixels[:, :, 2]
        
        # Ensure red is the dominant colour over green and blue
        red_mask = (r > 80) & (r > g + 30) & (r > b + 30)

        # Boolean for checking if the robot sees the box
        sees_raw_pixels = bool(np.any(red_mask))

        # Prevent robot from deviating motion because of gait
        if sees_raw_pixels: 
            vision_memory = 30 
        else: 
            vision_memory = max(0, vision_memory - 1)
        has_confident_vision = (vision_memory > 0)

        if current_dist < 0.09:
            current_state = 'STOP'
            steering_signal = 0.0
            
        # Determine state for FSM
        elif has_confident_vision:
            if true_alpha > 0:
                search_direction = 'RIGHT' 
            else: 
                search_direction = 'LEFT'

            if current_state == 'SEARCH':
                if true_alpha > 0:
                    current_state = 'RIGHT'
                else:
                    current_state = 'LEFT'

            elif current_state == 'FORWARD':
                if true_alpha > 0.3: 
                    current_state = 'RIGHT'
                elif true_alpha < -0.3: 
                    current_state = 'LEFT'

            else: 
                if abs(true_alpha) < 0.1: 
                    current_state = 'FORWARD'
                elif true_alpha > 0.3: 
                    current_state = 'RIGHT'
                elif true_alpha < -0.3: 
                    current_state = 'LEFT'
            
            # Choose which state to use from the FSM: || FORWARD | LEFT | RIGHT ||
            active_weights = brains[current_state]
            if current_state == 'FORWARD': 
                steering_signal = 0.0
            else: 
                steering_signal = np.clip(true_alpha / (np.pi/2), -1.0, 1.0)
                
        # Search state actions
        else:
            current_state = 'SEARCH'
            if search_direction == 'RIGHT':
                active_weights = brains['RIGHT']
                steering_signal = 1.0 
            else:
                active_weights = brains['LEFT']
                steering_signal = -1.0 

        # Decide robot gait based on matrix values
        if current_state != 'STOP':
            cycle_time = video_data.time % 2.0
            steering_mask = np.array([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0])

            # Two phase CPG movement
            if cycle_time < 1.0:
                base_gait = active_weights[:, 0]
                steering = active_weights[:, 2] * steering_signal * steering_mask
            else:
                base_gait = active_weights[:, 1]
                steering = active_weights[:, 3] * steering_signal * steering_mask
                
            commands = base_gait + steering
            video_data.ctrl[:] = np.clip(commands, -1.5, 1.5)
        
        mujoco.mj_step(video_model, video_data)

        # Data collection
        if step % int(1.0/(video_model.opt.timestep * fps)) == 0:
            fsm_thesis_data.append([
                round(video_data.time, 3), 
                round(video_data.qpos[0], 4), 
                round(video_data.qpos[1], 4), 
                current_state, 
                sees_raw_pixels,
                has_confident_vision,
                round(true_alpha, 4)
            ])

        step += 1
        
        # Robot stops moving when target is reached
        if current_dist < 0.09: video_data.ctrl[:] = np.zeros(8)
            
        if frame_count < video_data.time * fps:
            video_renderer.update_scene(video_data)
            video.write(frame=video_renderer.render())
            frame_count += 1
            
        if current_state == 'STOP': break

    video.release()
    video_renderer.close()
    eye_renderer.close() 

    # Save run data
    csv_filename = os.path.join(telemetry_dir, f"fsm_telemetry_{scenario_name}.csv")
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(fsm_thesis_data)
    print(f"Finished recording: gecko_fsm_{scenario_name}.mp4 and saved telemetry data.\n")

# MAIN EXECUTION LOOP
def main():
    # Directory creation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join("run_results", f"playback_CV_{timestamp}")
    
    videos_dir = os.path.join(base_dir, "videos")
    telemetry_dir = os.path.join(base_dir, "telemetry")

    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(telemetry_dir, exist_ok=True)
    
    print(f"Initializing new run: {base_dir}")
    
    print("Loading trained brains for video recording...")
    try:
        brains = {
            'FORWARD': load_brain('FORWARD'),
            'LEFT':    load_brain('LEFT'),
            'RIGHT':   load_brain('RIGHT')
        }
    except Exception as e:
        print(f"\n{e}")
        return

    scenarios = {
        "forward": [0.0, -2.0],  
        "left":    [-2.0, -1.5], 
        "right":   [2.0, -1.5],  
        "behind":  [0.0, 2.0]    
    }

    print("Starting Recording...")
    for name, loc in scenarios.items():
        record_scenario(name, loc, brains, videos_dir, telemetry_dir)

    print(f"\nAll files have been organized into: {base_dir}")

if __name__ == "__main__":
    main()