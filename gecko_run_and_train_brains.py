import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import mujoco
import numpy as np
import nevergrad as ng
import time
import gc
import concurrent.futures
import multiprocessing as mp
import csv
from datetime import datetime

# ARIEL specific imports
from ariel.simulation.environments import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.utils.video_recorder import VideoRecorder

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

# Worker setup

WORKER_MODEL = None
WORKER_DATA = None
BOX_ID = None

def _init_worker():
    global WORKER_MODEL, WORKER_DATA, BOX_ID
    
    world = SimpleFlatWorld()
    box = mujoco.MjSpec()
    cube = box.worldbody.add_body(name="cube")
    cube.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.05, 0.05, 0.05), rgba=(0.8, 0.2, 0.2, 1.0), name="target_cube")
    
    robot = gecko()
    world.spawn(robot.spec, position=[0, 0, 0.2])
    world.spawn(box, position=[0.0, -2.0, 0.2])

    WORKER_MODEL = world.spec.compile()
    WORKER_DATA = mujoco.MjData(WORKER_MODEL)
    BOX_ID = WORKER_MODEL.geom("robot2_target_cube").id

# Robot Evaluation
def evaluate_robot(args):
    weights, mode = args
    local_weights = np.array(weights).reshape((8,4))

    if mode == 'FORWARD': loc = [0.0, -2.0]
    elif mode == 'LEFT':  loc = [-2.0, -2.0]
    elif mode == 'RIGHT': loc = [2.0, -2.0]

    mujoco.mj_resetData(WORKER_MODEL, WORKER_DATA)
    cube_body_id = WORKER_MODEL.geom_bodyid[BOX_ID]
    WORKER_MODEL.body_pos[cube_body_id][0] = loc[0]
    WORKER_MODEL.body_pos[cube_body_id][1] = loc[1]
    mujoco.mj_forward(WORKER_MODEL, WORKER_DATA)

    step = 0
    last_check_pos = np.array(WORKER_DATA.qpos[:2].copy())

    box_pos = WORKER_DATA.geom_xpos[BOX_ID]
    torso_pos = WORKER_DATA.qpos

    while WORKER_DATA.time < 180.0:
        alpha = calculate_alpha(WORKER_DATA, np.array([loc[0], loc[1], 0.2]))
        steering_signal = np.clip(alpha / (np.pi/2), -1.0, 1.0)

        if step % 50 == 0:
            current_dist = float(np.linalg.norm(box_pos[0:2] - torso_pos[0:2]))

            if mode == 'FORWARD':
                if current_dist < 0.15: break 
                if step > 0 and step % 1000 == 0:
                    if np.linalg.norm(WORKER_DATA.qpos[:2] - last_check_pos) < 0.02: break
                    last_check_pos = np.array(WORKER_DATA.qpos[:2].copy())
            else: 
                if abs(alpha) < 0.15: break 
                if current_dist > 5.0: break 
                if step > 0 and step % 1000 == 0:
                    if np.linalg.norm(WORKER_DATA.qpos[:2] - last_check_pos) < 0.02: break
                    last_check_pos = np.array(WORKER_DATA.qpos[:2].copy())

        cycle_time = WORKER_DATA.time % 2.0
        steering_mask = np.array([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0])

        if cycle_time < 1.0:
            base_gait = local_weights[:, 0]
            steering = local_weights[:, 2] * steering_signal * steering_mask
        else:
            base_gait = local_weights[:, 1]
            steering = local_weights[:, 3] * steering_signal * steering_mask
        
        commands = base_gait + steering
        WORKER_DATA.ctrl[:] = np.clip(commands, -1.5, 1.5)

        mujoco.mj_step(WORKER_MODEL, WORKER_DATA)
        step += 1 

    if np.random.rand() < 0.01: gc.collect()

    if mode == 'FORWARD': return float(np.linalg.norm(box_pos[0:2] - torso_pos[0:2]))
    else: return abs(alpha) + (float(np.linalg.norm(box_pos[0:2] - torso_pos[0:2])) * 0.1)

# Scenario Recorder

def record_scenario(scenario_name: str, target_loc: list, brains: dict, videos_dir: str, telemetry_dir: str):
    print(f"Setting up scenario: {scenario_name} | Target: {target_loc}")
    
    world = SimpleFlatWorld()
    box = mujoco.MjSpec()
    cube = box.worldbody.add_body(name="cube")
    cube.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.05, 0.05, 0.05), rgba=(0.8, 0.2, 0.2, 1.0), name="target_cube")
    
    robot = gecko()
    world.spawn(robot.spec, position=[0, 0, 0.2])
    world.spawn(box, position=[target_loc[0], target_loc[1], 0.2])

    video_model = world.spec.compile()
    video_data = mujoco.MjData(video_model)
    
    video_model.vis.global_.offwidth = 640
    video_model.vis.global_.offheight = 480
    video_renderer = mujoco.Renderer(video_model, height=480, width=640)
    
    # Save video to videos folder
    video = VideoRecorder(file_name=f"gecko_fsm_{scenario_name}", output_folder=videos_dir)
    
    box_id = video_model.geom("robot2_target_cube").id
    
    step = 0
    frame_count = 0
    fps = 30
    runtime = 160.0 
    
    FOV_LIMIT = np.pi / 2 
    current_state = 'SEARCH'
    search_direction = 'RIGHT' 

    fsm_data = [["Time_sec", "X_Pos", "Y_Pos", "Active_State", "Target_Visible"]]

    while video_data.time < runtime:
        true_alpha = calculate_alpha(video_data, np.array([target_loc[0], target_loc[1], 0.2]))
        sees_box = abs(true_alpha) <= FOV_LIMIT

        if sees_box:
            search_direction = 'RIGHT' if true_alpha > 0 else 'LEFT'
            if current_state == 'SEARCH': current_state = 'RIGHT' if true_alpha > 0 else 'LEFT'
            elif current_state == 'FORWARD':
                if true_alpha > 0.3: current_state = 'RIGHT'
                elif true_alpha < -0.3: current_state = 'LEFT'
            else: 
                if abs(true_alpha) < 0.1: current_state = 'FORWARD'
                elif true_alpha > 0.3: current_state = 'RIGHT'
                elif true_alpha < -0.3: current_state = 'LEFT'
            
            active_weights = brains[current_state]
            if current_state == 'FORWARD': steering_signal = 0.0
            else: steering_signal = np.clip(true_alpha / (np.pi/2), -1.0, 1.0)
        else:
            current_state = 'SEARCH'
            if search_direction == 'RIGHT':
                active_weights = brains['RIGHT']
                steering_signal = 1.0 
            else:
                active_weights = brains['LEFT']
                steering_signal = -1.0 

        cycle_time = video_data.time % 2.0
        steering_mask = np.array([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0])

        if cycle_time < 1.0:
            base_gait = active_weights[:, 0]
            steering = active_weights[:, 2] * steering_signal * steering_mask
        else:
            base_gait = active_weights[:, 1]
            steering = active_weights[:, 3] * steering_signal * steering_mask
            
        commands = base_gait + steering
        video_data.ctrl[:] = np.clip(commands, -1.5, 1.5)
        
        mujoco.mj_step(video_model, video_data)

        if step % int(1.0/(video_model.opt.timestep * fps)) == 0:
            fsm_data.append([
                round(video_data.time, 3), 
                round(video_data.qpos[0], 4), 
                round(video_data.qpos[1], 4), 
                current_state, 
                sees_box
            ])

        step += 1
        
        current_dist = float(np.linalg.norm(video_data.geom_xpos[box_id][0:2] - video_data.qpos[0:2]))
        if sees_box and current_dist < 0.08: video_data.ctrl[:] = np.zeros(8)
            
        if frame_count < video_data.time * fps:
            video_renderer.update_scene(video_data)
            video.write(frame=video_renderer.render())
            frame_count += 1
            
        if sees_box and current_dist < 0.08 and (video_data.time > (frame_count / fps) + 2.0): break

    video.release()
    video_renderer.close()

    # Save telemetry
    csv_filename = os.path.join(telemetry_dir, f"fsm_telemetry_{scenario_name}.csv")
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(fsm_data)
    print(f"Finished recording: gecko_fsm_{scenario_name}.mp4 and saved Telemetry Data.\n")

# MAIN EVOLUTION LOOP
def main():
    # Directory creation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join("run_results", f"run_{timestamp}")
    
    brains_dir = os.path.join(base_dir, "brains")
    training_dir = os.path.join(base_dir, "training_data")
    videos_dir = os.path.join(base_dir, "videos")
    telemetry_dir = os.path.join(base_dir, "telemetry")

    os.makedirs(brains_dir, exist_ok=True)
    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(telemetry_dir, exist_ok=True)
    
    print(f"INITIALIZING NEW RUN: {base_dir}")

    modes = ['FORWARD', 'LEFT', 'RIGHT']
    num_cores = 14 
    generations = 200
    pop_size = 24
    budget = pop_size * generations 
    
    baseline_array = np.array([
    [ 1.0, -1.0, 0.8, 0.8],
    [ 1.0, -1.0, 0.8, 0.8],
    [ 0.0,  0.0, 0.5, 0.5],
    [ 1.2, -1.2, 0.8, 0.8],
    [ 0.0,  0.0, 0.5, 0.5],
    [-1.2,  1.2, 0.8, 0.8],
    [ 1.0, -1.0, 0.5, 0.5],
    [ 1.0, -1.0, 0.5, 0.5]]).flatten()

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores, initializer=_init_worker) as executor:
        for mode in modes:
            print(f"TRAINING BRAIN: {mode}")
            
            parametrization = ng.p.Array(shape=(32,))
            parametrization.set_mutation(sigma=0.5) 
            optimizer = ng.optimizers.CMA(parametrization=parametrization, budget=budget)
            optimizer.suggest(baseline_array)
            
            training_data = [["Generation", "Best_Score", "Average_Score"]]
            start_time = time.time()
            
            for gen in range(generations):
                candidates = []
                for _ in range(pop_size):
                    candidates.append(optimizer.ask())

                args_list = []
                for c in candidates:
                    args_list.append((c.value, mode))
                results = list(executor.map(evaluate_robot, args_list, chunksize=1))
                
                for candidate, result in zip(candidates, results):
                    optimizer.tell(candidate, result)
                
                best_score = min(results)
                mean_score = np.mean(results)
                training_data.append([gen + 1, round(best_score, 4), round(mean_score, 4)])

                if gen % 10 == 0:
                    print(f"Gen {gen + 1}/{generations} | Best: {best_score:.4f} | Mean: {mean_score:.4f}")
            
            end_time = time.time()
            print(f"[{mode}] Evolution took {(end_time - start_time) / 60:.2f} minutes\n")
            
            # Save the brains
            best_weights = np.array(optimizer.provide_recommendation().value).reshape((8,4))
            
            npy_path = os.path.join(brains_dir, f"gecko_brain_{mode.lower()}.npy")
            np.save(npy_path, best_weights)
            
            csv_path = os.path.join(brains_dir, f"gecko_brain_{mode.lower()}.csv")
            brain_headers = "Phase1_Base,Phase2_Base,Phase1_Steer,Phase2_Steer"
            np.savetxt(csv_path, best_weights, delimiter=",", header=brain_headers, comments="")
            
            # Save training data
            train_csv_path = os.path.join(training_dir, f"training_data_{mode.lower()}.csv")
            with open(train_csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(training_data)
            
    print("ALL 3 BRAINS TRAINED AND DATA LOGGED SUCCESSFULLY\n")
    
    # Load brains for video
    print("Loading trained brains for video recording...")
    brains = {
        'FORWARD': np.load(os.path.join(brains_dir, "gecko_brain_forward.npy")),
        'LEFT':    np.load(os.path.join(brains_dir, "gecko_brain_left.npy")),
        'RIGHT':   np.load(os.path.join(brains_dir, "gecko_brain_right.npy"))
    }

    scenarios = {
        "forward": [0.0, -2.0],  
        "left":    [-2.0, -1.5], 
        "right":   [2.0, -1.5],  
        "behind":  [0.0, 2.0]    
    }

    print("Starting FSM Scenario Recording...")
    for name, loc in scenarios.items():
        # Pass the dynamically generated directories so it knows where to save
        record_scenario(name, loc, brains, videos_dir, telemetry_dir)

    print(f"\nSUCCESS All files have been organized into: {base_dir}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True) 
    main()