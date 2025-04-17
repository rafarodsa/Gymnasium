import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco.ant_variable_legs import AntVariableLegsEnv
import matplotlib.pyplot as plt
import os
import argparse

# Set EGL as the rendering backend for headless environments
os.environ['MUJOCO_GL'] = 'egl'

def run_and_record(num_legs, camera_view="default", num_steps=500, video_folder="videos_camera_selection"):
    """Run a rollout and record a video of an ant with specified number of legs and camera view."""
    
    # Define camera IDs based on view name
    camera_ids = {
        "default": None,       # Default view (uses default camera config)
        "topview": 1,          # Top-down view using birdseye camera
        "topview_far": 0,      # Far top-down view using track camera  
        "side": 2              # Side view (like original ant env)
    }
    
    # Select the camera ID based on the view name
    camera_id = camera_ids.get(camera_view)
    
    # Create folder with view name for organization
    video_folder = f"{video_folder}_{camera_view}"
    os.makedirs(video_folder, exist_ok=True)
    
    print(f"\nRunning rollout for ant with {num_legs} legs using '{camera_view}' camera view...")
    
    try:
        # Create environment with rendering, passing camera_id if specified
        env = AntVariableLegsEnv(num_legs=num_legs, render_mode="rgb_array", camera_id=camera_id)
        
        # Setup video recording
        env = gym.wrappers.RecordVideo(
            env, 
            video_folder=video_folder,
            name_prefix=f"ant_{num_legs}_legs_{camera_view}",
            episode_trigger=lambda x: True  # Record every episode
        )
        
        # Initialize lists to store positions and rewards
        positions = []
        total_reward = 0
        
        # Reset the environment
        obs, _ = env.reset()
        positions.append(env.unwrapped.get_body_com("torso")[:2].copy())
        
        # Perform the rollout
        for step in range(num_steps):
            # Sample a random action
            action = env.action_space.sample()
            
            # Take a step
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Store position
            positions.append(env.unwrapped.get_body_com("torso")[:2].copy())
            
            # Accumulate reward
            total_reward += reward
            
            # Check if episode is done
            if terminated or truncated:
                print(f"Episode finished at step {step}")
                break
        
        env.close()
        print(f"Total reward: {total_reward}")
        
        # Convert positions to numpy array
        positions_np = np.array(positions)
        
        # Plot and save trajectory
        plot_trajectory(positions_np, num_legs, camera_view, save_path=f"ant_{num_legs}_legs_{camera_view}_trajectory.png")
        
        return total_reward
    
    except Exception as e:
        print(f"Error during rendering: {e}")
        return None

def plot_trajectory(positions, num_legs, camera_view, save_path=None):
    """Plot the trajectory of the ant."""
    plt.figure(figsize=(10, 10))
    plt.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.5, label='Trajectory')
    plt.plot(positions[0, 0], positions[0, 1], 'go', label='Start')
    plt.plot(positions[-1, 0], positions[-1, 1], 'ro', label='End')
    
    # Add arrows to show direction of movement
    for i in range(0, len(positions)-1, 50):  # Add arrow every 50 steps
        if i + 1 < len(positions):
            dx = positions[i+1, 0] - positions[i, 0]
            dy = positions[i+1, 1] - positions[i, 1]
            plt.arrow(positions[i, 0], positions[i, 1], dx, dy, 
                     head_width=0.1, head_length=0.1, fc='k', ec='k', alpha=0.3)
    
    plt.title(f'Ant with {num_legs} legs - {camera_view.capitalize()} View Trajectory')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def generate_summary_video(leg_counts, camera_view, num_steps=500):
    """Generate a summary of performance metrics for different leg counts with a specific camera view."""
    results = {}
    
    for num_legs in leg_counts:
        total_reward = run_and_record(num_legs, camera_view, num_steps)
        results[num_legs] = total_reward
    
    # Print summary
    print(f"\nSummary of rewards (Camera: {camera_view}):")
    for legs, reward in results.items():
        if reward is not None:
            print(f"Ant with {legs} legs: {reward:.2f}")
        else:
            print(f"Ant with {legs} legs: Error during rendering")
    
    return results

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Ant environment with variable legs and camera selection")
    
    parser.add_argument("--legs", type=int, default=4, 
                        help="Number of legs for the ant (default: 4)")
    
    parser.add_argument("--camera", type=str, default="default", 
                        choices=["default", "topview", "topview_far", "side"],
                        help="Camera view to use (default: default)")
    
    parser.add_argument("--steps", type=int, default=500,
                        help="Number of steps to run the simulation (default: 500)")
    
    parser.add_argument("--all-legs", action="store_true",
                        help="Run simulation with multiple leg configurations: 3, 4, 6, 7, 8")
    
    return parser.parse_args()

def main():
    """Main function to run the ant with camera selection."""
    args = parse_arguments()
    
    if args.all_legs:
        # Run with multiple leg configurations
        leg_counts = [3, 4, 6, 7, 8]
        results = generate_summary_video(leg_counts, args.camera, args.steps)
        
        # Plot rewards by leg count (only for successful runs)
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if valid_results:
            plt.figure(figsize=(10, 6))
            legs = list(valid_results.keys())
            rewards = [valid_results[l] for l in legs]
            
            plt.bar(legs, rewards)
            plt.xlabel('Number of Legs')
            plt.ylabel('Total Reward')
            plt.title(f'Reward vs Number of Legs ({args.camera.capitalize()} View)')
            plt.xticks(legs)
            plt.grid(True, alpha=0.3)
            plt.savefig(f"reward_by_leg_count_{args.camera}.png")
            print(f"Reward comparison plot saved to reward_by_leg_count_{args.camera}.png")
    else:
        # Run with a single leg configuration
        run_and_record(args.legs, args.camera, args.steps)

if __name__ == "__main__":
    main() 