import torch
import numpy as np
import os
import soccer_twos
import ray
from ray import tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.tune.logger import UnifiedLogger
from THE_CHOSEN_ONE_AGENT.model import PPOGeneralistNetwork

class TeamTrackerCallback(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        team_a_reward = 0.0
        team_b_reward = 0.0

        for (agent_id, policy_id), reward in episode.agent_rewards.items():
            if agent_id in [0, 1]:
                team_a_reward += reward
            elif agent_id in [2, 3]:
                team_b_reward += reward

        episode.custom_metrics["team_a_reward"] = team_a_reward
        episode.custom_metrics["team_b_reward"] = team_b_reward

class TrainingSoccerWrapper(MultiAgentEnv):
    """Handles Observation Engineering and Reward Shaping during training."""
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs

    def step(self, action_dict):
        obs, rewards, dones, infos = self.env.step(action_dict)
        shaped_rewards = {}

        distances_to_ball = {}
        for agent_id in infos.keys():
            player_pos = infos[agent_id]['player_info']['position']
            ball_pos = infos[agent_id]['ball_info']['position']
            distances_to_ball[agent_id] = np.linalg.norm(ball_pos - player_pos)
        
        closest_team_a = 0 if distances_to_ball.get(0, 99) < distances_to_ball.get(1, 99) else 1
        closest_team_b = 2 if distances_to_ball.get(2, 99) < distances_to_ball.get(3, 99) else 3
        
        for agent_id, base_reward in rewards.items():
            custom_reward = 0.0 

            if base_reward > 0.5:
                custom_reward += 25.0
            elif base_reward < -0.5:
                custom_reward -= 25.0

            agent_info = infos.get(agent_id)
            if agent_info is not None:
                player_pos = agent_info['player_info']['position']
                player_vel = agent_info['player_info']['velocity']
                ball_pos = agent_info['ball_info']['position']
                ball_vel = agent_info['ball_info']['velocity']

                dist_to_ball = distances_to_ball[agent_id]
                vector_to_ball = ball_pos - player_pos
                is_team_a = (agent_id in [0, 1]) 
                is_closest_to_ball = (agent_id == closest_team_a) if is_team_a else (agent_id == closest_team_b)

                if is_closest_to_ball and dist_to_ball > 0.01:
                    direction_to_ball = vector_to_ball / dist_to_ball
                    vel_towards_ball = np.dot(player_vel, direction_to_ball)
                    custom_reward += vel_towards_ball * 0.001
                
                TOUCH_THRESHOLD = 0.5 
                has_possession = dist_to_ball < TOUCH_THRESHOLD
                if has_possession and is_closest_to_ball:
                    custom_reward += 0.002

                ball_vel_x = ball_vel[0]
                ball_vel_towards_goal = ball_vel_x if is_team_a else -ball_vel_x
                custom_reward += ball_vel_towards_goal * 0.002
                        
            shaped_rewards[agent_id] = custom_reward
        
        if "__all__" not in dones:
            dones["__all__"] = all(dones.values()) if dones else False

        return obs, shaped_rewards, dones, infos

class BaselineSoccerWrapper(MultiAgentEnv):
    """A minimal wrapper that passes through environment data with NO reward shaping."""
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action_dict):
        obs, rewards, dones, infos = self.env.step(action_dict)

        if "__all__" not in dones:
            dones["__all__"] = all(dones.values()) if dones else False

        return obs, rewards, dones, infos

class RLlibAdapterModel(TorchModelV2, torch.nn.Module):
    """Wraps our clean PyTorch model so RLlib can use it to train."""
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        torch.nn.Module.__init__(self)
        action_branches = action_space.nvec.tolist()
        
        self.core_network = PPOGeneralistNetwork(obs_space.shape[0], action_branches)
        self._cur_value = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        action_logits = self.core_network.actor(obs)
        self._cur_value = self.core_network.critic(obs).squeeze(-1)
        return action_logits, state

    def value_function(self):
        return self._cur_value

ModelCatalog.register_custom_model("generalist_adapter", RLlibAdapterModel)

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "shared_policy"

def env_creator(env_config):
    worker_idx = env_config.worker_index if hasattr(env_config, "worker_index") else 0
    vector_idx = env_config.vector_index if hasattr(env_config, "vector_index") else 0
    unique_worker_id = worker_idx * 100 + vector_idx

    raw_env = soccer_twos.make(
        worker_id=unique_worker_id,
        render=False, 
        watch=False
    )
    return TrainingSoccerWrapper(raw_env)

if __name__ == "__main__":
    ray.init()
    tune.register_env("SoccerWrapped", env_creator)

    config = {
        "env": "SoccerWrapped",
        "framework": "torch",
        "num_gpus": 1,
        "disable_env_checking": True,
        "multiagent": {
            "policies": {
                "shared_policy": (None, None, None, {}),
            },
            "policy_mapping_fn": policy_mapping_fn,
        },
        "model": {"custom_model": "generalist_adapter"},
        "callbacks": TeamTrackerCallback,
        "num_workers": 10,
        "num_envs_per_worker": 2,
        "train_batch_size": 10000,
        "sgd_minibatch_size": 2048,
        "rollout_fragment_length": 500,
        "num_gpus_per_worker": 0
    }

    # Define a custom folder name for your logs
    LOG_DIR = os.path.join(os.getcwd(), "training_logs")

    def custom_logger_creator(config):
        os.makedirs(LOG_DIR, exist_ok=True)
        return UnifiedLogger(config, LOG_DIR, loggers=None)

    trainer = PPOTrainer(config=config, logger_creator=custom_logger_creator)

    target_timesteps = 10_000_000
    total_timesteps = 0
    iteration = 0
    
    print(f"Starting training for {target_timesteps:,} steps...")
    os.makedirs("backup_checkpoints", exist_ok=True)
    while total_timesteps < target_timesteps:
        result = trainer.train()
        total_timesteps = result["timesteps_total"]
        iteration += 1

        team_a_mean = result["custom_metrics"].get("team_a_reward_mean", 0)
        team_b_mean = result["custom_metrics"].get("team_b_reward_mean", 0)

        print(f"Iter {iteration} | Steps: {total_timesteps:,}/{target_timesteps:,} | "
              f"Team A Reward: {team_a_mean:.2f} | Team B Reward: {team_b_mean:.2f}")

        if iteration % 50 == 0:
            weights = trainer.get_policy("shared_policy").model.core_network.state_dict()
            torch.save(weights, f"backup_checkpoints/shared_iter_{iteration}.pth")
            print(f"--> Saved safe backup at {total_timesteps:,} steps!")

    print("Training complete! Extracting final weights...")

    final_weights = trainer.get_policy("shared_policy").model.core_network.state_dict()
    os.makedirs("THE_CHOSEN_ONE_AGENT", exist_ok=True)
    torch.save(final_weights, "THE_CHOSEN_ONE_AGENT/shared_checkpoint.pth")
    
    print("Exported raw PyTorch weights successfully!")
    ray.shutdown()