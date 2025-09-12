import math
import random
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from airhockey.env.physics import (
    update_paddle,
    update_ball,
    handle_collision,
    clamp_positions
)
from airhockey.env.render import render_env
from airhockey.env.rewards import compute_rewards

class MiniAirHockeyEnv(gym.Env):
    def set_learner_side(self, learner_is_red=True):
        """Set which paddle is the learner (True=red, False=blue)."""
        self.learner_is_red = learner_is_red
    """
    MiniAirHockeyEnv is a custom Gymnasium environment simulating a simple air hockey game.
    The agent controls a paddle to interact with a puck, with physics, rewards, and rendering handled by submodules.
    """
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, width=800, height=400, max_steps=600, render_mode=None, center_serve_prob=0.0, paddle_speed=220.0, touch_reward=30.0, nudge_reward=0.01, consecutive_touch_reward=0.0, touch_reset=False, puck_drag=0.997, max_puck_speed=300.0, solo_mode=False):
        """
        Initialize the MiniAirHockeyEnv.
        Args:
            width (int): Width of the environment.
            height (int): Height of the environment.
            max_steps (int): Maximum steps per episode.
            render_mode (str): Rendering mode ('human' for Pygame window).
            center_serve_prob (float): Probability to serve puck at center.
            paddle_speed (float): Maximum paddle speed.
            touch_reward (float): Reward for touching the puck.
            nudge_reward (float): Reward for moving toward the puck.
            consecutive_touch_reward (float): Reward for consecutive touches.
            touch_reset (bool): End episode after first touch.
            puck_drag (float): Drag coefficient for puck movement.
            solo_mode (bool): If True, train agent with no opponent.
        """
        super().__init__()
        self.W = width
        self.H = height
        self.render_mode = render_mode
        self.max_steps = int(max_steps)
        self.paddle_radius = 20.0
        margin = 32
        # Paddle 1 (left)
        self.paddle_x = margin + self.paddle_radius
        self.paddle_y = height / 2.0
        self.paddle_speed = float(paddle_speed)
        self.paddle_vx = 0.0
        self.paddle_vy = 0.0
        # Paddle 2 (right)
        self.paddle2_x = width - margin - self.paddle_radius
        self.paddle2_y = height / 2.0
        self.paddle2_vx = 0.0
        self.paddle2_vy = 0.0
        self.ball_radius = 10.0
        self.ball_x = width / 2.0
        self.ball_y = height / 2.0
        self.ball_vx = 120.0
        self.ball_vy = 80.0
        self.puck_drag = float(puck_drag)
        self.max_puck_speed = float(max_puck_speed)
        self.dt = 1.0 / 60.0
        self.center_serve_prob = float(center_serve_prob)
        self.touch_reward = float(touch_reward)
        self.nudge_reward = float(nudge_reward)
        self.consecutive_touch_reward = float(consecutive_touch_reward)
        self.touch_reset = touch_reset
        self.solo_mode = solo_mode
        # Observation: [paddle1_x, paddle1_y, paddle2_x, paddle2_y, ball_x, ball_y, ball_vx, ball_vy, paddle1_vx, paddle1_vy, paddle2_vx, paddle2_vy]
        high = np.array([self.W, self.H, self.W, self.H, self.W, self.H, 1e3, 1e3, self.paddle_speed, self.paddle_speed, self.paddle_speed, self.paddle_speed], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        # Actions: [paddle1_dx, paddle1_dy, paddle2_dx, paddle2_dy]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.viewer = None
        self._closed = False
        self._step_count = 0
        self._ball_in_opp = False
        self._last_reward = 0.0
        self._center_static = False
        self._prev_approach_dist = None
        self._ep_reward = 0.0
        self._ep_length = 0
        self._ep_touches = 0
        self._ep_consecutive_touches = 0
        self._prev_touched = False
        self._touch_active = False
        self._touch2_active = False

    def reset(self, seed=None, options=None):
        """
        Reset the environment state for a new episode.
        Returns:
            obs (np.ndarray): Initial observation.
            info (dict): Additional info (empty).
        """
        super().reset(seed=seed)
        from airhockey.env.utils import reset_env_state
        reset_env_state(self)
        # Reset paddle 2
        margin = 32
        self.paddle2_x = self.W - margin - self.paddle_radius
        self.paddle2_y = self.H / 2.0
        self.paddle2_vx = 0.0
        self.paddle2_vy = 0.0
        self._touch2_active = False
        # Alternate learner side every episode, unless solo_mode
        if getattr(self, 'solo_mode', False):
            self.learner_is_red = True
        else:
            if not hasattr(self, 'learner_is_red'):
                self.learner_is_red = True
            else:
                self.learner_is_red = not self.learner_is_red
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        """
        Get the current observation of the environment, mirrored if blue is learner.
        Returns:
            np.ndarray: Observation vector.
        """
        if hasattr(self, 'learner_is_red') and not self.learner_is_red:
            # Mirror horizontally for blue learner
            W = self.W
            obs = np.array([
                W - self.paddle2_x, self.paddle2_y,  # blue paddle as "player 1"
                W - self.paddle_x, self.paddle_y,    # red paddle as "player 2"
                W - self.ball_x, self.ball_y,
                -self.ball_vx, self.ball_vy,
                -self.paddle2_vx, self.paddle2_vy,
                -self.paddle_vx, self.paddle_vy
            ], dtype=np.float32)
            # Optionally, add a flag for learner side
            # obs = np.concatenate([obs, np.array([0.0], dtype=np.float32)])
            return obs
        else:
            # Red is learner (default, no mirroring)
            obs = np.array([
                self.paddle_x, self.paddle_y,
                self.paddle2_x, self.paddle2_y,
                self.ball_x, self.ball_y,
                self.ball_vx, self.ball_vy,
                self.paddle_vx, self.paddle_vy,
                self.paddle2_vx, self.paddle2_vy
            ], dtype=np.float32)
            # Optionally, add a flag for learner side
            # obs = np.concatenate([obs, np.array([1.0], dtype=np.float32)])
            return obs

    def step(self, action):
        """
        Take an action in the environment.
        Args:
            action (np.ndarray): Action vector for both paddles [p1_dx, p1_dy, p2_dx, p2_dy].
        Returns:
            obs (np.ndarray): Next observation.
            reward (float): Reward for this step.
            done (bool): Whether the episode is finished.
            truncated (bool): Whether the episode was truncated.
            info (dict): Additional info.
        """
        # Mirror actions if blue is learner
        if hasattr(self, 'learner_is_red') and not self.learner_is_red:
            # Blue is learner: mirror actions
            action1 = -action[2:4]  # blue paddle (now "player 1")
            action2 = -action[0:2]  # red paddle (now "player 2")
        else:
            # Red is learner (default)
            action1 = action[:2]
            action2 = action[2:]
        # If solo_mode, ignore opponent actions and keep paddle2 stationary
        if getattr(self, 'solo_mode', False):
            action2 = np.array([0.0, 0.0], dtype=np.float32)
        # Update paddle 1
        from airhockey.env.physics import update_paddle, update_paddle2
        reward1, touched1, rel_norm_pos1 = update_paddle(self, action1)
        # Update paddle 2
        reward2, touched2, rel_norm_pos2 = update_paddle2(self, action2)
        # Update ball and collisions
        update_ball(self)
        handle_collision(self)
        clamp_positions(self)
        from airhockey.env.physics import clamp_paddle_positions
        clamp_paddle_positions(self)
        # Compute reward and stats for the learner side
        if hasattr(self, 'learner_is_red') and not self.learner_is_red:
            # Blue is learner
            reward = compute_rewards(self, reward2, touched2, rel_norm_pos2)
            self._step_count += 1
            self._ep_reward += reward
            self._ep_length += 1
            if touched2:
                self._ep_touches += 1
            current_in_opp = self.ball_y > (self.H / 2.0)
            self._ball_in_opp = current_in_opp
            self._last_reward = float(reward)
            done = self._step_count >= self.max_steps
            if self.touch_reset and touched2:
                done = True
            info = {'learner_side': 'blue'}
            if done:
                info['episode'] = {'r': float(self._ep_reward), 'l': int(self._ep_length), 'touches': int(self._ep_touches)}
                try:
                    print(f"EP DONE (blue) len={self._ep_length} reward={self._ep_reward:.3f} touches={self._ep_touches}")
                except Exception:
                    pass
            return self._get_obs(), reward, done, False, info
        else:
            # Red is learner (default)
            reward = compute_rewards(self, reward1, touched1, rel_norm_pos1)
            self._step_count += 1
            self._ep_reward += reward
            self._ep_length += 1
            if touched1:
                self._ep_touches += 1
            current_in_opp = self.ball_y < (self.H / 2.0)
            self._ball_in_opp = current_in_opp
            self._last_reward = float(reward)
            done = self._step_count >= self.max_steps
            if self.touch_reset and touched1:
                done = True
            info = {'learner_side': 'red'}
            if done:
                info['episode'] = {'r': float(self._ep_reward), 'l': int(self._ep_length), 'touches': int(self._ep_touches)}
                try:
                    print(f"EP DONE (red) len={self._ep_length} reward={self._ep_reward:.3f} touches={self._ep_touches}")
                except Exception:
                    pass
            return self._get_obs(), reward, done, False, info

    def render(self):
        """
        Render the current environment state using Pygame.
        """
        render_env(self)

    def close(self):
        """
        Close the environment and clean up resources.
        """
        try:
            import pygame
            if self.viewer is not None:
                pygame.quit()
        except Exception:
            pass
