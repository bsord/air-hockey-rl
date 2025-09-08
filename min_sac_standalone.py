"""
Standalone minimal Air-Hockey-like env + SAC baseline (self-contained).

- No dependency on the repository's existing code.
- Implements a simple Gymnasium env `MiniAirHockeyEnv` with a single paddle and ball.
- Reward is proximity between paddle center and ball center, normalized to [0,1].
- Optional visualization via pygame.
- Trains an SAC agent (stable-baselines3) for a few timesteps as a baseline.

Usage:
  python min_sac_standalone.py --timesteps 2000
  python min_sac_standalone.py --timesteps 20000 --visible

Dependencies (only for training/visualization):
  pip install gymnasium stable-baselines3[extra] pygame
"""

import argparse
import math
import random
import time
import os

import numpy as np
import csv
import gymnasium as gym
from gymnasium import spaces

# Try to import stable-baselines3 only when training is requested

# ---------- Minimal environment ----------
class MiniAirHockeyEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, width=640, height=480, max_steps=600, render_mode=None, center_serve_prob=0.0, paddle_speed=220.0):
        super().__init__()
        self.W = width
        self.H = height
        self.render_mode = render_mode
        self.max_steps = int(max_steps)

        # Paddle (agent) parameters
        self.paddle_radius = 20.0
        self.paddle_x = width / 2.0
        self.paddle_y = height - 40.0
        self.paddle_speed = float(paddle_speed)
        # Ball parameters
        self.ball_radius = 10.0
        self.ball_x = width / 2.0
        self.ball_y = height / 2.0
        self.ball_vx = 120.0  # px / s
        self.ball_vy = 80.0

        # Timing for physics
        self.dt = 1.0 / 60.0
        # Probability that reset places the puck static in the center (serve)
        self.center_serve_prob = float(center_serve_prob)

        # Observation: [paddle_x, paddle_y, ball_x, ball_y, ball_vx, ball_vy]
        high = np.array([self.W, self.H, self.W, self.H, 1e3, 1e3], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Action: continuous 2D paddle velocity direction in [-1,1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Rendering
        self.viewer = None
        # whether the pygame window has been closed by the user
        self._closed = False
        self._step_count = 0
        # track whether ball was in opponent (blue) half last step
        self._ball_in_opp = False
        # last step reward (for HUD)
        self._last_reward = 0.0
        # center-serve tracking (kept for potential future use)
        self._center_static = False
        self._prev_approach_dist = None
        # episode logging
        self._ep_reward = 0.0
        self._ep_length = 0
        self._ep_touches = 0
        # CSV path for episode stats (assumption: save in cwd)
        self._csv_path = os.path.join(os.getcwd(), 'min_sac_standalone_ep_stats.csv')

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.paddle_x = self.W / 2.0
        self.paddle_y = self.H - 40.0
        self.ball_x = self.W / 2.0
        self.ball_y = self.H / 2.0
    # With some probability keep the puck static in the center to simulate a serve.
        if random.random() < getattr(self, 'center_serve_prob', 0.0):
            # place puck static exactly at center (serve)
            self.ball_vx = 0.0
            self.ball_vy = 0.0
            # mark that this episode started with a center static serve
            self._center_static = True
        else:
            angle = random.uniform(0, 2 * math.pi)
            speed = 140.0
            self.ball_vx = speed * math.cos(angle)
            self.ball_vy = speed * math.sin(angle)
            self._center_static = False
        # initialize opponent-half flag
        self._ball_in_opp = self.ball_y < (self.H / 2.0)
    # reset episode counters
        self._step_count = 0
        self._ep_reward = 0.0
        self._ep_length = 0
        self._ep_touches = 0
        # reset any transient per-episode vars
        self._prev_approach_dist = None
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        return np.array([self.paddle_x, self.paddle_y, self.ball_x, self.ball_y, self.ball_vx, self.ball_vy], dtype=np.float32)

    def step(self, action):
        # action is desired velocity vector in [-1,1] for each axis
        ax = float(np.clip(action[0], -1.0, 1.0))
        ay = float(np.clip(action[1], -1.0, 1.0))
        # Compute intended velocity vector
        intended_vx = ax * self.paddle_speed
        intended_vy = ay * self.paddle_speed
        # Cap magnitude to max speed
        mag = math.hypot(intended_vx, intended_vy)
        if mag > self.paddle_speed:
            scale = self.paddle_speed / mag
            intended_vx *= scale
            intended_vy *= scale
        # Move paddle by velocity * dt
        self.paddle_x += intended_vx * self.dt
        self.paddle_y += intended_vy * self.dt
        # clamp paddle
        self.paddle_x = float(np.clip(self.paddle_x, self.paddle_radius, self.W - self.paddle_radius))
        self.paddle_y = float(np.clip(self.paddle_y, self.H / 2.0, self.H - self.paddle_radius))
        # Nudge reward: small bonus for moving toward the puck (not for getting closer)
        dx = self.ball_x - self.paddle_x
        dy = self.ball_y - self.paddle_y
        to_puck = np.array([dx, dy])
        move_vec = np.array([intended_vx, intended_vy])
        if np.dot(to_puck, move_vec) > 0:
            reward = 0.1  # nudge for moving toward the puck
        else:
            reward = 0.0

        # ball physics
        self.ball_x += self.ball_vx * self.dt
        self.ball_y += self.ball_vy * self.dt

        # bounce off walls
        if self.ball_x - self.ball_radius < 0:
            self.ball_x = self.ball_radius
            self.ball_vx = -self.ball_vx
        if self.ball_x + self.ball_radius > self.W:
            self.ball_x = self.W - self.ball_radius
            self.ball_vx = -self.ball_vx
        if self.ball_y - self.ball_radius < 0:
            self.ball_y = self.ball_radius
            self.ball_vy = -self.ball_vy
        if self.ball_y + self.ball_radius > self.H:
            self.ball_y = self.H - self.ball_radius
            self.ball_vy = -self.ball_vy

        # simple paddle collision: bounce if overlapping
        dist = math.hypot(dx, dy)
        touched = False
        if dist < (self.ball_radius + self.paddle_radius):
            touched = True
            # compute paddle velocity (px/s) from the action so we can
            # reward harder impacts. Use the normal from paddle->ball
            # to measure closing/pushing speed.
            paddle_vx = intended_vx
            paddle_vy = intended_vy
            # store pre-collision ball velocity
            old_bvx = float(self.ball_vx)
            old_bvy = float(self.ball_vy)
            # reflect ball away from paddle center, but add a portion of
            # the paddle's closing speed to the outgoing speed so impacts
            # transfer paddle momentum to the puck.
            ang = math.atan2(dy, dx)
            old_speed = math.hypot(old_bvx, old_bvy)
            old_speed = max(old_speed, 0.0)
            # normal from paddle->ball
            if dist > 1e-6:
                nx = dx / dist
                ny = dy / dist
            else:
                nx, ny = 0.0, -1.0
            # positive when paddle is pushing the ball along the normal
            rel_norm_vel = (paddle_vx - old_bvx) * nx + (paddle_vy - old_bvy) * ny
            rel_norm_pos = max(0.0, rel_norm_vel)
            # base outgoing speed at least old speed (or a small min)
            base_speed = max(old_speed, 50.0)
            # transfer a fraction of the paddle's normal closing speed into the puck
            IMPACT_TRANSFER = 0.8
            new_speed = base_speed + rel_norm_pos * IMPACT_TRANSFER
            self.ball_vx = new_speed * math.cos(ang)
            self.ball_vy = new_speed * math.sin(ang)
            # nudge ball outside overlap
            overlap = (self.ball_radius + self.paddle_radius) - dist
            if dist > 1e-6:
                self.ball_x += math.cos(ang) * overlap
                self.ball_y += math.sin(ang) * overlap

        # reward: only give reward for touching/hitting the puck
        current_in_opp = self.ball_y < (self.H / 2.0)
        # Note: center-approach and progress shaping have been removed to
        # keep rewards sparse and rely on touch/impact rewards. Keep
        # _prev_approach_dist updated only to a neutral state.
        self._prev_approach_dist = None
        # small shaping bonus for touching the puck to encourage contact
        if touched:
            # make touching more valuable than raw proximity
            TOUCH_BONUS = 10.0
            # reward a bit for any touch
            reward += TOUCH_BONUS
            # reward additional bonus proportional to how hard the paddle
            # hit the puck. Scale and clamp to avoid overpowering the base reward.
            # compute hit bonus from the relative normal speed BEFORE we
            # updated the ball velocity (use rel_norm_pos computed above).
            impact_speed = rel_norm_pos
            # stronger scaling so harder hits matter more
            HIT_REWARD_SCALE = 0.005
            MAX_HIT_BONUS = 2.0
            impact_bonus = min(impact_speed * HIT_REWARD_SCALE, MAX_HIT_BONUS)
            reward += float(impact_bonus)

        self._step_count += 1
        # update episode counters
        self._ep_reward += reward
        self._ep_length += 1
        if touched:
            self._ep_touches += 1
        # update flag for next step (keep tracking the half, but no reward tie)
        self._ball_in_opp = current_in_opp
        # store for HUD
        self._last_reward = float(reward)
        done = self._step_count >= self.max_steps
        info = {}
        if done:
            # expose episode-level stats in the Gym 'episode' info (used by many loggers)
            info['episode'] = {'r': float(self._ep_reward), 'l': int(self._ep_length), 'touches': int(self._ep_touches)}
            # also print a concise summary so it's visible during training
            try:
                print(f"EP DONE len={self._ep_length} reward={self._ep_reward:.3f} touches={self._ep_touches}")
            except Exception:
                pass
            # append to CSV (safe, ignore errors)
            try:
                write_header = not os.path.exists(self._csv_path)
                with open(self._csv_path, 'a', newline='') as _f:
                    _w = csv.writer(_f)
                    if write_header:
                        _w.writerow(['timestamp', 'episode_length', 'episode_reward', 'touches'])
                    _w.writerow([time.time(), int(self._ep_length), float(self._ep_reward), int(self._ep_touches)])
            except Exception:
                pass
        return self._get_obs(), reward, done, False, info

    def render(self):
        try:
            import pygame
        except Exception:
            return
        # if the user already closed the window, don't recreate it
        if getattr(self, '_closed', False):
            return
        if self.viewer is None:
            pygame.init()
            self.viewer = pygame.display.set_mode((self.W, self.H))
            pygame.display.set_caption('Mini Air Hockey (SAC baseline)')
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 24)
        # process events to keep window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # mark closed so external code (callbacks) can stop training
                self._closed = True
                try:
                    pygame.quit()
                except Exception:
                    pass
                self.viewer = None
                # Ensure the entire process exits immediately so training or
                # evaluation loops do not continue running after the user
                # closed the render window.
                try:
                    print('Render window closed by user — exiting process immediately')
                except Exception:
                    pass
                os._exit(0)
        surface = self.viewer
        surface.fill((20, 80, 20))
        # draw ball
        pygame.draw.circle(surface, (240, 200, 60), (int(self.ball_x), int(self.ball_y)), int(self.ball_radius))
        # draw paddle
        pygame.draw.circle(surface, (180, 50, 50), (int(self.paddle_x), int(self.paddle_y)), int(self.paddle_radius))
        # HUD: show last-step reward and episode totals
        txt = self.font.render(f"Last reward: {getattr(self, '_last_reward', 0.0):.3f}", True, (240,240,240))
        surface.blit(txt, (8, 8))
        # episode stats
        ep_txt = self.font.render(f"EP len={getattr(self, '_ep_length', 0)} r={getattr(self, '_ep_reward', 0.0):.2f} touches={getattr(self, '_ep_touches', 0)}", True, (240,240,240))
        surface.blit(ep_txt, (8, 32))
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        try:
            import pygame
            if self.viewer is not None:
                pygame.quit()
        except Exception:
            pass


# ---------- Trainer / CLI ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=2000)
    parser.add_argument('--visible', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--save-path', type=str, default='min_sac_standalone')
    parser.add_argument('--max-steps', type=int, default=120,
                        help='Maximum steps per episode (TimeLimit). Default 120 for faster episodes')
    parser.add_argument('--device', type=str, default='auto',
                        help="Device to use: 'auto'|'cpu'|'cuda' (auto selects cuda if available)")
    parser.add_argument('--center-serve-prob', type=float, default=0.60,
                        help='Probability [0..1] that reset places the puck static at center (serve)')
    parser.add_argument('--paddle-speed', type=float, default=220.0,
                        help='Maximum paddle speed in px/s (agent can move at any speed up to this value, e.g. 350)')
    args = parser.parse_args()

    env = MiniAirHockeyEnv(render_mode='human' if args.visible else None,
                           center_serve_prob=args.center_serve_prob,
                           paddle_speed=args.paddle_speed)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=int(args.max_steps))

    # Try to import stable-baselines3 lazily
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import BaseCallback
    except Exception as e:
        print('stable-baselines3 not installed or failed to import. Install with: pip install stable-baselines3[extra]')
        raise

    class RenderCallback(BaseCallback):
        def __init__(self, render_freq=10):
            super().__init__()
            self.render_freq = render_freq

        def _on_step(self) -> bool:
            if self.n_calls % max(1, self.render_freq) == 0:
                # training_env might be a VecEnv, try to access envs[0]
                try:
                    env = getattr(self.training_env, 'envs', None)
                    if env:
                        e0 = env[0]
                    else:
                        e0 = self.training_env
                except Exception:
                    return True

                # if the user closed the window for any env, stop training by
                # returning False (sb3 respects callback False to stop).
                if getattr(e0, '_closed', False):
                    print('Render window closed by user — exiting process immediately')
                    # Force immediate process termination without attempting further cleanup
                    os._exit(0)

                # call render but swallow render-time exceptions
                try:
                    e0.render()
                except Exception:
                    pass
            return True

    # resolve device choice (auto => use cuda if available)
    device = args.device
    if device == 'auto':
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except Exception:
            device = 'cpu'
    print(f"Using device: {device}")

    # If a saved model exists, load it and continue training; otherwise create a new model
    model_path = f"{args.save_path}.zip"
    if os.path.exists(model_path):
        print(f"Found existing model at {model_path}, loading and continuing training")
        model = SAC.load(args.save_path, env=env, device=device)
    else:
        # increase entropy coefficient slightly to encourage exploration
        # (helps in sparse-reward settings where the agent must try moves)
        model = SAC('MlpPolicy', env, verbose=1, seed=args.seed, device=device, ent_coef=0.5)

    print(f"Training SAC for {args.timesteps} timesteps (visible={args.visible})")
    callback = RenderCallback(render_freq=5) if args.visible else None
    try:
        try:
            model.learn(total_timesteps=args.timesteps, callback=callback)
        except KeyboardInterrupt:
            print('\nTraining interrupted by user (Ctrl+C). Saving model and exiting...')
    finally:
        # always save model when learn returns or is interrupted so
        # completed training is captured.
        try:
            model.save(args.save_path)
            print(f"Model saved to {args.save_path}.zip")
        except Exception as e:
            print('Failed to save model after training interruption:', e)

    # quick visual evaluation
    if not args.visible:
        try:
            viz = MiniAirHockeyEnv(paddle_speed=args.paddle_speed)
            for ep in range(3):
                obs, _ = viz.reset()
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, r, done, _, _ = viz.step(action)
                    viz.render()
                    # If the user closed the visualization window, exit now.
                    if getattr(viz, '_closed', False):
                        try:
                            print('Render window closed by user — exiting process immediately')
                        except Exception:
                            pass
                        os._exit(0)
            viz.close()
        except Exception as e:
            print('Visual eval failed (pygame/display may be unavailable):', e)


if __name__ == '__main__':
    main()
