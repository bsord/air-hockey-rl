import math
import random
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class MiniAirHockeyEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, width=800, height=400, max_steps=600, render_mode=None, center_serve_prob=0.0, paddle_speed=220.0, touch_reward=30.0, nudge_reward=0.01, consecutive_touch_reward=0.0, touch_reset=False, puck_drag=0.997):
        super().__init__()
        self.W = width
        self.H = height
        self.render_mode = render_mode
        self.max_steps = int(max_steps)
        self.paddle_radius = 20.0
        margin = 32
        self.paddle_x = margin + self.paddle_radius
        self.paddle_y = height / 2.0
        self.paddle_speed = float(paddle_speed)
        self.paddle_vx = 0.0
        self.paddle_vy = 0.0
        self.ball_radius = 10.0
        self.ball_x = width / 2.0
        self.ball_y = height / 2.0
        self.ball_vx = 120.0
        self.ball_vy = 80.0
        self.puck_drag = float(puck_drag)
        self.dt = 1.0 / 60.0
        self.center_serve_prob = float(center_serve_prob)
        self.touch_reward = float(touch_reward)
        self.nudge_reward = float(nudge_reward)
        self.consecutive_touch_reward = float(consecutive_touch_reward)
        self.touch_reset = touch_reset
        high = np.array([self.W, self.H, self.W, self.H, 1e3, 1e3], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        margin = 32
        self.paddle_x = margin + self.paddle_radius
        self.paddle_y = self.H / 2.0
        self.paddle_vx = 0.0
        self.paddle_vy = 0.0
        self.ball_x = self.W / 2.0
        self.ball_y = self.H / 2.0
        if random.random() < getattr(self, 'center_serve_prob', 0.0):
            self.ball_vx = 0.0
            self.ball_vy = 0.0
            self._center_static = True
        else:
            angle = random.uniform(0, 2 * math.pi)
            speed = 140.0
            self.ball_vx = speed * math.cos(angle)
            self.ball_vy = speed * math.sin(angle)
            self._center_static = False
        self._ball_in_opp = self.ball_y < (self.H / 2.0)
        self._step_count = 0
        self._ep_reward = 0.0
        self._ep_length = 0
        self._ep_touches = 0
        self._ep_consecutive_touches = 0
        self._prev_approach_dist = None
        self._prev_touched = False
        self._touch_active = False
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        return np.array([self.paddle_x, self.paddle_y, self.ball_x, self.ball_y, self.ball_vx, self.ball_vy], dtype=np.float32)

    def step(self, action):
        ax = float(np.clip(action[0], -1.0, 1.0))
        ay = float(np.clip(action[1], -1.0, 1.0))
        intended_vx = ax * self.paddle_speed
        intended_vy = ay * self.paddle_speed
        mag = math.hypot(intended_vx, intended_vy)
        if mag > self.paddle_speed:
            scale = self.paddle_speed / mag
            intended_vx *= scale
            intended_vy *= scale
        MAX_ACCEL = self.paddle_speed * 0.15
        accel_x = np.clip(intended_vx - self.paddle_vx, -MAX_ACCEL, MAX_ACCEL)
        accel_y = np.clip(intended_vy - self.paddle_vy, -MAX_ACCEL, MAX_ACCEL)
        self.paddle_vx += accel_x
        self.paddle_vy += accel_y
        vel_mag = math.hypot(self.paddle_vx, self.paddle_vy)
        if vel_mag > self.paddle_speed:
            scale = self.paddle_speed / vel_mag
            self.paddle_vx *= scale
            self.paddle_vy *= scale
        self.paddle_x += self.paddle_vx * self.dt
        self.paddle_y += self.paddle_vy * self.dt
        margin = 32
        self.paddle_x = float(np.clip(self.paddle_x, margin + self.paddle_radius, self.W / 2.0 - self.paddle_radius))
        self.paddle_y = float(np.clip(self.paddle_y, margin + self.paddle_radius, self.H - margin - self.paddle_radius))
        if self.ball_x - self.ball_radius < margin:
            self.ball_x = margin + self.ball_radius
            self.ball_vx = -self.ball_vx
        if self.ball_x + self.ball_radius > self.W - margin:
            self.ball_x = self.W - margin - self.ball_radius
            self.ball_vx = -self.ball_vx
        if self.ball_y - self.ball_radius < margin:
            self.ball_y = margin + self.ball_radius
            self.ball_vy = -self.ball_vy
        if self.ball_y + self.ball_radius > self.H - margin:
            self.ball_y = self.H - margin - self.ball_radius
            self.ball_vy = -self.ball_vy
        self.ball_x = float(np.clip(self.ball_x, margin + self.ball_radius, self.W - margin - self.ball_radius))
        self.ball_y = float(np.clip(self.ball_y, margin + self.ball_radius, self.H - margin - self.ball_radius))
        dx = self.ball_x - self.paddle_x
        dy = self.ball_y - self.paddle_y
        to_puck = np.array([dx, dy])
        move_vec = np.array([self.paddle_vx, self.paddle_vy])
        if np.dot(to_puck, move_vec) > 0:
            reward = self.nudge_reward
        else:
            reward = 0.0
        self.ball_vx *= self.puck_drag
        self.ball_vy *= self.puck_drag
        self.ball_x += self.ball_vx * self.dt
        self.ball_y += self.ball_vy * self.dt
        margin = 32
        if self.ball_x - self.ball_radius < margin:
            self.ball_x = margin + self.ball_radius
            self.ball_vx = -self.ball_vx
        if self.ball_x + self.ball_radius > self.W - margin:
            self.ball_x = self.W - margin - self.ball_radius
            self.ball_vx = -self.ball_vx
        if self.ball_y - self.ball_radius < margin:
            self.ball_y = margin + self.ball_radius
            self.ball_vy = -self.ball_vy
        if self.ball_y + self.ball_radius > self.H - margin:
            self.ball_y = self.H - margin - self.ball_radius
            self.ball_vy = -self.ball_vy
        self.ball_x = float(np.clip(self.ball_x, margin + self.ball_radius, self.W - margin - self.ball_radius))
        self.ball_y = float(np.clip(self.ball_y, margin + self.ball_radius, self.H - margin - self.ball_radius))
        if not (margin + self.ball_radius <= self.ball_x <= self.W - margin - self.ball_radius) or not (margin + self.ball_radius <= self.ball_y <= self.H - margin - self.ball_radius):
            self.ball_x = self.W / 2.0
            self.ball_y = self.H / 2.0
            self.ball_vx = 0.0
            self.ball_vy = 0.0
        dist = math.hypot(dx, dy)
        touched = False
        if dist < (self.ball_radius + self.paddle_radius):
            paddle_vx = self.paddle_vx
            paddle_vy = self.paddle_vy
            old_bvx = float(self.ball_vx)
            old_bvy = float(self.ball_vy)
            ang = math.atan2(dy, dx)
            old_speed = math.hypot(old_bvx, old_bvy)
            old_speed = max(old_speed, 0.0)
            if dist > 1e-6:
                nx = dx / dist
                ny = dy / dist
            else:
                nx, ny = 0.0, -1.0
            rel_norm_vel = (paddle_vx - old_bvx) * nx + (paddle_vy - old_bvy) * ny
            rel_norm_pos = max(0.0, rel_norm_vel)
            base_speed = max(old_speed, 50.0)
            IMPACT_TRANSFER = 0.8
            new_speed = base_speed + rel_norm_pos * IMPACT_TRANSFER
            self.ball_vx = new_speed * math.cos(ang)
            self.ball_vy = new_speed * math.sin(ang)
            overlap = (self.ball_radius + self.paddle_radius) - dist
            if dist > 1e-6:
                self.ball_x += math.cos(ang) * overlap
                self.ball_y += math.sin(ang) * overlap
            if not self._touch_active:
                touched = True
                self._touch_active = True
            else:
                touched = False
        else:
            self._touch_active = False
        current_in_opp = self.ball_y < (self.H / 2.0)
        self._prev_approach_dist = None
        if touched:
            reward += self.touch_reward
            impact_speed = rel_norm_pos
            HIT_REWARD_SCALE = 0.01
            MAX_HIT_BONUS = 5.0
            impact_bonus = min(impact_speed * HIT_REWARD_SCALE, MAX_HIT_BONUS)
            reward += float(impact_bonus)
            if self._ep_touches > 0:
                if self.consecutive_touch_reward > 0.0:
                    reward += self.consecutive_touch_reward
                self._ep_consecutive_touches += 1
        self._prev_touched = touched
        self._step_count += 1
        self._ep_reward += reward
        self._ep_length += 1
        if touched:
            self._ep_touches += 1
        self._ball_in_opp = current_in_opp
        self._last_reward = float(reward)
        done = self._step_count >= self.max_steps
        if self.touch_reset and touched:
            done = True
        info = {}
        if done:
            info['episode'] = {'r': float(self._ep_reward), 'l': int(self._ep_length), 'touches': int(self._ep_touches)}
            try:
                print(f"EP DONE len={self._ep_length} reward={self._ep_reward:.3f} touches={self._ep_touches}")
            except Exception:
                pass
        return self._get_obs(), reward, done, False, info

    def render(self):
        try:
            import pygame
        except Exception:
            return
        if getattr(self, '_closed', False):
            return
        if self.viewer is None:
            pygame.init()
            self.viewer = pygame.display.set_mode((self.W, self.H))
            pygame.display.set_caption('Mini Air Hockey (SAC baseline)')
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 24)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._closed = True
                try:
                    pygame.quit()
                except Exception:
                    pass
                self.viewer = None
                try:
                    print('Render window closed by user — exiting process immediately')
                except Exception:
                    pass
                os._exit(0)
        surface = self.viewer
        table_color = (240, 240, 255)
        border_color = (60, 60, 80)
        center_line_color = (180, 0, 0)
        faceoff_circle_color = (0, 120, 255)
        goal_color = (200, 200, 200)
        paddle_color = (220, 40, 40)
        puck_color = (60, 60, 60)
        margin = 32
        table_rect = pygame.Rect(margin, margin, self.W - 2 * margin, self.H - 2 * margin)
        surface.fill((30, 30, 40))
        pygame.draw.rect(surface, table_color, table_rect, border_radius=40)
        pygame.draw.rect(surface, border_color, table_rect, width=6, border_radius=40)
        pygame.draw.line(surface, center_line_color, (self.W // 2, margin), (self.W // 2, self.H - margin), 4)
        pygame.draw.circle(surface, faceoff_circle_color, (self.W // 2, self.H // 2), 48, 3)
        goal_length = 120
        goal_height = 16
        pygame.draw.rect(surface, goal_color, pygame.Rect(margin - goal_height // 2, (self.H - goal_length) // 2, goal_height, goal_length), border_radius=8)
        pygame.draw.rect(surface, goal_color, pygame.Rect(self.W - margin - goal_height // 2, (self.H - goal_length) // 2, goal_height, goal_length), border_radius=8)
        pygame.draw.circle(surface, puck_color, (int(self.ball_x), int(self.ball_y)), int(self.ball_radius))
        pygame.draw.circle(surface, paddle_color, (int(self.paddle_x), int(self.paddle_y)), int(self.paddle_radius))
        ep_txt = self.font.render(
            f"EP len={getattr(self, '_ep_length', 0)} r={getattr(self, '_ep_reward', 0.0):.2f} touches={getattr(self, '_ep_touches', 0)} consecutive: {getattr(self, '_ep_consecutive_touches', 0)}",
            True, (255,255,255))
        ep_rect = ep_txt.get_rect(center=(self.W // 2, 16))
        surface.blit(ep_txt, ep_rect)
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        try:
            import pygame
            if self.viewer is not None:
                pygame.quit()
        except Exception:
            pass
