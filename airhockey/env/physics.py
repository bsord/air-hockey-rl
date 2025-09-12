# Physics and collision logic for MiniAirHockeyEnv

# Add functions here for movement, collision, and related calculations.

import math
import numpy as np

def update_paddle(env, action):
    """
    Update paddle 1 (left) position and velocity, and check for puck collision.
    """
    ax = float(np.clip(action[0], -1.0, 1.0))
    ay = float(np.clip(action[1], -1.0, 1.0))
    intended_vx = ax * env.paddle_speed
    intended_vy = ay * env.paddle_speed
    mag = math.hypot(intended_vx, intended_vy)
    if mag > env.paddle_speed:
        scale = env.paddle_speed / mag
        intended_vx *= scale
        intended_vy *= scale
    MAX_ACCEL = env.paddle_speed * 0.15
    accel_x = np.clip(intended_vx - env.paddle_vx, -MAX_ACCEL, MAX_ACCEL)
    accel_y = np.clip(intended_vy - env.paddle_vy, -MAX_ACCEL, MAX_ACCEL)
    env.paddle_vx += accel_x
    env.paddle_vy += accel_y
    vel_mag = math.hypot(env.paddle_vx, env.paddle_vy)
    if vel_mag > env.paddle_speed:
        scale = env.paddle_speed / vel_mag
        env.paddle_vx *= scale
        env.paddle_vy *= scale
    env.paddle_x += env.paddle_vx * env.dt
    env.paddle_y += env.paddle_vy * env.dt
    margin = 32
    env.paddle_x = float(np.clip(env.paddle_x, margin + env.paddle_radius, env.W / 2.0 - env.paddle_radius))
    env.paddle_y = float(np.clip(env.paddle_y, margin + env.paddle_radius, env.H - margin - env.paddle_radius))
    dx = env.ball_x - env.paddle_x
    dy = env.ball_y - env.paddle_y
    to_puck = np.array([dx, dy])
    move_vec = np.array([env.paddle_vx, env.paddle_vy])
    if np.dot(to_puck, move_vec) > 0:
        reward = env.nudge_reward
    else:
        reward = 0.0
    dist = math.hypot(dx, dy)
    touched = False
    rel_norm_pos = 0.0
    if dist < (env.ball_radius + env.paddle_radius):
        paddle_vx = env.paddle_vx
        paddle_vy = env.paddle_vy
        old_bvx = float(env.ball_vx)
        old_bvy = float(env.ball_vy)
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
        env.ball_vx = new_speed * math.cos(ang)
        env.ball_vy = new_speed * math.sin(ang)
        overlap = (env.ball_radius + env.paddle_radius) - dist
        if dist > 1e-6:
            env.ball_x += math.cos(ang) * overlap
            env.ball_y += math.sin(ang) * overlap
        if not env._touch_active:
            touched = True
            env._touch_active = True
        else:
            touched = False
    else:
        env._touch_active = False
    return reward, touched, rel_norm_pos
def update_paddle2(env, action):
    """
    Update paddle 2 (right) position and velocity, and check for puck collision.
    """
    ax = float(np.clip(action[0], -1.0, 1.0))
    ay = float(np.clip(action[1], -1.0, 1.0))
    intended_vx = ax * env.paddle_speed
    intended_vy = ay * env.paddle_speed
    mag = math.hypot(intended_vx, intended_vy)
    if mag > env.paddle_speed:
        scale = env.paddle_speed / mag
        intended_vx *= scale
        intended_vy *= scale
    MAX_ACCEL = env.paddle_speed * 0.15
    accel_x = np.clip(intended_vx - env.paddle2_vx, -MAX_ACCEL, MAX_ACCEL)
    accel_y = np.clip(intended_vy - env.paddle2_vy, -MAX_ACCEL, MAX_ACCEL)
    env.paddle2_vx += accel_x
    env.paddle2_vy += accel_y
    vel_mag = math.hypot(env.paddle2_vx, env.paddle2_vy)
    if vel_mag > env.paddle_speed:
        scale = env.paddle_speed / vel_mag
        env.paddle2_vx *= scale
        env.paddle2_vy *= scale
    env.paddle2_x += env.paddle2_vx * env.dt
    env.paddle2_y += env.paddle2_vy * env.dt
    margin = 32
    env.paddle2_x = float(np.clip(env.paddle2_x, env.W / 2.0 + env.paddle_radius, env.W - margin - env.paddle_radius))
    env.paddle2_y = float(np.clip(env.paddle2_y, margin + env.paddle_radius, env.H - margin - env.paddle_radius))
    dx = env.ball_x - env.paddle2_x
    dy = env.ball_y - env.paddle2_y
    to_puck = np.array([dx, dy])
    move_vec = np.array([env.paddle2_vx, env.paddle2_vy])
    if np.dot(to_puck, move_vec) > 0:
        reward = env.nudge_reward
    else:
        reward = 0.0
    dist = math.hypot(dx, dy)
    touched = False
    rel_norm_pos = 0.0
    if dist < (env.ball_radius + env.paddle_radius):
        paddle_vx = env.paddle2_vx
        paddle_vy = env.paddle2_vy
        old_bvx = float(env.ball_vx)
        old_bvy = float(env.ball_vy)
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
        env.ball_vx = new_speed * math.cos(ang)
        env.ball_vy = new_speed * math.sin(ang)
        overlap = (env.ball_radius + env.paddle_radius) - dist
        if dist > 1e-6:
            env.ball_x += math.cos(ang) * overlap
            env.ball_y += math.sin(ang) * overlap
        if not env._touch2_active:
            touched = True
            env._touch2_active = True
        else:
            touched = False
    else:
        env._touch2_active = False
    return reward, touched, rel_norm_pos
    ax = float(np.clip(action[0], -1.0, 1.0))
    ay = float(np.clip(action[1], -1.0, 1.0))
    intended_vx = ax * env.paddle_speed
    intended_vy = ay * env.paddle_speed
    mag = math.hypot(intended_vx, intended_vy)
    if mag > env.paddle_speed:
        scale = env.paddle_speed / mag
        intended_vx *= scale
        intended_vy *= scale
    MAX_ACCEL = env.paddle_speed * 0.15
    accel_x = np.clip(intended_vx - env.paddle_vx, -MAX_ACCEL, MAX_ACCEL)
    accel_y = np.clip(intended_vy - env.paddle_vy, -MAX_ACCEL, MAX_ACCEL)
    env.paddle_vx += accel_x
    env.paddle_vy += accel_y
    vel_mag = math.hypot(env.paddle_vx, env.paddle_vy)
    if vel_mag > env.paddle_speed:
        scale = env.paddle_speed / vel_mag
        env.paddle_vx *= scale
        env.paddle_vy *= scale
    env.paddle_x += env.paddle_vx * env.dt
    env.paddle_y += env.paddle_vy * env.dt
    margin = 32
    env.paddle_x = float(np.clip(env.paddle_x, margin + env.paddle_radius, env.W / 2.0 - env.paddle_radius))
    env.paddle_y = float(np.clip(env.paddle_y, margin + env.paddle_radius, env.H - margin - env.paddle_radius))
    dx = env.ball_x - env.paddle_x
    dy = env.ball_y - env.paddle_y
    to_puck = np.array([dx, dy])
    move_vec = np.array([env.paddle_vx, env.paddle_vy])
    if np.dot(to_puck, move_vec) > 0:
        reward = env.nudge_reward
    else:
        reward = 0.0
    dist = math.hypot(dx, dy)
    touched = False
    rel_norm_pos = 0.0
    if dist < (env.ball_radius + env.paddle_radius):
        paddle_vx = env.paddle_vx
        paddle_vy = env.paddle_vy
        old_bvx = float(env.ball_vx)
        old_bvy = float(env.ball_vy)
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
        env.ball_vx = new_speed * math.cos(ang)
        env.ball_vy = new_speed * math.sin(ang)
        overlap = (env.ball_radius + env.paddle_radius) - dist
        if dist > 1e-6:
            env.ball_x += math.cos(ang) * overlap
            env.ball_y += math.sin(ang) * overlap
        if not env._touch_active:
            touched = True
            env._touch_active = True
        else:
            touched = False
    else:
        env._touch_active = False
    return reward, touched, rel_norm_pos

def update_ball(env):
    env.ball_vx *= env.puck_drag
    env.ball_vy *= env.puck_drag
    # Clamp puck speed
    speed = math.hypot(env.ball_vx, env.ball_vy)
    if hasattr(env, 'max_puck_speed') and speed > env.max_puck_speed:
        scale = env.max_puck_speed / speed
        env.ball_vx *= scale
        env.ball_vy *= scale
    env.ball_x += env.ball_vx * env.dt
    env.ball_y += env.ball_vy * env.dt

def handle_collision(env):
    margin = 32
    # Check for collision with straight walls (square corners)
    if env.ball_x - env.ball_radius < margin:
        env.ball_x = margin + env.ball_radius
        env.ball_vx = -env.ball_vx
    if env.ball_x + env.ball_radius > env.W - margin:
        env.ball_x = env.W - margin - env.ball_radius
        env.ball_vx = -env.ball_vx
    if env.ball_y - env.ball_radius < margin:
        env.ball_y = margin + env.ball_radius
        env.ball_vy = -env.ball_vy
    if env.ball_y + env.ball_radius > env.H - margin:
        env.ball_y = env.H - margin - env.ball_radius
        env.ball_vy = -env.ball_vy

def clamp_positions(env):
    margin = 32
    # Clamp to straight walls (square corners)
    env.ball_x = float(np.clip(env.ball_x, margin + env.ball_radius, env.W - margin - env.ball_radius))
    env.ball_y = float(np.clip(env.ball_y, margin + env.ball_radius, env.H - margin - env.ball_radius))
    
    # If out of bounds, reset to center
    if not (margin + env.ball_radius <= env.ball_x <= env.W - margin - env.ball_radius) or not (margin + env.ball_radius <= env.ball_y <= env.H - margin - env.ball_radius):
        env.ball_x = env.W / 2.0
        env.ball_y = env.H / 2.0
        env.ball_vx = 0.0
        env.ball_vy = 0.0

def clamp_paddle_positions(env):
    margin = 32
    # Paddle 1 - clamp to square boundaries
    env.paddle_x = float(np.clip(env.paddle_x, margin + env.paddle_radius, env.W / 2.0 - env.paddle_radius))
    env.paddle_y = float(np.clip(env.paddle_y, margin + env.paddle_radius, env.H - margin - env.paddle_radius))
    # Paddle 2 - clamp to square boundaries
    env.paddle2_x = float(np.clip(env.paddle2_x, env.W / 2.0 + env.paddle_radius, env.W - margin - env.paddle_radius))
    env.paddle2_y = float(np.clip(env.paddle2_y, margin + env.paddle_radius, env.H - margin - env.paddle_radius))
