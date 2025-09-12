# Shared helpers/utilities for MiniAirHockeyEnv

# Add any utility functions needed across env modules.

def reset_env_state(env):
    margin = 32
    env.paddle_x = margin + env.paddle_radius
    env.paddle_y = env.H / 2.0
    env.paddle_vx = 0.0
    env.paddle_vy = 0.0
    # Paddle 2 (right)
    env.paddle2_x = env.W - margin - env.paddle_radius
    env.paddle2_y = env.H / 2.0
    env.paddle2_vx = 0.0
    env.paddle2_vy = 0.0
    env.ball_x = env.W / 2.0
    env.ball_y = env.H / 2.0
    import random, math
    if random.random() < getattr(env, 'center_serve_prob', 0.0):
        env.ball_vx = 0.0
        env.ball_vy = 0.0
        env._center_static = True
    else:
        # Always launch toward learner (left) side
        speed = 140.0
        # Center angle at pi (left), with small random spread
        angle = math.pi + random.uniform(-math.pi/6, math.pi/6)
        env.ball_vx = speed * math.cos(angle)  # vx negative: leftward
        env.ball_vy = speed * math.sin(angle)
        env._center_static = False
    env._ball_in_opp = env.ball_y < (env.H / 2.0)
    env._step_count = 0
    env._ep_reward = 0.0
    env._ep_length = 0
    env._ep_touches = 0
    env._ep_consecutive_touches = 0
    env._prev_approach_dist = None
    env._prev_touched = False
    env._touch_active = False
