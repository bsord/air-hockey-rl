import argparse
import os
import gymnasium as gym
from airhockey.env import MiniAirHockeyEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=6000)
    parser.add_argument('--visible', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--save-path', type=str, default='min_sac_standalone')
    parser.add_argument('--max-steps', type=int, default=600,
                        help='Maximum steps per episode (TimeLimit). Default 600 to match Optuna tuning')
    parser.add_argument('--device', type=str, default='auto',
                        help="Device to use: 'auto'|'cpu'|'cuda' (auto selects cuda if available)")
    parser.add_argument('--center-serve-prob', type=float, default=0.60,
                        help='Probability [0..1] that reset places the puck static at center (serve)')
    parser.add_argument('--paddle-speed', type=float, default=300.0,
                        help='Maximum paddle speed in px/s (agent can move at any speed up to this value, e.g. 350)')
    parser.add_argument('--touch-reward', type=float, default=30.0, help='Reward for touching the puck')
    parser.add_argument('--nudge-reward', type=float, default=0.01, help='Reward for moving toward the puck')
    parser.add_argument('--consecutive-touch-reward', type=float, default=0.0, help='Reward for consecutive touches (Phase 2 support)')
    parser.add_argument('--touch-reset', action='store_true', help='End episode after first touch')
    parser.add_argument('--puck-drag', type=float, default=0.997, help='Puck drag coefficient per frame (default 0.997, closer to 1 is less drag)')
    args = parser.parse_args()

    env = MiniAirHockeyEnv(render_mode='human' if args.visible else None,
                           center_serve_prob=args.center_serve_prob,
                           paddle_speed=args.paddle_speed,
                           touch_reward=args.touch_reward,
                           nudge_reward=args.nudge_reward,
                           consecutive_touch_reward=args.consecutive_touch_reward,
                           touch_reset=args.touch_reset,
                           puck_drag=args.puck_drag)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=int(args.max_steps))

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
                try:
                    env = getattr(self.training_env, 'envs', None)
                    if env:
                        e0 = env[0]
                    else:
                        e0 = self.training_env
                except Exception:
                    return True
                if getattr(e0, '_closed', False):
                    print('Render window closed by user — exiting process immediately')
                    os._exit(0)
                try:
                    e0.render()
                except Exception:
                    pass
            return True

    device = args.device
    if device == 'auto':
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except Exception:
            device = 'cpu'
    print(f"Using device: {device}")

    model_path = f"{args.save_path}.zip"
    if os.path.exists(model_path):
        print(f"Found existing model at {model_path}, loading and continuing training")
        model = SAC.load(args.save_path, env=env, device=device)
    else:
        model = SAC(
            'MlpPolicy',
            env,
            learning_rate=0.000042,
            ent_coef=0.443913,
            batch_size=256,
            buffer_size=50000,
            verbose=1,
            seed=args.seed,
            device=device
        )

    print(f"Training SAC for {args.timesteps} timesteps (visible={args.visible})")
    callback = RenderCallback(render_freq=5) if args.visible else None
    try:
        try:
            model.learn(total_timesteps=args.timesteps, callback=callback)
        except KeyboardInterrupt:
            print('\nTraining interrupted by user (Ctrl+C). Saving model and exiting...')
    finally:
        try:
            model.save(args.save_path)
            print(f"Model saved to {args.save_path}.zip")
        except Exception as e:
            print('Failed to save model after training interruption:', e)

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
