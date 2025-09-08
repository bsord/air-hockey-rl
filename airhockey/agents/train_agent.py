import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
import torch
import time
import os
import argparse
# Register custom env
import airhockey.envs as envs


class RenderCallback(BaseCallback):
    def __init__(self, render_freq=1):
        super().__init__()
        self.render_freq = render_freq

    def handle_speed_event(self, env):
        import pygame
        try:
            events = pygame.event.get()
        except pygame.error:
            return
        for event in events:
            if event.type == pygame.QUIT:
                try:
                    pygame.quit()
                except Exception:
                    pass
                try:
                    base = getattr(env, "unwrapped", env)
                    setattr(base, "_window_closed", True)
                    setattr(base, "render_mode", None)
                except Exception:
                    pass

    def _on_step(self) -> bool:
        env = self.training_env.envs[0]
        base_env = env.unwrapped
        if self.n_calls % self.render_freq == 0:
            base_env.render()
            self.handle_speed_event(env)
            if getattr(base_env, "_window_closed", False):
                return False
        return True


# Argument for visibility
parser = argparse.ArgumentParser()
parser.add_argument('--visible', action='store_true', help='Enable rendering for preview')
parser.add_argument('--blue-random', action='store_true', help='Use random actions for blue agent')
parser.add_argument('--steps', '--timesteps', type=int, default=10000,
                    help='Total training timesteps (alias: --timesteps)')
# Legacy wrapper flags removed. Use explicit reward component flags instead.
# New explicit reward component flags (replace brittle wrappers)
parser.add_argument('--reward-touch', dest='reward_touch', action='store_true', help='Enable puck-touch reward component (1.0 on first agent touch by default)')
parser.add_argument('--reward-proximity', dest='reward_proximity', action='store_true', help='Enable proximity shaping reward component')
parser.add_argument('--reward-scoring', dest='reward_scoring', action='store_true', help='Enable scoring reward component (positive goal reward)')
parser.add_argument('--reward-winning', dest='reward_winning', action='store_true', help='Enable match-winning reward component')
parser.add_argument('--reward-concede', dest='reward_concede', action='store_true', help='Enable penalty when the agent is scored on')
# Magnitude tuning for composite components
parser.add_argument('--touch-reward', type=float, default=1.0, help='Magnitude for touch reward')
parser.add_argument('--near-bonus', type=float, default=0.2, help='Near-puck one-time bonus (proximity)')
parser.add_argument('--step-shaping', type=float, default=0.01, help='Per-step shaping when decreasing distance')
parser.add_argument('--scoring-reward', type=float, default=1.0, help='Magnitude for scoring reward component')
parser.add_argument('--concede-penalty', type=float, default=1.0, help='Penalty magnitude when conceded')
parser.add_argument('--winning-reward', type=float, default=1.0, help='Reward magnitude for winning a match')
parser.add_argument('--diagnostics-csv', type=str, default=None, help='Path to per-episode diagnostics CSV file (optional)')
parser.add_argument('--inactivity-steps', type=int, default=None,
                    help='End episode if puck not touched for this many steps (env-level inactivity timeout)')
parser.add_argument('--inactivity-penalty', type=float, default=0.0,
                    help='Reward penalty applied on inactivity timeout')
parser.add_argument('--max-episode-steps', type=int, default=None,
                    help='Hard cap on steps per episode (env-level)')
parser.add_argument('--debug-env', action='store_true', help='Enable environment debug prints')
parser.add_argument('--no-randomize-sides', dest='randomize_sides', action='store_false', help='Disable randomizing which paddle the policy controls each episode')
parser.set_defaults(randomize_sides=True)
parser.add_argument('--disable-goals', action='store_true', help='Run environment without goals/resets (diagnostic)')
args = parser.parse_args()

VISIBLE = args.visible
BLUE_RANDOM = args.blue_random
render_mode = "human" if VISIBLE else None

# Note: phases are intentionally not enforced by the CLI. Use --goal-only or
# --touch-only explicitly to enable the lightweight curriculum wrappers.

# Legacy wrapper flags removed; use explicit reward component flags instead.

# Base environment
env_kwargs = dict(render_mode=render_mode, blue_random=BLUE_RANDOM,
                  inactivity_steps=args.inactivity_steps,
                  inactivity_penalty=args.inactivity_penalty,
                  max_episode_steps=args.max_episode_steps,
                  debug=args.debug_env,
                  disable_goals=args.disable_goals,
                  randomize_sides=args.randomize_sides)
base_env = gym.make("AirHockey-v0", **{k: v for k, v in env_kwargs.items() if v is not None})


# Legacy wrappers removed. Use CompositeRewardWrapper with explicit flags instead.


class CompositeRewardWrapper(gym.Wrapper):
    """Compose explicit reward components into a single returned reward.

    Components (enabled via CLI flags):
    - touch: one-time first-touch reward for the agent
    - proximity: small shaping based on distance to puck (near bonus + step shaping)
    - scoring: convert env positive reward to unit scoring reward
    - winning: reward for match win detected via env.last_winner
    - concede: penalty when env reports negative reward (opponent scored)
    - inactivity: env timeout penalty is applied if info['timeout']=='inactivity'
    """
    def __init__(self, env, touch=False, proximity=False, scoring=False, winning=False, concede=False,
                 touch_reward=1.0, near_bonus=0.2, step_shaping=0.01, scoring_reward=1.0, concede_penalty=1.0, winning_reward=1.0,
                 diagnostics_csv=None):
        super().__init__(env)
        self.touch = bool(touch)
        self.proximity = bool(proximity)
        self.scoring = bool(scoring)
        self.winning = bool(winning)
        self.concede = bool(concede)
        # touch bookkeeping
        self._touched = False
        self._near_given = False
        self._prev_dist = None
        # magnitudes
        self.touch_reward = float(touch_reward)
        self.near_bonus = float(near_bonus)
        self.step_shaping = float(step_shaping)
        self.scoring_reward = float(scoring_reward)
        self.concede_penalty = float(concede_penalty)
        self.winning_reward = float(winning_reward)
        # diagnostics
        self._diag_path = diagnostics_csv
        self._diag_file = None
        self._diag_header_written = False
        self._episode_acc = None
        if diagnostics_csv:
            try:
                import csv
                self._diag_file = open(diagnostics_csv, 'a', newline='')
                self._diag_writer = csv.writer(self._diag_file)
            except Exception:
                self._diag_file = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._touched = False
        self._near_given = False
        self._prev_dist = None
        # start per-episode accumulator for diagnostics
        if self._diag_file is not None:
            self._episode_acc = {'first_touch_step': None, 'touch_count': 0, 'steps': 0, 'timeout': None}
        return obs

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)

        shaped = 0.0

        # determine agent physical index for touches/distance
        try:
            agent_phys_idx = 1 if getattr(self.env.unwrapped, '_swapped', False) else 0
            state = getattr(self.env.unwrapped, 'state', None)
            if state is not None:
                agent_x = float(state[agent_phys_idx * 2])
                agent_y = float(state[agent_phys_idx * 2 + 1])
                puck_x = float(state[4])
                puck_y = float(state[5])
                dist = float(((agent_x - puck_x) ** 2 + (agent_y - puck_y) ** 2) ** 0.5)
            else:
                dist = None
        except Exception:
            dist = None
            agent_phys_idx = None

        # touch detection
        try:
            last = getattr(self.env.unwrapped, 'last_toucher', None)
            touched_now = (last == agent_phys_idx)
        except Exception:
            touched_now = False

        if self.touch and touched_now and not self._touched:
            shaped += self.touch_reward
            self._touched = True
            if self._episode_acc is not None:
                self._episode_acc['first_touch_step'] = self.env.unwrapped._step_count
            # count touches
        if touched_now and self._episode_acc is not None:
            self._episode_acc['touch_count'] += 1

        # proximity shaping
        if self.proximity and not self._touched and dist is not None:
            NEAR_RADIUS = 80.0
            NEAR_BONUS = 0.2
            STEP_SHAPING = 0.01
            if dist <= NEAR_RADIUS and not self._near_given:
                shaped += self.near_bonus
                self._near_given = True
            if self._prev_dist is not None and dist < self._prev_dist - 1e-3:
                shaped += self.step_shaping
            self._prev_dist = dist

        # scoring (env positive goal)
        if self.scoring:
            try:
                if env_reward is not None and env_reward > 0:
                    shaped += self.scoring_reward
            except Exception:
                pass

        # concede (penalty when env negative reward)
        if self.concede:
            try:
                if env_reward is not None and env_reward < 0:
                    shaped -= self.concede_penalty
            except Exception:
                pass

        # winning (detect match_over + last_winner)
        if self.winning:
            try:
                info_match = info or {}
                if info_match.get('match_over'):
                    last_winner = getattr(self.env.unwrapped, 'last_winner', None)
                    if last_winner == 'Red':
                        shaped += self.winning_reward
                    elif last_winner == 'Blue':
                        shaped -= self.winning_reward
            except Exception:
                pass

        # inactivity timeout penalty (if env reports it)
        try:
            if info and info.get('timeout') == 'inactivity':
                penalty = getattr(self.env.unwrapped, 'inactivity_penalty', 0.0)
                shaped -= float(penalty)
        except Exception:
            pass

        # expose simple metadata for UI
        try:
            base = self.env.unwrapped
            base._last_env_reward = float(env_reward) if env_reward is not None else None
            base._last_wrapped_reward = float(shaped)
        except Exception:
            pass

        # diagnostics: accumulate per-step and flush on episode end
        try:
            if self._episode_acc is not None:
                self._episode_acc['steps'] += 1
                if info and info.get('timeout') == 'inactivity':
                    self._episode_acc['timeout'] = 'inactivity'
                if info and info.get('match_over'):
                    self._episode_acc['match_over'] = True
        except Exception:
            pass

        # If episode ended, write diagnostics
        if self._diag_file is not None and (terminated or truncated or (info and info.get('match_over'))):
            try:
                row = [getattr(self.env.unwrapped, '_step_count', None),
                       self._episode_acc.get('first_touch_step') if self._episode_acc else None,
                       self._episode_acc.get('touch_count') if self._episode_acc else 0,
                       self._episode_acc.get('timeout') if self._episode_acc else None,
                       getattr(self.env.unwrapped, 'last_winner', None)]
                if not self._diag_header_written:
                    self._diag_writer.writerow(['step', 'first_touch_step', 'touch_count', 'timeout', 'last_winner'])
                    self._diag_header_written = True
                self._diag_writer.writerow(row)
                self._diag_file.flush()
            except Exception:
                pass

        return obs, shaped, terminated, truncated, info


env = base_env
try:
    base = env.unwrapped
    base._reward_config = {
        "wrapper": None,
        "components": {
            "touch": False,
            "proximity": False,
            "scoring": True,
            "winning": True,
            "concede": False,
        },
        "magnitudes": {
            "touch_reward": 0.0,
            "near_bonus": 0.0,
            "step_shaping": 0.0,
            "scoring_reward": 1.0,
            "concede_penalty": 0.0,
            "winning_reward": 1.0,
        },
        "inactivity_steps": getattr(base, 'inactivity_steps', None),
        "inactivity_penalty": getattr(base, 'inactivity_penalty', 0.0),
        "blue_random": getattr(base, 'blue_random', False),
        "randomize_sides": getattr(base, 'randomize_sides', False),
    }
    eff = ["Scoring", "Winning"]
    if getattr(base, 'inactivity_steps', None) is not None and getattr(base, 'inactivity_penalty', 0.0) != 0.0:
        eff.append("Inactivity")
    base._reward_effective = eff
except Exception:
    pass

# If explicit reward component flags were provided, prefer the composite wrapper
reward_flags = any([args.reward_touch, args.reward_proximity, args.reward_scoring, args.reward_winning, args.reward_concede])
if reward_flags:
    print("Using CompositeRewardWrapper with explicit reward components.")
    base_env = gym.make("AirHockey-v0", **{k: v for k, v in env_kwargs.items() if v is not None})
    comp = CompositeRewardWrapper(base_env,
                                  touch=args.reward_touch,
                                  proximity=args.reward_proximity,
                                  scoring=args.reward_scoring,
                                  winning=args.reward_winning,
                                  concede=args.reward_concede,
                                  touch_reward=args.touch_reward,
                                  near_bonus=args.near_bonus,
                                  step_shaping=args.step_shaping,
                                  scoring_reward=args.scoring_reward,
                                  concede_penalty=args.concede_penalty,
                                  winning_reward=args.winning_reward,
                                  diagnostics_csv=args.diagnostics_csv)
    env = comp
    try:
        base = env.unwrapped
        base._reward_config = {
            "wrapper": "composite",
            "components": {
                "touch": bool(args.reward_touch),
                "proximity": bool(args.reward_proximity),
                "scoring": bool(args.reward_scoring),
                "winning": bool(args.reward_winning),
                "concede": bool(args.reward_concede),
            },
            "magnitudes": {
                "touch_reward": args.touch_reward,
                "near_bonus": args.near_bonus,
                "step_shaping": args.step_shaping,
                "scoring_reward": args.scoring_reward,
                "concede_penalty": args.concede_penalty,
                "winning_reward": args.winning_reward,
            },
            "inactivity_steps": getattr(base, 'inactivity_steps', None),
            "inactivity_penalty": getattr(base, 'inactivity_penalty', 0.0),
            "blue_random": getattr(base, 'blue_random', False),
            "randomize_sides": getattr(base, 'randomize_sides', False),
        }
        eff = []
        if args.reward_proximity:
            eff.append("Proximity")
        if args.reward_touch:
            eff.append("Touch")
        if args.reward_scoring:
            eff.append("Scoring")
            eff.append("Winning")
        if args.reward_winning and not args.reward_scoring:
            eff.append("Winning")
        if args.reward_concede:
            eff.append("Concede")
        if getattr(base, 'inactivity_steps', None) is not None and getattr(base, 'inactivity_penalty', 0.0) != 0.0:
            eff.append("Inactivity")
        base._reward_effective = eff
    except Exception:
        pass

# (Legacy touch-only/goals-only wrappers removed.)

# Save model inside the agents package instead of repository root for better organization
agents_dir = os.path.dirname(__file__)
model_basename = os.path.join(agents_dir, "sac_air_hockey")
model_zip_path = model_basename + ".zip"
if os.path.exists(model_zip_path):
    print(f"Loading existing model from {model_zip_path}...")
    model = SAC.load(model_basename, env=env, verbose=1)
else:
    print("Creating new model...")
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

# Adjust render_freq and fps for speed control
callback = RenderCallback(render_freq=1) if VISIBLE else None

model.learn(total_timesteps=args.steps, callback=callback, log_interval=1)

# Save model to agents directory
model.save(model_basename)

env.close()
