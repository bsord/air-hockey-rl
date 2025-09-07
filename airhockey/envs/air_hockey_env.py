import gymnasium as gym
import numpy as np
import time


class AirHockeyEnv(gym.Env):
    _profile_total = 0.0
    _profile_count = 0
    metadata = {"render_modes": ["human"], "render_fps": 60}
    """Custom 2D Air Hockey environment."""

    def __init__(self, render_mode=None, blue_random=False, own_goal_penalty=0.0, own_goal_time_window=10,
                 inactivity_steps=None, inactivity_penalty=0.0, max_episode_steps=None, debug=False,
                 randomize_sides=True):
        super().__init__()
        self.render_mode = render_mode
        self.blue_random = blue_random

        # Penalty to apply when a paddle last touched the puck and that same paddle
        # later causes a goal for the opponent (an own-goal). Default 0 (disabled).
        self.own_goal_penalty = float(own_goal_penalty)
        # Time-window (in environment steps) within which a last touch counts as
        # responsibility for an own-goal. If 0 or None, fall back to the previous
        # immediate-touch heuristic (no time window).
        self.own_goal_time_window = int(own_goal_time_window) if own_goal_time_window is not None else 0

        # Rendering and window state
        self.viewer = None
        self.clock = None
        self._window_closed = False

        # Match/state bookkeeping
        self.score = [0, 0]  # [player1, player2]
        self.game_count = 0
        self.last_winner = None
        self.last_final_score = None
        self.waiting_for_serve = False

        # Track which paddle last touched the puck: 0 = red (player1), 1 = blue (player2), None = none
        self.last_toucher = None
        self.last_toucher_step = None

        # Step counter for time-windowing last touches
        self._step_count = 0
        # Track steps since last touch for inactivity detection
        self._steps_since_touch = 0

        # Optional debug logging
        self.debug = bool(debug)

        # Optionally randomize which paddle the policy controls each episode.
        self.randomize_sides = bool(randomize_sides)
        self._swapped = False

        # Game constants
        self.BOARD_WIDTH, self.HEIGHT = 600, 400
        self.STATS_WIDTH = 200
        self.WIDTH = self.BOARD_WIDTH + self.STATS_WIDTH
        self.PADDLE_RADIUS = 20
        self.PUCK_RADIUS = 15
        self.PADDLE_SPEED = 8
        self.PUCK_SPEED = 10
        self.GOAL_WIDTH = 100
        self.GOAL_HEIGHT = 10
        self.FPS = 60

        # Inactivity timeout: number of steps without paddle touch that triggers episode end
        if inactivity_steps is None:
            # Default to 20 seconds of inactivity if not provided
            self.inactivity_steps = int(20 * self.FPS)
        else:
            self.inactivity_steps = int(inactivity_steps)
        # Penalty to apply on inactivity timeout
        self.inactivity_penalty = float(inactivity_penalty)
        # Hard max steps per episode
        self.max_episode_steps = int(max_episode_steps) if max_episode_steps is not None else None

        # Observation: [paddle1_x, paddle1_y, paddle2_x, paddle2_y, puck_x, puck_y, puck_vx, puck_vy]
        high = np.array([
            self.BOARD_WIDTH, self.HEIGHT, self.BOARD_WIDTH, self.HEIGHT,
            self.BOARD_WIDTH, self.HEIGHT, self.PUCK_SPEED, self.PUCK_SPEED
        ], dtype=np.float32)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        # Actions: [dx1, dy1, dx2, dy2] (movement for both paddles)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

        # State variables
        self.state = None
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Center paddles and puck for new match
        # Place paddles centered in the board area (BOARD_WIDTH), not including the stats area
        paddle1_x, paddle1_y = self.BOARD_WIDTH // 2, self.HEIGHT - self.PADDLE_RADIUS - 10
        paddle2_x, paddle2_y = self.BOARD_WIDTH // 2, self.PADDLE_RADIUS + 10
        puck_x, puck_y = self.BOARD_WIDTH // 2, self.HEIGHT // 2  # Center of board
        puck_vx, puck_vy = 0.0, 0.0  # Puck does not move until touched
        self.waiting_for_serve = True
        self.state = np.array([
            paddle1_x, paddle1_y, paddle2_x, paddle2_y,
            puck_x, puck_y, puck_vx, puck_vy
        ], dtype=np.float32)
        self.done = False
        # Reset score only if options specify (for new match), otherwise keep
        if options and options.get("reset_score", False):
            self.score = [0, 0]
            self.game_count = 0
        # Reset last toucher on new episode/match
        self.last_toucher = None
        self._step_count = 0
        self._steps_since_touch = 0
        # Randomize sides per episode if enabled
        if self.randomize_sides:
            # swap with 50% probability; prefer gym's RNG if available
            try:
                r = self.np_random.integers(0, 2)
            except Exception:
                r = np.random.randint(0, 2)
            self._swapped = bool(r)
        else:
            self._swapped = False
        if self.debug:
            print(f"[DEBUG] reset: paddle1=({paddle1_x},{paddle1_y}) paddle2=({paddle2_x},{paddle2_y}) puck=({puck_x},{puck_y}) swapped={self._swapped}")
        # Return swapped observation if needed (swap paddle fields)
        obs = self.state.copy()
        if self._swapped:
            obs[[0,1,2,3]] = obs[[2,3,0,1]]
        return obs, {}

    def step(self, action):
        start_time = time.perf_counter()
        # Increment global step counter
        self._step_count += 1
        # If observation/action were swapped for this episode, unswap the incoming action
        # so the environment receives actions in its native ordering (red, then blue).
        if self._swapped:
            # Expect action layout [dx_self,dy_self,dx_opp,dy_opp] where 'self' maps to paddle2 in env
            a = action.copy()
            # swap back to env order
            a[[0,1,2,3]] = a[[2,3,0,1]]
            dx1, dy1, dx2, dy2 = a
        else:
            dx1, dy1, dx2, dy2 = action
        # Unpack state
        paddle1_x, paddle1_y, paddle2_x, paddle2_y, puck_x, puck_y, puck_vx, puck_vy = self.state

        # If blue_random is True, override blue agent's action with random
        if self.blue_random:
            dx2, dy2 = self.action_space.sample()[2:]

        # Move paddles
        paddle1_x = np.clip(paddle1_x + dx1 * self.PADDLE_SPEED, self.PADDLE_RADIUS, self.BOARD_WIDTH - self.PADDLE_RADIUS)
        paddle1_y = np.clip(paddle1_y + dy1 * self.PADDLE_SPEED, self.HEIGHT // 2, self.HEIGHT - self.PADDLE_RADIUS)
        paddle2_x = np.clip(paddle2_x + dx2 * self.PADDLE_SPEED, self.PADDLE_RADIUS, self.BOARD_WIDTH - self.PADDLE_RADIUS)
        paddle2_y = np.clip(paddle2_y + dy2 * self.PADDLE_SPEED, self.PADDLE_RADIUS, self.HEIGHT // 2)

        # Move puck only if not waiting for serve
        if not self.waiting_for_serve:
            puck_x += puck_vx
            puck_y += puck_vy

            # Restrict puck to board area only
            puck_x = np.clip(puck_x, self.PUCK_RADIUS, self.BOARD_WIDTH - self.PUCK_RADIUS)
            puck_y = np.clip(puck_y, self.PUCK_RADIUS, self.HEIGHT - self.PUCK_RADIUS)

            # Wall collision (board area only)
            if puck_y <= self.PUCK_RADIUS or puck_y >= self.HEIGHT - self.PUCK_RADIUS:
                puck_vy *= -1
            if puck_x <= self.PUCK_RADIUS or puck_x >= self.BOARD_WIDTH - self.PUCK_RADIUS:
                puck_vx *= -1

        # Paddle collision (simple distance check)
        paddle_touched = False
        # Iterate with index so we can record which paddle touched the puck
        for idx, (px, py) in enumerate([(paddle1_x, paddle1_y), (paddle2_x, paddle2_y)]):
            dist = np.hypot(puck_x - px, puck_y - py)
            if dist < self.PADDLE_RADIUS + self.PUCK_RADIUS:
                paddle_touched = True
                # Record last toucher: 0 = red (player1), 1 = blue (player2)
                self.last_toucher = idx
                # Record the step when this touch happened for time-windowing
                self.last_toucher_step = self._step_count
                # Reset inactivity counter when puck is touched
                self._steps_since_touch = 0
                if self.debug:
                    print(f"[DEBUG] step {self._step_count}: paddle {idx} touched puck at ({puck_x:.1f},{puck_y:.1f})")
                if self.waiting_for_serve:
                    # Serve: give puck a random velocity
                    angle = np.random.uniform(0, 2 * np.pi)
                    puck_vx = self.PUCK_SPEED * np.cos(angle)
                    puck_vy = self.PUCK_SPEED * np.sin(angle)
                    self.waiting_for_serve = False
                else:
                    # Normal collision: reflect puck
                    angle = np.arctan2(puck_y - py, puck_x - px)
                    puck_vx = self.PUCK_SPEED * np.cos(angle)
                    puck_vy = self.PUCK_SPEED * np.sin(angle)

    # Goal check (score only if puck enters the top or bottom goal area)
        reward = 0.0
        goal_left = self.BOARD_WIDTH // 2 - self.GOAL_WIDTH // 2
        goal_right = self.BOARD_WIDTH // 2 + self.GOAL_WIDTH // 2
        match_over = False
        match_win_bonus = 10.0
        # Top goal (Red scores)
        goal_scored = False
        if puck_y <= self.PUCK_RADIUS and goal_left <= puck_x <= goal_right:
            reward = 1.0
            self.score[0] += 1
            goal_scored = True
        # Bottom goal (Blue scores)
        elif puck_y >= self.HEIGHT - self.PUCK_RADIUS and goal_left <= puck_x <= goal_right:
            reward = -1.0
            self.score[1] += 1
            goal_scored = True

        # Reset paddles and puck after every goal
        if goal_scored:
            paddle1_x, paddle1_y = self.BOARD_WIDTH // 2, self.HEIGHT - self.PADDLE_RADIUS - 10
            paddle2_x, paddle2_y = self.BOARD_WIDTH // 2, self.PADDLE_RADIUS + 10
            # Official air hockey: after a goal, the player who was scored upon serves from in front of their paddle
            serve_distance = self.PADDLE_RADIUS + self.PUCK_RADIUS + 20  # Increased distance for puck placement
            if reward == 1.0:
                # Blue lost, puck in front of blue's paddle
                puck_x = paddle2_x
                puck_y = paddle2_y + serve_distance
            elif reward == -1.0:
                # Red lost, puck in front of red's paddle
                puck_x = paddle1_x
                puck_y = paddle1_y - serve_distance
            else:
                puck_x, puck_y = self.BOARD_WIDTH // 2, self.HEIGHT // 2
            puck_vx, puck_vy = 0.0, 0.0  # Puck does not move until touched
            self.waiting_for_serve = True
            # Detect own-goal using a time-limited last-touch window
            own_goal = False
            if self.last_toucher is not None and self.own_goal_penalty != 0.0:
                # If a time window is configured, require last touch to be recent
                if self.own_goal_time_window and getattr(self, 'last_toucher_step', None) is not None:
                    age = self._step_count - self.last_toucher_step
                    recent = age <= self.own_goal_time_window
                else:
                    # If no window configured (0), treat any last_toucher as recent
                    recent = True

                if recent:
                    # If reward positive (red scored) but last_toucher was blue -> blue own-goal
                    if reward > 0 and self.last_toucher == 1:
                        own_goal = True
                    # If reward negative (blue scored) but last_toucher was red -> red own-goal
                    if reward < 0 and self.last_toucher == 0:
                        own_goal = True
                    if own_goal:
                        # Penalize by subtracting penalty from the scalar reward
                        reward -= float(self.own_goal_penalty)
            # Clear last_toucher after goal regardless
            self.last_toucher = None
            self.last_toucher_step = None
            # Reset inactivity counter after goal/serve
            self._steps_since_touch = 0
            # Pause for 1 second in visible mode
            if self.render_mode == "human":
                self.state = np.array([
                    paddle1_x, paddle1_y, paddle2_x, paddle2_y,
                    puck_x, puck_y, puck_vx, puck_vy
                ], dtype=np.float32)
                self.render()
                time.sleep(1)

        # Check for match end (first to 7 goals)
        if self.score[0] >= 7 or self.score[1] >= 7:
            self.game_count += 1
            match_over = True
            # Give bonus reward for match win
            if self.score[0] >= 7:
                reward += match_win_bonus
                self.last_winner = "Red"
            elif self.score[1] >= 7:
                reward -= match_win_bonus
                self.last_winner = "Blue"
            self.last_final_score = f"{self.score[0]} - {self.score[1]}"
            self.done = True
            self.score = [0, 0]  # Reset for next match
        else:
            self.done = False

        # If paddle wasn't touched this step, increment inactivity counter
        if not paddle_touched:
            self._steps_since_touch += 1

        # Check inactivity timeout
        if self.inactivity_steps is not None and self._steps_since_touch >= self.inactivity_steps:
            # Apply penalty and end episode
            reward -= float(self.inactivity_penalty)
            self.done = True
            info = {"match_over": False, "timeout": "inactivity"}
            return self.state, reward, self.done, False, info

        # Check hard max steps per episode
        if self.max_episode_steps is not None and self._step_count >= self.max_episode_steps:
            self.done = True
            info = {"match_over": False, "timeout": "max_steps"}
            return self.state, reward, self.done, False, info

        self.state = np.array([
            paddle1_x, paddle1_y, paddle2_x, paddle2_y,
            puck_x, puck_y, puck_vx, puck_vy
        ], dtype=np.float32)
        # Return match_over in info for tracking
        info = {"match_over": match_over}
        if goal_scored:
            info["own_goal"] = own_goal
        # Profiling
        AirHockeyEnv._profile_total += time.perf_counter() - start_time
        AirHockeyEnv._profile_count += 1
        if AirHockeyEnv._profile_count % 1000 == 0:
            avg = AirHockeyEnv._profile_total / AirHockeyEnv._profile_count
            print(f"[PROFILE] Average step time: {avg*1000:.3f} ms")
        return self.state, reward, self.done, False, info

    def render(self, fps_display=60):
        # If the window was closed by the user, don't recreate it.
        if getattr(self, "_window_closed", False):
            return
        if self.render_mode != "human":
            return
        import pygame
        if self.viewer is None or not pygame.display.get_init() or not pygame.display.get_surface():
            # If the display is not initialized, try to initialize unless the window
            # was closed previously.
            try:
                pygame.init()
                self.viewer = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
                pygame.display.set_caption("Air Hockey RL")
                self.clock = pygame.time.Clock()
            except pygame.error:
                # If initialization fails (for example headless), avoid raising and
                # simply return without rendering.
                return

        # Process pygame events to keep window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Mark that the user closed the window so we don't recreate it later.
                self._window_closed = True
                # Also turn off render_mode to prevent future render attempts
                self.render_mode = None
                try:
                    pygame.quit()
                except Exception:
                    pass
                self.viewer = None
                return

        # Draw board area
        self.viewer.fill((0, 120, 0), rect=(0, 0, self.BOARD_WIDTH, self.HEIGHT))
        # Draw stats area
        self.viewer.fill((30, 30, 30), rect=(self.BOARD_WIDTH, 0, self.STATS_WIDTH, self.HEIGHT))
        # Draw center line
        pygame.draw.line(self.viewer, (255, 255, 255), (0, self.HEIGHT//2), (self.BOARD_WIDTH, self.HEIGHT//2), 2)
        # Draw goals (top and bottom)
        pygame.draw.rect(self.viewer, (0, 0, 200), (self.BOARD_WIDTH//2 - self.GOAL_WIDTH//2, 0, self.GOAL_WIDTH, self.GOAL_HEIGHT))
        pygame.draw.rect(self.viewer, (200, 0, 0), (self.BOARD_WIDTH//2 - self.GOAL_WIDTH//2, self.HEIGHT - self.GOAL_HEIGHT, self.GOAL_WIDTH, self.GOAL_HEIGHT))
        # Draw paddles
        paddle1_x, paddle1_y, paddle2_x, paddle2_y, puck_x, puck_y, _, _ = self.state
        # Determine which physical paddle the agent controls this episode.
        # If _swapped is True the agent is mapped to paddle index 1 (the top paddle),
        # otherwise to index 0 (the bottom paddle). We always visually show the
        # training agent as Red and the opponent as Blue so the visible UI is unambiguous.
        agent_phys_idx = 1 if getattr(self, '_swapped', False) else 0
        agent_color_rgb = (255, 0, 0)
        opp_color_rgb = (0, 0, 255)
        if agent_phys_idx == 0:
            pygame.draw.circle(self.viewer, agent_color_rgb, (int(paddle1_x), int(paddle1_y)), self.PADDLE_RADIUS)
            pygame.draw.circle(self.viewer, opp_color_rgb, (int(paddle2_x), int(paddle2_y)), self.PADDLE_RADIUS)
        else:
            # Agent is physically paddle2 (top) this episode; draw it red.
            pygame.draw.circle(self.viewer, opp_color_rgb, (int(paddle1_x), int(paddle1_y)), self.PADDLE_RADIUS)
            pygame.draw.circle(self.viewer, agent_color_rgb, (int(paddle2_x), int(paddle2_y)), self.PADDLE_RADIUS)
        # Draw puck
        pygame.draw.circle(self.viewer, (255, 255, 255), (int(puck_x), int(puck_y)), self.PUCK_RADIUS)
        # Draw stats in the right area (simplified)
        font = pygame.font.SysFont(None, 32)
        stats_x = self.BOARD_WIDTH + 20
        # If sides are swapped this episode, the training agent is physically player2
        # so swap the displayed scores so Red (Agent) always shows the agent's score.
        if getattr(self, '_swapped', False):
            displayed_red_score = self.score[1]
            displayed_blue_score = self.score[0]
        else:
            displayed_red_score = self.score[0]
            displayed_blue_score = self.score[1]

        # Red is always the training agent in the UI
        red_text = font.render(f"Red (Agent): {displayed_red_score}", True, (255, 0, 0))
        # Blue shows opponent mode in parentheses
        blue_mode = "random" if self.blue_random else "self-play"
        blue_text = font.render(f"Blue ({blue_mode}): {displayed_blue_score}", True, (0, 0, 255))
        self.viewer.blit(red_text, (stats_x, 40))
        self.viewer.blit(blue_text, (stats_x, 80))
    # (Numeric reward values removed — UI shows which reward components are active.)
    # Show active reward configuration (wrapper, shaping, inactivity settings)
        try:
            cfg = getattr(self, '_reward_config', None)
            cfg_font = pygame.font.SysFont(None, 22)
            if cfg is not None:
                wrapper = cfg.get('wrapper') or 'none'
                shaping = 'on' if cfg.get('touch_shaping') else 'off'
                inact_steps = cfg.get('inactivity_steps')
                inact_pen = cfg.get('inactivity_penalty')
                blue_rand = 'on' if cfg.get('blue_random') else 'off'
                sides = 'random' if cfg.get('randomize_sides') else 'fixed'
                self.viewer.blit(cfg_font.render(f"Wrapper: {wrapper}", True, (200,200,200)), (stats_x, 120))
                self.viewer.blit(cfg_font.render(f"Touch shaping: {shaping}", True, (200,200,200)), (stats_x, 144))
                self.viewer.blit(cfg_font.render(f"Inactivity: steps={inact_steps}", True, (200,200,200)), (stats_x, 168))
                self.viewer.blit(cfg_font.render(f"Inactivity penalty: {inact_pen}", True, (200,200,200)), (stats_x, 192))
                self.viewer.blit(cfg_font.render(f"Blue random: {blue_rand}", True, (200,200,200)), (stats_x, 216))
                self.viewer.blit(cfg_font.render(f"Sides: {sides}", True, (200,200,200)), (stats_x, 240))
        except Exception:
            pass
        # Show which reward mechanisms are effective (clear, non-numeric list)
        try:
            eff = getattr(self, '_reward_effective', None)
            if eff:
                header = cfg_font.render("Rewards:", True, (255,255,255))
                self.viewer.blit(header, (stats_x, 272))
                y = 300
                bullet_font = pygame.font.SysFont(None, 20)
                for item in eff:
                    line = bullet_font.render(f"- {item}", True, (200,200,200))
                    self.viewer.blit(line, (stats_x, y))
                    y += 22
        except Exception:
            pass
    # Note: match-level Games counter intentionally not shown in the UI
        # Show match result if available
        if self.last_winner is not None and self.last_final_score is not None:
            winner_text = font.render(f"Winner: {self.last_winner}", True, (255,255,255))
            final_score_text = font.render(f"Final Score: {self.last_final_score}", True, (255,255,255))
            self.viewer.blit(winner_text, (stats_x, 200))
            self.viewer.blit(final_score_text, (stats_x, 240))
            # Clear after a few renders
            if hasattr(self, '_result_display_count'):
                self._result_display_count += 1
            else:
                self._result_display_count = 1
            if self._result_display_count > 120:  # ~2 seconds at 60 FPS
                self.last_winner = None
                self.last_final_score = None
                self._result_display_count = 0
        pygame.display.flip()
        self.clock.tick(60)
