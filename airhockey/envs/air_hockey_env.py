import gymnasium as gym
import numpy as np
import time

class AirHockeyEnv(gym.Env):
    _profile_total = 0.0
    _profile_count = 0
    metadata = {"render_modes": ["human"], "render_fps": 60}
    """Custom 2D Air Hockey environment."""
    def __init__(self, render_mode=None, blue_random=False):
        super().__init__()
        self.render_mode = render_mode
        self.blue_random = blue_random
        self.viewer = None
        self.clock = None
        # When user closes the pygame window we set this and avoid recreating it.
        self._window_closed = False
        self.score = [0, 0]  # [player1, player2]
        self.game_count = 0
        self.last_winner = None
        self.last_final_score = None
        self.waiting_for_serve = False  # True when puck is waiting to be served
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

        # Observation: [paddle1_x, paddle1_y, paddle2_x, paddle2_y, puck_x, puck_y, puck_vx, puck_vy]
        high = np.array([
            self.BOARD_WIDTH, self.HEIGHT, self.BOARD_WIDTH, self.HEIGHT,
            self.BOARD_WIDTH, self.HEIGHT, self.PUCK_SPEED, self.PUCK_SPEED
        ], dtype=np.float32)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        # Actions: [dx1, dy1, dx2, dy2] (movement for both paddles)
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(4,), dtype=np.float32
        )

        # State variables
        self.state = None
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Center paddles and puck for new match
        paddle1_x, paddle1_y = self.WIDTH // 2, self.HEIGHT - self.PADDLE_RADIUS - 10
        paddle2_x, paddle2_y = self.WIDTH // 2, self.PADDLE_RADIUS + 10
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
        return self.state, {}

    def step(self, action):
        start_time = time.perf_counter()
        # Unpack state
        paddle1_x, paddle1_y, paddle2_x, paddle2_y, puck_x, puck_y, puck_vx, puck_vy = self.state
        dx1, dy1, dx2, dy2 = action

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
        for px, py in [(paddle1_x, paddle1_y), (paddle2_x, paddle2_y)]:
            dist = np.hypot(puck_x - px, puck_y - py)
            if dist < self.PADDLE_RADIUS + self.PUCK_RADIUS:
                paddle_touched = True
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

        self.state = np.array([
            paddle1_x, paddle1_y, paddle2_x, paddle2_y,
            puck_x, puck_y, puck_vx, puck_vy
        ], dtype=np.float32)
        # Return match_over in info for tracking
        info = {"match_over": match_over}
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
        pygame.draw.circle(self.viewer, (255, 0, 0), (int(paddle1_x), int(paddle1_y)), self.PADDLE_RADIUS)
        pygame.draw.circle(self.viewer, (0, 0, 255), (int(paddle2_x), int(paddle2_y)), self.PADDLE_RADIUS)
        # Draw puck
        pygame.draw.circle(self.viewer, (255, 255, 255), (int(puck_x), int(puck_y)), self.PUCK_RADIUS)
        # Draw stats in the right area
        font = pygame.font.SysFont(None, 32)
        score_text = font.render(f"Red: {self.score[0]}", True, (255,0,0))
        blue_text = font.render(f"Blue: {self.score[1]}", True, (0,0,255))
        game_text = font.render(f"Games: {self.game_count}", True, (255,255,0))
        stats_x = self.BOARD_WIDTH + 20
        self.viewer.blit(score_text, (stats_x, 40))
        self.viewer.blit(blue_text, (stats_x, 80))
        self.viewer.blit(game_text, (stats_x, 120))
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
