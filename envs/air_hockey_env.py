import gymnasium as gym
import numpy as np

class AirHockeyEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}
    """Custom 2D Air Hockey environment."""
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.viewer = None
        self.clock = None
        # Game constants
        self.WIDTH, self.HEIGHT = 600, 400
        self.PADDLE_RADIUS = 20
        self.PUCK_RADIUS = 15
        self.PADDLE_SPEED = 8
        self.PUCK_SPEED = 10
        self.GOAL_WIDTH = 100
        self.FPS = 60

        # Observation: [paddle1_x, paddle1_y, paddle2_x, paddle2_y, puck_x, puck_y, puck_vx, puck_vy]
        high = np.array([
            self.WIDTH, self.HEIGHT, self.WIDTH, self.HEIGHT,
            self.WIDTH, self.HEIGHT, self.PUCK_SPEED, self.PUCK_SPEED
        ], dtype=np.float32)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        # Actions: [dx1, dy1, dx2, dy2] (movement for both paddles)
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(4,), dtype=np.float32
        )

        # State variables
        self.state = None
        self.done = False
        self.viewer = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Center paddles and puck
        paddle1_x, paddle1_y = self.PADDLE_RADIUS + 10, self.HEIGHT // 2
        paddle2_x, paddle2_y = self.WIDTH - self.PADDLE_RADIUS - 10, self.HEIGHT // 2
        puck_x, puck_y = self.WIDTH // 2, self.HEIGHT // 2
        puck_vx, puck_vy = np.random.uniform(-self.PUCK_SPEED, self.PUCK_SPEED, size=2)
        self.state = np.array([
            paddle1_x, paddle1_y, paddle2_x, paddle2_y,
            puck_x, puck_y, puck_vx, puck_vy
        ], dtype=np.float32)
        self.done = False
        return self.state, {}

    def step(self, action):
        # Unpack state
        paddle1_x, paddle1_y, paddle2_x, paddle2_y, puck_x, puck_y, puck_vx, puck_vy = self.state
        dx1, dy1, dx2, dy2 = action

        # Move paddles
        paddle1_x = np.clip(paddle1_x + dx1 * self.PADDLE_SPEED, self.PADDLE_RADIUS, self.WIDTH // 2 - self.PADDLE_RADIUS)
        paddle1_y = np.clip(paddle1_y + dy1 * self.PADDLE_SPEED, self.PADDLE_RADIUS, self.HEIGHT - self.PADDLE_RADIUS)
        paddle2_x = np.clip(paddle2_x + dx2 * self.PADDLE_SPEED, self.WIDTH // 2 + self.PADDLE_RADIUS, self.WIDTH - self.PADDLE_RADIUS)
        paddle2_y = np.clip(paddle2_y + dy2 * self.PADDLE_SPEED, self.PADDLE_RADIUS, self.HEIGHT - self.PADDLE_RADIUS)

        # Move puck
        puck_x += puck_vx
        puck_y += puck_vy

        # Wall collision
        if puck_y < self.PUCK_RADIUS or puck_y > self.HEIGHT - self.PUCK_RADIUS:
            puck_vy *= -1
        if puck_x < self.PUCK_RADIUS or puck_x > self.WIDTH - self.PUCK_RADIUS:
            puck_vx *= -1

        # Paddle collision (simple distance check)
        for px, py in [(paddle1_x, paddle1_y), (paddle2_x, paddle2_y)]:
            dist = np.hypot(puck_x - px, puck_y - py)
            if dist < self.PADDLE_RADIUS + self.PUCK_RADIUS:
                # Reflect puck
                angle = np.arctan2(puck_y - py, puck_x - px)
                puck_vx = self.PUCK_SPEED * np.cos(angle)
                puck_vy = self.PUCK_SPEED * np.sin(angle)

        # Goal check
        reward = 0.0
        if puck_x < self.PUCK_RADIUS:
            # Player 2 scores
            reward = -1.0
            self.done = True
        elif puck_x > self.WIDTH - self.PUCK_RADIUS:
            # Player 1 scores
            reward = 1.0
            self.done = True

        self.state = np.array([
            paddle1_x, paddle1_y, paddle2_x, paddle2_y,
            puck_x, puck_y, puck_vx, puck_vy
        ], dtype=np.float32)
        return self.state, reward, self.done, False, {}

    def render(self):
        if self.render_mode != "human":
            return
        import pygame
        if self.viewer is None:
            pygame.init()
            self.viewer = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Air Hockey RL")
            self.clock = pygame.time.Clock()

        self.viewer.fill((0, 120, 0))
        # Draw center line
        pygame.draw.line(self.viewer, (255, 255, 255), (self.WIDTH//2, 0), (self.WIDTH//2, self.HEIGHT), 2)
        # Draw goals
        pygame.draw.rect(self.viewer, (200, 0, 0), (0, self.HEIGHT//2 - self.GOAL_WIDTH//2, 5, self.GOAL_WIDTH))
        pygame.draw.rect(self.viewer, (0, 0, 200), (self.WIDTH-5, self.HEIGHT//2 - self.GOAL_WIDTH//2, 5, self.GOAL_WIDTH))
        # Draw paddles
        paddle1_x, paddle1_y, paddle2_x, paddle2_y, puck_x, puck_y, _, _ = self.state
        pygame.draw.circle(self.viewer, (255, 0, 0), (int(paddle1_x), int(paddle1_y)), self.PADDLE_RADIUS)
        pygame.draw.circle(self.viewer, (0, 0, 255), (int(paddle2_x), int(paddle2_y)), self.PADDLE_RADIUS)
        # Draw puck
        pygame.draw.circle(self.viewer, (255, 255, 255), (int(puck_x), int(puck_y)), self.PUCK_RADIUS)
        pygame.display.flip()
        self.clock.tick(self.FPS)
