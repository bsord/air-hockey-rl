"""
Main environment class for Physics Air Hockey

Integrates physics, rendering, and utilities into a cohesive environment.
"""

import pygame
import sys

from .physics import PhysicsManager, PhysicsConfig
from .render import Renderer
from .utils import EventHandler, GameClock, safe_exit, print_controls


class PhysicsAirHockeyEnv:
    """
    Physics-based Air Hockey Environment
    
    A modular air hockey simulation environment with realistic physics,
    proper rendering, and utility functions for training and evaluation.
    """
    
    def __init__(self, width=800, height=400, maximize_window=False):
        pygame.init()
        
        # Screen setup
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        pygame.display.set_caption("Physics Air Hockey Environment v2")
        
        # Maximize if requested
        if maximize_window:
            from .utils import WindowManager
            WindowManager.maximize_window()
        
        # Initialize modules
        self.config = PhysicsConfig()
        self.physics = PhysicsManager(self.config)
        self.renderer = Renderer(self.screen, self.config)
        self.event_handler = EventHandler(self)
        self.clock = GameClock()
        
        # Create physics objects
        self.boundary_body, self.boundary_shapes = self.physics.update_table_boundaries_with_goals(width, height)
        self.ball_body, self.ball_shape = self.physics.create_ball(width, height)
        
        # Create paddles
        (self.player_paddle_body, self.player_paddle_shape), \
        (self.opponent_paddle_body, self.opponent_paddle_shape) = self.physics.create_paddles(width, height)
        
        # Create goals
        self.goal_body, self.goal_shapes = self.physics.create_goals(width, height)
        
        # Game state
        self.player_score = 0
        self.opponent_score = 0
        self.last_goal_scorer = None
        
        # State
        self.running = True
    
    def reset_ball(self):
        """Reset ball position and velocity"""
        self.physics.reset_ball(self.ball_body, self.width, self.height)
    
    def step(self):
        """Step the environment one frame"""
        # Handle events
        self.running = self.event_handler.handle_events()
        
        # Handle player paddle movement (WASD keys)
        keys = pygame.key.get_pressed()
        paddle_speed = 50
        
        if keys[pygame.K_w]:
            self.physics.move_paddle(self.player_paddle_body, 0, -paddle_speed)
        if keys[pygame.K_s]:
            self.physics.move_paddle(self.player_paddle_body, 0, paddle_speed)
        if keys[pygame.K_a]:
            self.physics.move_paddle(self.player_paddle_body, -paddle_speed, 0)
        if keys[pygame.K_d]:
            self.physics.move_paddle(self.player_paddle_body, paddle_speed, 0)
        
        # Simple AI for opponent paddle (follows ball)
        ball_pos = self.ball_body.position
        opponent_pos = self.opponent_paddle_body.position
        ai_speed = 30
        
        # AI follows ball vertically (up/down)
        if ball_pos.y > opponent_pos.y:
            self.physics.move_paddle(self.opponent_paddle_body, 0, ai_speed)
        elif ball_pos.y < opponent_pos.y:
            self.physics.move_paddle(self.opponent_paddle_body, 0, -ai_speed)
        
        # AI moves slightly toward ball horizontally but stays on left side
        if ball_pos.x > opponent_pos.x and opponent_pos.x < self.config.TABLE_MARGIN + 120:
            self.physics.move_paddle(self.opponent_paddle_body, ai_speed * 0.3, 0)
        elif ball_pos.x < opponent_pos.x and opponent_pos.x > self.config.TABLE_MARGIN + 60:
            self.physics.move_paddle(self.opponent_paddle_body, -ai_speed * 0.3, 0)
        
        # Step physics
        dt = self.clock.tick()
        self.physics.step(dt)
        
        # Check for goals
        goal_scorer = self.physics.check_goal_scored(self.ball_body)
        if goal_scorer:
            if goal_scorer == 'player':
                self.player_score += 1
                print(f"Player scores! Score: Player {self.player_score} - {self.opponent_score} Opponent")
            elif goal_scorer == 'opponent':
                self.opponent_score += 1
                print(f"Opponent scores! Score: Player {self.player_score} - {self.opponent_score} Opponent")
            
            self.last_goal_scorer = goal_scorer
            self.reset_ball()
        
        # Render
        self._render()
        
        return self.running
    
    def _render(self):
        """Internal rendering method"""
        def render_scene():
            self.renderer.draw_table(self.width, self.height)
            self.renderer.draw_goals(self.width, self.height)
            self.renderer.draw_paddles(self.player_paddle_body, self.opponent_paddle_body)
            self.renderer.draw_ball(self.ball_body)
            self.renderer.draw_debug(self.physics.space)
            
            # Draw score
            font = pygame.font.Font(None, 36)
            score_text = font.render(f"Player: {self.player_score}  Opponent: {self.opponent_score}", 
                                   True, (0, 0, 0))
            self.renderer.screen.blit(score_text, (10, 10))
        
        # Handle scaling for resizable window
        self.renderer.render_scaled(self.width, self.height, render_scene)
        pygame.display.flip()
    
    def run(self):
        """Main game loop"""
        print_controls()
        
        while self.running:
            if not self.step():
                break
        
        safe_exit()
    
    def get_ball_position(self):
        """Get current ball position"""
        return self.ball_body.position.x, self.ball_body.position.y
    
    def get_ball_velocity(self):
        """Get current ball velocity"""
        return self.ball_body.velocity.x, self.ball_body.velocity.y
    
    def set_ball_velocity(self, vx, vy):
        """Set ball velocity"""
        self.ball_body.velocity = vx, vy
    
    def get_paddle_positions(self):
        """Get both paddle positions for RL training"""
        return {
            'player': (self.player_paddle_body.position.x, self.player_paddle_body.position.y),
            'opponent': (self.opponent_paddle_body.position.x, self.opponent_paddle_body.position.y)
        }
    
    def move_player_paddle(self, dx, dy):
        """Move player paddle (for RL training)"""
        self.physics.move_paddle(self.player_paddle_body, dx, dy)
    
    def move_opponent_paddle(self, dx, dy):
        """Move opponent paddle (for RL training)"""
        self.physics.move_paddle(self.opponent_paddle_body, dx, dy)
    
    def get_scores(self):
        """Get current scores"""
        return {'player': self.player_score, 'opponent': self.opponent_score}
    
    def reset_scores(self):
        """Reset scores to 0"""
        self.player_score = 0
        self.opponent_score = 0
        self.last_goal_scorer = None
    
    def get_game_state(self):
        """Get complete game state for RL training"""
        return {
            'ball_position': self.get_ball_position(),
            'ball_velocity': self.get_ball_velocity(),
            'paddle_positions': self.get_paddle_positions(),
            'scores': self.get_scores(),
            'last_goal_scorer': self.last_goal_scorer
        }
    
    def get_table_bounds(self):
        """Get table boundaries for RL algorithms"""
        return {
            'x_min': self.config.TABLE_MARGIN + self.config.CORNER_RADIUS,
            'x_max': self.width - self.config.TABLE_MARGIN - self.config.CORNER_RADIUS,
            'y_min': self.config.TABLE_MARGIN + self.config.CORNER_RADIUS,
            'y_max': self.height - self.config.TABLE_MARGIN - self.config.CORNER_RADIUS
        }
    
    def close(self):
        """Close the environment"""
        self.running = False
        safe_exit()


def main():
    """Main function for standalone execution"""
    env = PhysicsAirHockeyEnv(maximize_window=False)
    env.run()


if __name__ == "__main__":
    main()