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
        self.boundary_body, self.boundary_shapes = self.physics.create_table_boundaries(width, height)
        self.ball_body, self.ball_shape = self.physics.create_ball(width, height)
        
        # State
        self.running = True
    
    def reset_ball(self):
        """Reset ball position and velocity"""
        self.physics.reset_ball(self.ball_body, self.width, self.height)
    
    def step(self):
        """Step the environment one frame"""
        # Handle events
        self.running = self.event_handler.handle_events()
        
        # Step physics
        dt = self.clock.tick()
        self.physics.step(dt)
        
        # Render
        self._render()
        
        return self.running
    
    def _render(self):
        """Internal rendering method"""
        def render_scene():
            self.renderer.draw_table(self.width, self.height)
            self.renderer.draw_ball(self.ball_body)
            self.renderer.draw_debug(self.physics.space)
        
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