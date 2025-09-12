"""
Rendering module for Air Hockey Environment

Contains all drawing and visual rendering functionality.
"""

import pygame
import pymunk.pygame_util
from .physics import PhysicsConfig


class Colors:
    """Color constants for rendering"""
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    BLUE = (100, 150, 255)
    RED = (255, 100, 100)
    GRAY = (128, 128, 128)


class Renderer:
    """Handles all rendering operations for the air hockey environment"""
    
    def __init__(self, screen, config=None):
        self.screen = screen
        self.config = config or PhysicsConfig()
        self.colors = Colors()
        
        # Drawing options for debug
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.debug_draw = False
    
    def draw_table(self, width, height):
        """Draw the table visually as a rounded rectangle"""
        table_width = width - 2 * self.config.TABLE_MARGIN
        table_height = height - 2 * self.config.TABLE_MARGIN
        table_x = self.config.TABLE_MARGIN
        table_y = self.config.TABLE_MARGIN
        
        # Create a surface to draw the rounded rectangle
        table_surface = pygame.Surface((table_width, table_height), pygame.SRCALPHA)
        
        # Draw rounded rectangle using pygame.draw functions
        # Main rectangle (without corners)
        inner_rect = pygame.Rect(self.config.CORNER_RADIUS, 0, 
                                table_width - 2 * self.config.CORNER_RADIUS, table_height)
        pygame.draw.rect(table_surface, self.colors.BLUE, inner_rect)
        
        # Top and bottom strips
        top_rect = pygame.Rect(0, self.config.CORNER_RADIUS, 
                              table_width, table_height - 2 * self.config.CORNER_RADIUS)
        pygame.draw.rect(table_surface, self.colors.BLUE, top_rect)
        
        # Four corner circles
        # Top-left
        pygame.draw.circle(table_surface, self.colors.BLUE, 
                          (self.config.CORNER_RADIUS, self.config.CORNER_RADIUS), self.config.CORNER_RADIUS)
        # Top-right  
        pygame.draw.circle(table_surface, self.colors.BLUE, 
                          (table_width - self.config.CORNER_RADIUS, self.config.CORNER_RADIUS), self.config.CORNER_RADIUS)
        # Bottom-right
        pygame.draw.circle(table_surface, self.colors.BLUE, 
                          (table_width - self.config.CORNER_RADIUS, table_height - self.config.CORNER_RADIUS), self.config.CORNER_RADIUS)
        # Bottom-left
        pygame.draw.circle(table_surface, self.colors.BLUE, 
                          (self.config.CORNER_RADIUS, table_height - self.config.CORNER_RADIUS), self.config.CORNER_RADIUS)
        
        # Blit the surface to the main screen
        self.screen.blit(table_surface, (table_x, table_y))
    
    def draw_ball(self, ball_body):
        """Draw the ball"""
        pos = int(ball_body.position.x), int(ball_body.position.y)
        pygame.draw.circle(self.screen, self.colors.RED, pos, self.config.BALL_RADIUS)
        pygame.draw.circle(self.screen, self.colors.BLACK, pos, self.config.BALL_RADIUS, 2)
    
    def draw_debug(self, space):
        """Draw debug physics visualization"""
        if self.debug_draw:
            space.debug_draw(self.draw_options)
    
    def clear_screen(self):
        """Clear the screen with white background"""
        self.screen.fill(self.colors.WHITE)
    
    def render_scaled(self, width, height, render_func):
        """Render to a temporary surface and scale to window size"""
        window_size = self.screen.get_size()
        
        if window_size != (width, height):
            # Render to temp surface at original size, then scale
            temp_surface = pygame.Surface((width, height))
            old_screen = self.screen
            self.screen = temp_surface
            
            # Clear and render
            self.clear_screen()
            render_func()
            
            # Restore original screen and scale
            self.screen = old_screen
            scaled = pygame.transform.scale(temp_surface, window_size)
            self.screen.blit(scaled, (0, 0))
        else:
            # Direct rendering at original size
            self.clear_screen()
            render_func()
    
    def toggle_debug(self):
        """Toggle debug drawing mode"""
        self.debug_draw = not self.debug_draw
        return self.debug_draw