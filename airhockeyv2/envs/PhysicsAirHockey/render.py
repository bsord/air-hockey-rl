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
    DARK_GRAY = (64, 64, 64)
    LIGHT_BLUE = (150, 200, 255)
    GREEN = (100, 255, 100)
    YELLOW = (255, 255, 100)
    ORANGE = (255, 165, 0)
    # Air hockey table colors
    TABLE_SURFACE = (240, 248, 255)  # Alice blue - ice-like
    TABLE_BORDER = (70, 130, 180)    # Steel blue
    CENTER_LINE = (100, 149, 237)    # Cornflower blue
    GOAL_COLOR = (255, 69, 0)        # Red orange


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
        """Draw the air hockey table with proper markings"""
        table_width = width - 2 * self.config.TABLE_MARGIN
        table_height = height - 2 * self.config.TABLE_MARGIN
        table_x = self.config.TABLE_MARGIN
        table_y = self.config.TABLE_MARGIN
        
        # Create a surface to draw the table
        table_surface = pygame.Surface((table_width, table_height), pygame.SRCALPHA)
        
        # Draw main table surface (ice-like color)
        # Main rectangle (without corners)
        inner_rect = pygame.Rect(self.config.CORNER_RADIUS, 0, 
                                table_width - 2 * self.config.CORNER_RADIUS, table_height)
        pygame.draw.rect(table_surface, self.colors.TABLE_SURFACE, inner_rect)
        
        # Top and bottom strips
        top_rect = pygame.Rect(0, self.config.CORNER_RADIUS, 
                              table_width, table_height - 2 * self.config.CORNER_RADIUS)
        pygame.draw.rect(table_surface, self.colors.TABLE_SURFACE, top_rect)
        
        # Four corner circles
        # Top-left
        pygame.draw.circle(table_surface, self.colors.TABLE_SURFACE, 
                          (self.config.CORNER_RADIUS, self.config.CORNER_RADIUS), self.config.CORNER_RADIUS)
        # Top-right  
        pygame.draw.circle(table_surface, self.colors.TABLE_SURFACE, 
                          (table_width - self.config.CORNER_RADIUS, self.config.CORNER_RADIUS), self.config.CORNER_RADIUS)
        # Bottom-right
        pygame.draw.circle(table_surface, self.colors.TABLE_SURFACE, 
                          (table_width - self.config.CORNER_RADIUS, table_height - self.config.CORNER_RADIUS), self.config.CORNER_RADIUS)
        # Bottom-left
        pygame.draw.circle(table_surface, self.colors.TABLE_SURFACE, 
                          (self.config.CORNER_RADIUS, table_height - self.config.CORNER_RADIUS), self.config.CORNER_RADIUS)
        
        # Draw center line (vertical for horizontal table)
        center_x = table_width // 2
        pygame.draw.line(table_surface, self.colors.CENTER_LINE, 
                        (center_x, self.config.CORNER_RADIUS), 
                        (center_x, table_height - self.config.CORNER_RADIUS), 3)
        
        # Draw center circle
        center_y = table_height // 2
        pygame.draw.circle(table_surface, self.colors.CENTER_LINE, 
                          (center_x, center_y), 40, 3)
        
        # Draw face-off circles
        face_off_radius = 25
        face_off_distance = table_width // 4
        
        # Left face-off circle
        pygame.draw.circle(table_surface, self.colors.CENTER_LINE, 
                          (face_off_distance, center_y), face_off_radius, 2)
        
        # Right face-off circle
        pygame.draw.circle(table_surface, self.colors.CENTER_LINE, 
                          (table_width - face_off_distance, center_y), face_off_radius, 2)
        
        # Blit the surface to the main screen
        self.screen.blit(table_surface, (table_x, table_y))
    
    def draw_goals(self, width, height):
        """Draw the goals"""
        table_width = width - 2 * self.config.TABLE_MARGIN
        table_height = height - 2 * self.config.TABLE_MARGIN
        table_x = self.config.TABLE_MARGIN
        table_y = self.config.TABLE_MARGIN
        
        goal_center_y = table_y + table_height // 2
        goal_half_height = self.config.GOAL_WIDTH // 2
        
        # Left goal (opponent's goal)
        left_goal_rect = pygame.Rect(
            table_x - self.config.GOAL_DEPTH,
            goal_center_y - goal_half_height,
            self.config.GOAL_DEPTH,
            self.config.GOAL_WIDTH
        )
        pygame.draw.rect(self.screen, self.colors.GOAL_COLOR, left_goal_rect)
        pygame.draw.rect(self.screen, self.colors.BLACK, left_goal_rect, 3)
        
        # Right goal (player's goal)
        right_goal_rect = pygame.Rect(
            table_x + table_width,
            goal_center_y - goal_half_height,
            self.config.GOAL_DEPTH,
            self.config.GOAL_WIDTH
        )
        pygame.draw.rect(self.screen, self.colors.GOAL_COLOR, right_goal_rect)
        pygame.draw.rect(self.screen, self.colors.BLACK, right_goal_rect, 3)
    
    def draw_paddles(self, player_paddle_body, opponent_paddle_body):
        """Draw both paddles"""
        # Player paddle (green)
        player_pos = int(player_paddle_body.position.x), int(player_paddle_body.position.y)
        pygame.draw.circle(self.screen, self.colors.GREEN, player_pos, self.config.PADDLE_RADIUS)
        pygame.draw.circle(self.screen, self.colors.BLACK, player_pos, self.config.PADDLE_RADIUS, 3)
        
        # Opponent paddle (orange)
        opponent_pos = int(opponent_paddle_body.position.x), int(opponent_paddle_body.position.y)
        pygame.draw.circle(self.screen, self.colors.ORANGE, opponent_pos, self.config.PADDLE_RADIUS)
        pygame.draw.circle(self.screen, self.colors.BLACK, opponent_pos, self.config.PADDLE_RADIUS, 3)
    
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