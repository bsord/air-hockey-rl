"""
Utility functions for Air Hockey Environment

Contains helper functions and utility classes.
"""

import pygame
import sys
import math


class WindowManager:
    """Manages window operations and state"""
    
    @staticmethod
    def maximize_window():
        """Maximize the pygame window"""
        try:
            from pygame._sdl2 import Window
            Window.from_display_module().maximize()
        except ImportError:
            print("Warning: SDL2 not available for window maximization")
    
    @staticmethod
    def restore_window():
        """Restore the pygame window to original size"""
        try:
            from pygame._sdl2 import Window
            Window.from_display_module().restore()
        except ImportError:
            print("Warning: SDL2 not available for window restoration")
    
    @staticmethod
    def toggle_maximize(screen, original_size):
        """Toggle between maximized and restored window"""
        current_size = screen.get_size()
        if current_size == original_size:
            WindowManager.maximize_window()
        else:
            WindowManager.restore_window()


class EventHandler:
    """Handles pygame events"""
    
    def __init__(self, env):
        self.env = env
        self.running = True
    
    def handle_events(self):
        """Process pygame events and return running state"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self._handle_keydown(event)
        
        return self.running
    
    def _handle_keydown(self, event):
        """Handle keyboard input"""
        if event.key == pygame.K_ESCAPE:
            self.running = False
        elif event.key == pygame.K_SPACE:
            self.env.reset_ball()
        elif event.key == pygame.K_F1:
            # Toggle debug drawing
            debug_state = self.env.renderer.toggle_debug()
            print(f"Debug drawing: {'ON' if debug_state else 'OFF'}")
        elif event.key == pygame.K_m:
            # Toggle window maximization
            WindowManager.toggle_maximize(self.env.screen, (self.env.width, self.env.height))


class MathUtils:
    """Mathematical utility functions"""
    
    @staticmethod
    def angle_to_vector(angle, magnitude=1.0):
        """Convert angle in radians to a vector"""
        return (magnitude * math.cos(angle), magnitude * math.sin(angle))
    
    @staticmethod
    def vector_to_angle(vx, vy):
        """Convert velocity vector to angle in radians"""
        return math.atan2(vy, vx)
    
    @staticmethod
    def distance(pos1, pos2):
        """Calculate distance between two points"""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return math.sqrt(dx * dx + dy * dy)
    
    @staticmethod
    def normalize_vector(vx, vy):
        """Normalize a vector to unit length"""
        magnitude = math.sqrt(vx * vx + vy * vy)
        if magnitude == 0:
            return 0, 0
        return vx / magnitude, vy / magnitude


class GameClock:
    """Manages game timing and FPS"""
    
    def __init__(self, fps=60):
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.dt = 1.0 / fps
    
    def tick(self):
        """Tick the clock and return delta time"""
        self.clock.tick(self.fps)
        return self.dt
    
    def get_fps(self):
        """Get current FPS"""
        return self.clock.get_fps()


def safe_exit():
    """Safely exit pygame and python"""
    pygame.quit()
    sys.exit()


def print_controls():
    """Print control instructions to console"""
    print("🏒 Physics Air Hockey Environment v2")
    print("Controls:")
    print("- ESC: Quit")
    print("- SPACE: Reset ball")
    print("- WASD: Move player paddle (green)")
    print("- F1: Toggle debug physics drawing")
    print("- M: Toggle window maximize/restore")
    print("- Opponent paddle (orange) is AI controlled")
    print("- Score goals by getting the puck into the opponent's goal!")