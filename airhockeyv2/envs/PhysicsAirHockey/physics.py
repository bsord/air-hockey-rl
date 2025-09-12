"""
Physics module for Air Hockey Environment

Contains physics setup, constants, and physics-related utility functions.
"""

import pygame
import pymunk
import math


class PhysicsConfig:
    """Physics configuration constants"""
    GRAVITY = (0, 0)  # No gravity for air hockey
    DAMPING = 0.995  # Slight damping like ice friction
    
    # Wall properties
    WALL_FRICTION = 0.07  # Wall friction (affects sliding behavior along walls)
    WALL_ELASTICITY = 0.8  # Wall bounce
    
    # Ball properties
    BALL_FRICTION = 0.3  # Ball friction
    BALL_ELASTICITY = 0.8  # Ball bounce
    BALL_MASS = 1
    BALL_RADIUS = 12  # Realistic air hockey puck size
    
    # Table properties
    TABLE_MARGIN = 50
    CORNER_RADIUS = 40  # Realistic air hockey table corners


class PhysicsManager:
    """Manages physics space and operations"""
    
    def __init__(self, config=None):
        self.config = config or PhysicsConfig()
        self.space = pymunk.Space()
        self.space.gravity = self.config.GRAVITY
        self.space.damping = self.config.DAMPING
    
    def create_corner_segments(self, boundary_body, center, start_angle, num_segments=12):
        """Create segments for a rounded corner"""
        segments = []
        for i in range(num_segments):
            angle1 = start_angle + i * math.pi/2 / num_segments
            angle2 = start_angle + (i + 1) * math.pi/2 / num_segments
            
            x1 = center[0] + self.config.CORNER_RADIUS * math.cos(angle1)
            y1 = center[1] + self.config.CORNER_RADIUS * math.sin(angle1)
            x2 = center[0] + self.config.CORNER_RADIUS * math.cos(angle2)
            y2 = center[1] + self.config.CORNER_RADIUS * math.sin(angle2)
            
            segment = pymunk.Segment(boundary_body, (x1, y1), (x2, y2), 1)
            segment.friction = self.config.WALL_FRICTION
            segment.elasticity = self.config.WALL_ELASTICITY
            segments.append(segment)
        return segments
    
    def create_table_boundaries(self, width, height):
        """Create rounded rectangle table boundaries using physics bodies"""
        boundary_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        shapes = []
        
        table_width = width - 2 * self.config.TABLE_MARGIN
        table_height = height - 2 * self.config.TABLE_MARGIN
        table_x = self.config.TABLE_MARGIN
        table_y = self.config.TABLE_MARGIN
        
        # Corner centers
        corners = [
            ((table_x + self.config.CORNER_RADIUS, table_y + self.config.CORNER_RADIUS), math.pi),        # Top-left: 180°
            ((table_x + table_width - self.config.CORNER_RADIUS, table_y + self.config.CORNER_RADIUS), 3*math.pi/2),  # Top-right: 270°
            ((table_x + table_width - self.config.CORNER_RADIUS, table_y + table_height - self.config.CORNER_RADIUS), 0),  # Bottom-right: 0°
            ((table_x + self.config.CORNER_RADIUS, table_y + table_height - self.config.CORNER_RADIUS), math.pi/2)  # Bottom-left: 90°
        ]
        
        # Wall segments (top, right, bottom, left)
        walls = [
            ((table_x + self.config.CORNER_RADIUS, table_y), 
             (table_x + table_width - self.config.CORNER_RADIUS, table_y)),  # Top
            ((table_x + table_width, table_y + self.config.CORNER_RADIUS), 
             (table_x + table_width, table_y + table_height - self.config.CORNER_RADIUS)),  # Right
            ((table_x + table_width - self.config.CORNER_RADIUS, table_y + table_height), 
             (table_x + self.config.CORNER_RADIUS, table_y + table_height)),  # Bottom
            ((table_x, table_y + table_height - self.config.CORNER_RADIUS), 
             (table_x, table_y + self.config.CORNER_RADIUS))  # Left
        ]
        
        # Create boundaries in order: corner, wall, corner, wall, etc.
        for i in range(4):
            # Add corner
            center, start_angle = corners[i]
            shapes.extend(self.create_corner_segments(boundary_body, center, start_angle))
            
            # Add wall
            start_point, end_point = walls[i]
            wall = pymunk.Segment(boundary_body, start_point, end_point, 1)
            wall.friction = self.config.WALL_FRICTION
            wall.elasticity = self.config.WALL_ELASTICITY
            shapes.append(wall)
        
        # Add body and all shapes to space
        self.space.add(boundary_body, *shapes)
        return boundary_body, shapes
    
    def create_ball(self, width, height):
        """Create the ball with physics properties"""
        # Create ball body and shape
        inertia = pymunk.moment_for_circle(self.config.BALL_MASS, 0, self.config.BALL_RADIUS, (0, 0))
        ball_body = pymunk.Body(self.config.BALL_MASS, inertia)
        ball_shape = pymunk.Circle(ball_body, self.config.BALL_RADIUS, (0, 0))
        
        # Set ball properties - ice hockey puck behavior
        ball_shape.friction = self.config.BALL_FRICTION
        ball_shape.elasticity = self.config.BALL_ELASTICITY
        
        # Position and velocity
        self.reset_ball(ball_body, width, height)
        
        # Add to space
        self.space.add(ball_body, ball_shape)
        return ball_body, ball_shape
    
    def reset_ball(self, ball_body, width, height):
        """Reset ball position to move toward bottom-right corner"""
        table_width = width - 2 * self.config.TABLE_MARGIN
        table_height = height - 2 * self.config.TABLE_MARGIN
        table_x = self.config.TABLE_MARGIN
        table_y = self.config.TABLE_MARGIN
        
        # Position ball in the middle-left area of the table
        start_x = table_x + self.config.CORNER_RADIUS + 50  # Left side of table
        start_y = table_y + table_height // 2   # Middle height
        
        ball_body.position = start_x, start_y
        
        # Set velocity to move toward the bottom-right corner
        speed = 300  # Moderate speed for good observation
        
        # Angle toward the bottom-right corner (shallow downward angle)
        angle = math.radians(15)  # 15 degrees downward toward bottom-right
        
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)
        
        ball_body.velocity = vx, vy
    
    def step(self, dt):
        """Step the physics simulation"""
        self.space.step(dt)