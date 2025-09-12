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
    WALL_ELASTICITY = 1.0  # Wall bounce - 1.0 = perfectly elastic (no energy loss)
    
    # Ball properties
    BALL_FRICTION = 0.3  # Ball friction
    BALL_ELASTICITY = 1.0  # Ball bounce - 1.0 = perfectly elastic (hard collision)
    BALL_MASS = 1
    BALL_RADIUS = 12  # Realistic air hockey puck size
    
    # Paddle properties
    PADDLE_MASS = 2
    PADDLE_RADIUS = 20  # Slightly larger than ball
    PADDLE_FRICTION = 0.4
    PADDLE_ELASTICITY = 1.0  # Paddle bounce - 1.0 = perfectly elastic (hard collision)
    
    # Goal properties
    GOAL_WIDTH = 80  # Width of goal opening
    GOAL_DEPTH = 20  # How deep the goal extends beyond the table
    GOAL_POST_RADIUS = 8  # Radius of goal posts
    
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
        """Reset ball position to center, moving toward right goal"""
        table_width = width - 2 * self.config.TABLE_MARGIN
        table_height = height - 2 * self.config.TABLE_MARGIN
        table_x = self.config.TABLE_MARGIN
        table_y = self.config.TABLE_MARGIN
        
        # Position ball in the center of the table
        start_x = table_x + table_width // 2  # Center horizontally
        start_y = table_y + table_height // 2   # Center vertically
        
        ball_body.position = start_x, start_y
        
        # Set velocity to move toward the right goal (player's goal)
        speed = 300  # Moderate speed for good observation
        
        # Angle toward the right goal (slight angle for interesting gameplay)
        angle = math.radians(15)  # 15 degrees upward toward right goal
        
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)
        
        ball_body.velocity = vx, vy
    
    def create_paddles(self, width, height):
        """Create player and opponent paddles"""
        table_width = width - 2 * self.config.TABLE_MARGIN
        table_x = self.config.TABLE_MARGIN
        
        # Set center line for boundary checking
        self._center_x = table_x + table_width // 2
        
        table_height = height - 2 * self.config.TABLE_MARGIN
        table_y = self.config.TABLE_MARGIN
        
        # Player paddle (right side)
        player_x = table_x + table_width - 80  # Near right edge
        player_y = table_y + table_height // 2
        
        # Opponent paddle (left side)
        opponent_x = table_x + 80  # Near left edge
        opponent_y = table_y + table_height // 2
        
        # Create paddle bodies
        inertia = pymunk.moment_for_circle(self.config.PADDLE_MASS, 0, self.config.PADDLE_RADIUS, (0, 0))
        
        # Player paddle
        self.player_paddle_body = pymunk.Body(self.config.PADDLE_MASS, inertia)
        self.player_paddle_shape = pymunk.Circle(self.player_paddle_body, self.config.PADDLE_RADIUS, (0, 0))
        self.player_paddle_body.position = player_x, player_y
        self.player_paddle_shape.friction = self.config.PADDLE_FRICTION
        self.player_paddle_shape.elasticity = self.config.PADDLE_ELASTICITY
        
        # Opponent paddle
        self.opponent_paddle_body = pymunk.Body(self.config.PADDLE_MASS, inertia)
        self.opponent_paddle_shape = pymunk.Circle(self.opponent_paddle_body, self.config.PADDLE_RADIUS, (0, 0))
        self.opponent_paddle_body.position = opponent_x, opponent_y
        self.opponent_paddle_shape.friction = self.config.PADDLE_FRICTION
        self.opponent_paddle_shape.elasticity = self.config.PADDLE_ELASTICITY
        
        # Add to space
        self.space.add(self.player_paddle_body, self.player_paddle_shape)
        self.space.add(self.opponent_paddle_body, self.opponent_paddle_shape)
        
        return (self.player_paddle_body, self.player_paddle_shape), (self.opponent_paddle_body, self.opponent_paddle_shape)
    
    def create_goals(self, width, height):
        """Create goals with goal posts"""
        table_width = width - 2 * self.config.TABLE_MARGIN
        table_height = height - 2 * self.config.TABLE_MARGIN
        table_x = self.config.TABLE_MARGIN
        table_y = self.config.TABLE_MARGIN
        
        goal_shapes = []
        
        # Calculate goal positions (center of left and right walls)
        goal_center_y = table_y + table_height // 2
        goal_half_height = self.config.GOAL_WIDTH // 2  # Using GOAL_WIDTH for height
        
        # Left goal (opponent's goal)
        left_goal_x = table_x - self.config.GOAL_DEPTH
        left_goal_top_y = goal_center_y - goal_half_height
        left_goal_bottom_y = goal_center_y + goal_half_height
        
        # Right goal (player's goal)
        right_goal_x = table_x + table_width + self.config.GOAL_DEPTH
        right_goal_top_y = goal_center_y - goal_half_height
        right_goal_bottom_y = goal_center_y + goal_half_height
        
        # Create goal boundaries
        goal_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        
        # Left goal walls
        left_goal_back = pymunk.Segment(goal_body, 
                                       (left_goal_x, left_goal_top_y), 
                                       (left_goal_x, left_goal_bottom_y), 2)
        left_goal_top = pymunk.Segment(goal_body, 
                                      (left_goal_x, left_goal_top_y), 
                                      (table_x, left_goal_top_y), 2)
        left_goal_bottom = pymunk.Segment(goal_body, 
                                         (left_goal_x, left_goal_bottom_y), 
                                         (table_x, left_goal_bottom_y), 2)
        
        # Right goal walls
        right_goal_back = pymunk.Segment(goal_body, 
                                        (right_goal_x, right_goal_top_y), 
                                        (right_goal_x, right_goal_bottom_y), 2)
        right_goal_top = pymunk.Segment(goal_body, 
                                       (right_goal_x, right_goal_top_y), 
                                       (table_x + table_width, right_goal_top_y), 2)
        right_goal_bottom = pymunk.Segment(goal_body, 
                                          (right_goal_x, right_goal_bottom_y), 
                                          (table_x + table_width, right_goal_bottom_y), 2)
        
        # Set goal wall properties
        goal_walls = [left_goal_back, left_goal_top, left_goal_bottom, 
                     right_goal_back, right_goal_top, right_goal_bottom]
        
        for wall in goal_walls:
            wall.friction = self.config.WALL_FRICTION
            wall.elasticity = self.config.WALL_ELASTICITY
        
        # Add to space
        self.space.add(goal_body, *goal_walls)
        goal_shapes.extend(goal_walls)
        
        # Store goal areas for collision detection
        self.left_goal_area = {
            'left': left_goal_x,
            'right': table_x,
            'top': left_goal_top_y,
            'bottom': left_goal_bottom_y
        }
        
        self.right_goal_area = {
            'left': table_x + table_width,
            'right': right_goal_x,
            'top': right_goal_top_y,
            'bottom': right_goal_bottom_y
        }
        
        return goal_body, goal_shapes
    
    def update_table_boundaries_with_goals(self, width, height):
        """Create table boundaries with goal openings"""
        boundary_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        shapes = []
        
        table_width = width - 2 * self.config.TABLE_MARGIN
        table_height = height - 2 * self.config.TABLE_MARGIN
        table_x = self.config.TABLE_MARGIN
        table_y = self.config.TABLE_MARGIN
        
        goal_center_y = table_y + table_height // 2
        goal_half_height = self.config.GOAL_WIDTH // 2
        
        # Corner centers (same as before)
        corners = [
            ((table_x + self.config.CORNER_RADIUS, table_y + self.config.CORNER_RADIUS), math.pi),
            ((table_x + table_width - self.config.CORNER_RADIUS, table_y + self.config.CORNER_RADIUS), 3*math.pi/2),
            ((table_x + table_width - self.config.CORNER_RADIUS, table_y + table_height - self.config.CORNER_RADIUS), 0),
            ((table_x + self.config.CORNER_RADIUS, table_y + table_height - self.config.CORNER_RADIUS), math.pi/2)
        ]
        
        # Modified wall segments with goal openings on left and right
        walls = [
            # Top wall (full)
            ((table_x + self.config.CORNER_RADIUS, table_y), 
             (table_x + table_width - self.config.CORNER_RADIUS, table_y)),
            # Right wall (split by goal)
            ((table_x + table_width, table_y + self.config.CORNER_RADIUS), 
             (table_x + table_width, goal_center_y - goal_half_height)),
            ((table_x + table_width, goal_center_y + goal_half_height), 
             (table_x + table_width, table_y + table_height - self.config.CORNER_RADIUS)),
            # Bottom wall (full)
            ((table_x + table_width - self.config.CORNER_RADIUS, table_y + table_height), 
             (table_x + self.config.CORNER_RADIUS, table_y + table_height)),
            # Left wall (split by goal)
            ((table_x, table_y + table_height - self.config.CORNER_RADIUS), 
             (table_x, goal_center_y + goal_half_height)),
            ((table_x, goal_center_y - goal_half_height), 
             (table_x, table_y + self.config.CORNER_RADIUS))
        ]
        
        # Create corners
        for i in range(4):
            center, start_angle = corners[i]
            shapes.extend(self.create_corner_segments(boundary_body, center, start_angle))
        
        # Create wall segments
        for start_point, end_point in walls:
            if start_point != end_point:  # Only create non-zero length segments
                wall = pymunk.Segment(boundary_body, start_point, end_point, 1)
                wall.friction = self.config.WALL_FRICTION
                wall.elasticity = self.config.WALL_ELASTICITY
                shapes.append(wall)
        
        # Add body and all shapes to space
        self.space.add(boundary_body, *shapes)
        return boundary_body, shapes
    
    def move_paddle(self, paddle_body, dx, dy, max_speed=200):
        """Move a paddle with velocity limits"""
        current_vel = paddle_body.velocity
        new_vel_x = max(-max_speed, min(max_speed, current_vel.x + dx))
        new_vel_y = max(-max_speed, min(max_speed, current_vel.y + dy))
        paddle_body.velocity = new_vel_x, new_vel_y
    
    def check_goal_scored(self, ball_body):
        """Check if ball is in a goal area"""
        ball_pos = ball_body.position
        
        # Check left goal (opponent's goal - player scores)
        if (self.left_goal_area['left'] <= ball_pos.x <= self.left_goal_area['right'] and
            self.left_goal_area['top'] <= ball_pos.y <= self.left_goal_area['bottom']):
            return 'player'
        
        # Check right goal (player's goal - opponent scores)
        if (self.right_goal_area['left'] <= ball_pos.x <= self.right_goal_area['right'] and
            self.right_goal_area['top'] <= ball_pos.y <= self.right_goal_area['bottom']):
            return 'opponent'
        
        return None
    
    def step(self, dt):
        """Step the physics simulation"""
        self.space.step(dt)
        
        # Enforce center line restriction continuously
        if hasattr(self, '_center_x'):
            # Player paddle stays on right side
            if hasattr(self, 'player_paddle_body'):
                pos_x, pos_y = self.player_paddle_body.position
                if pos_x < self._center_x:
                    self.player_paddle_body.position = self._center_x, pos_y
                    self.player_paddle_body.velocity = (0, self.player_paddle_body.velocity.y)
            
            # Opponent paddle stays on left side
            if hasattr(self, 'opponent_paddle_body'):
                pos_x, pos_y = self.opponent_paddle_body.position
                if pos_x > self._center_x:
                    self.opponent_paddle_body.position = self._center_x, pos_y
                    self.opponent_paddle_body.velocity = (0, self.opponent_paddle_body.velocity.y)