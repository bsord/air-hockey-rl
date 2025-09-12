import pygame
import pymunk
import pymunk.pygame_util
import math
import sys

class PhysicsAirHockeyEnv:
    def __init__(self, width=800, height=400):
        pygame.init()
        
        # Screen setup
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Physics Air Hockey Environment")
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.BLUE = (100, 150, 255)
        self.RED = (255, 100, 100)
        self.GRAY = (128, 128, 128)
        
        # Physics space
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)  # No gravity for air hockey
        self.space.damping = 0.995  # Slight damping like ice friction
        
        # Physics properties - adjust these to change behavior
        self.wall_friction = 0.07  # Wall friction (affects sliding behavior along walls)
        self.wall_elasticity = 0.8  # Wall bounce
        self.ball_friction = 0.3  # Ball friction
        self.ball_elasticity = 0.8  # Ball bounce
        
        # Drawing options for debug
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        
        # Table properties
        self.table_margin = 50
        self.table_width = width - 2 * self.table_margin
        self.table_height = height - 2 * self.table_margin
        self.table_x = self.table_margin
        self.table_y = self.table_margin
        self.corner_radius = 40  # Smaller radius for realistic air hockey table corners
        
        # Create physics bodies
        self.create_table_boundaries()
        self.create_ball()
        
        # Clock
        self.clock = pygame.time.Clock()
        self.running = True
    
    def create_corner_segments(self, boundary_body, center, start_angle, num_segments=12):
        """Create segments for a rounded corner"""
        segments = []
        for i in range(num_segments):
            angle1 = start_angle + i * math.pi/2 / num_segments
            angle2 = start_angle + (i + 1) * math.pi/2 / num_segments
            
            x1 = center[0] + self.corner_radius * math.cos(angle1)
            y1 = center[1] + self.corner_radius * math.sin(angle1)
            x2 = center[0] + self.corner_radius * math.cos(angle2)
            y2 = center[1] + self.corner_radius * math.sin(angle2)
            
            segment = pymunk.Segment(boundary_body, (x1, y1), (x2, y2), 1)
            segment.friction = self.wall_friction
            segment.elasticity = self.wall_elasticity
            segments.append(segment)
        return segments

    def create_table_boundaries(self):
        """Create rounded rectangle table boundaries using physics bodies"""
        boundary_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        shapes = []
        
        # Corner centers
        corners = [
            ((self.table_x + self.corner_radius, self.table_y + self.corner_radius), math.pi),        # Top-left: 180°
            ((self.table_x + self.table_width - self.corner_radius, self.table_y + self.corner_radius), 3*math.pi/2),  # Top-right: 270°
            ((self.table_x + self.table_width - self.corner_radius, self.table_y + self.table_height - self.corner_radius), 0),  # Bottom-right: 0°
            ((self.table_x + self.corner_radius, self.table_y + self.table_height - self.corner_radius), math.pi/2)  # Bottom-left: 90°
        ]
        
        # Wall segments (top, right, bottom, left)
        walls = [
            ((self.table_x + self.corner_radius, self.table_y), 
             (self.table_x + self.table_width - self.corner_radius, self.table_y)),  # Top
            ((self.table_x + self.table_width, self.table_y + self.corner_radius), 
             (self.table_x + self.table_width, self.table_y + self.table_height - self.corner_radius)),  # Right
            ((self.table_x + self.table_width - self.corner_radius, self.table_y + self.table_height), 
             (self.table_x + self.corner_radius, self.table_y + self.table_height)),  # Bottom
            ((self.table_x, self.table_y + self.table_height - self.corner_radius), 
             (self.table_x, self.table_y + self.corner_radius))  # Left
        ]
        
        # Create boundaries in order: corner, wall, corner, wall, etc.
        for i in range(4):
            # Add corner
            center, start_angle = corners[i]
            shapes.extend(self.create_corner_segments(boundary_body, center, start_angle))
            
            # Add wall
            start_point, end_point = walls[i]
            wall = pymunk.Segment(boundary_body, start_point, end_point, 1)
            wall.friction = self.wall_friction
            wall.elasticity = self.wall_elasticity
            shapes.append(wall)
        
        # Add body and all shapes to space
        self.space.add(boundary_body, *shapes)
    
    def create_ball(self):
        """Create the ball with physics properties"""
        # Ball properties
        mass = 1
        radius = 8
        
        # Create ball body and shape
        inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        self.ball_body = pymunk.Body(mass, inertia)
        self.ball_shape = pymunk.Circle(self.ball_body, radius, (0, 0))
        
        # Set ball properties - ice hockey puck behavior
        self.ball_shape.friction = self.ball_friction
        self.ball_shape.elasticity = self.ball_elasticity
        
        # Position and velocity
        self.reset_ball()
        
        # Add to space
        self.space.add(self.ball_body, self.ball_shape)
    
    def reset_ball(self):
        """Reset ball position along bottom wall to roll around the corner"""
        # Position ball along the bottom wall, a bit away from the center
        bottom_wall_y = self.table_y + self.table_height - 15  # Just above the bottom wall
        start_x = self.table_x + self.corner_radius + 100  # Start partway along the bottom
        
        self.ball_body.position = start_x, bottom_wall_y
        
        # Set velocity to move toward the right corner at a shallow angle
        speed = 400  # Increased speed for better visibility
        
        # Angle toward the right curved corner - shallow angle to roll around
        angle = math.radians(-15)  # Slightly upward angle toward the right corner
        
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)
        
        self.ball_body.velocity = vx, vy
    
    def draw_table(self):
        """Draw the table visually as a rounded rectangle"""
        # Create a surface to draw the rounded rectangle
        table_surface = pygame.Surface((self.table_width, self.table_height), pygame.SRCALPHA)
        
        # Draw rounded rectangle using pygame.draw functions
        # Main rectangle (without corners)
        inner_rect = pygame.Rect(self.corner_radius, 0, 
                                self.table_width - 2 * self.corner_radius, self.table_height)
        pygame.draw.rect(table_surface, self.BLUE, inner_rect)
        
        # Top and bottom strips
        top_rect = pygame.Rect(0, self.corner_radius, 
                              self.table_width, self.table_height - 2 * self.corner_radius)
        pygame.draw.rect(table_surface, self.BLUE, top_rect)
        
        # Four corner circles
        # Top-left
        pygame.draw.circle(table_surface, self.BLUE, 
                          (self.corner_radius, self.corner_radius), self.corner_radius)
        # Top-right  
        pygame.draw.circle(table_surface, self.BLUE, 
                          (self.table_width - self.corner_radius, self.corner_radius), self.corner_radius)
        # Bottom-right
        pygame.draw.circle(table_surface, self.BLUE, 
                          (self.table_width - self.corner_radius, self.table_height - self.corner_radius), self.corner_radius)
        # Bottom-left
        pygame.draw.circle(table_surface, self.BLUE, 
                          (self.corner_radius, self.table_height - self.corner_radius), self.corner_radius)
        
        # Blit the surface to the main screen
        self.screen.blit(table_surface, (self.table_x, self.table_y))
    
    def draw_ball(self):
        """Draw the ball"""
        pos = int(self.ball_body.position.x), int(self.ball_body.position.y)
        pygame.draw.circle(self.screen, self.RED, pos, 8)
        pygame.draw.circle(self.screen, self.BLACK, pos, 8, 2)
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.reset_ball()
                elif event.key == pygame.K_d:
                    # Toggle debug drawing
                    self.debug_draw = not getattr(self, 'debug_draw', False)
    
    def run(self):
        """Main game loop"""
        dt = 1.0 / 60.0  # 60 FPS
        
        while self.running:
            self.handle_events()
            
            # Step physics
            self.space.step(dt)
            
            # Draw everything
            self.screen.fill(self.WHITE)
            self.draw_table()
            self.draw_ball()
            
            # Optional debug drawing (press 'D' to toggle)
            if getattr(self, 'debug_draw', False):
                self.space.debug_draw(self.draw_options)
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()

def main():
    """Main function"""
    env = PhysicsAirHockeyEnv()
    print("Physics Air Hockey Environment")
    print("Controls:")
    print("- ESC: Quit")
    print("- SPACE: Reset ball")
    print("- D: Toggle debug physics drawing")
    env.run()

if __name__ == "__main__":
    main()