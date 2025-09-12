# Rendering and visualization logic for MiniAirHockeyEnv

# Add functions/classes here for pygame-based rendering.

def render_env(env):
    try:
        import pygame
    except Exception:
        return
    if getattr(env, '_closed', False):
        return
    if env.viewer is None:
        pygame.init()
        env.viewer = pygame.display.set_mode((env.W, env.H))
        pygame.display.set_caption('Mini Air Hockey (SAC baseline)')
        env.clock = pygame.time.Clock()
        env.font = pygame.font.SysFont(None, 24)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            env._closed = True
            try:
                pygame.quit()
            except Exception:
                pass
            env.viewer = None
            try:
                print('Render window closed by user — exiting process immediately')
            except Exception:
                pass
            import os
            os._exit(0)
    surface = env.viewer
    table_color = (240, 240, 255)
    border_color = (60, 60, 80)
    center_line_color = (180, 0, 0)
    faceoff_circle_color = (0, 120, 255)
    goal_color = (200, 200, 200)
    paddle1_color = (220, 40, 40)  # Red for paddle 1
    paddle2_color = (40, 40, 220)  # Blue for paddle 2
    puck_color = (60, 60, 60)
    margin = 32
    table_rect = pygame.Rect(margin, margin, env.W - 2 * margin, env.H - 2 * margin)
    surface.fill((30, 30, 40))
    pygame.draw.rect(surface, table_color, table_rect)
    pygame.draw.rect(surface, border_color, table_rect, width=6)
    pygame.draw.line(surface, center_line_color, (env.W // 2, margin), (env.W // 2, env.H - margin), 4)
    pygame.draw.circle(surface, faceoff_circle_color, (env.W // 2, env.H // 2), 48, 3)
    goal_length = 120
    goal_height = 16
    pygame.draw.rect(surface, goal_color, pygame.Rect(margin - goal_height // 2, (env.H - goal_length) // 2, goal_height, goal_length), border_radius=8)
    pygame.draw.rect(surface, goal_color, pygame.Rect(env.W - margin - goal_height // 2, (env.H - goal_length) // 2, goal_height, goal_length), border_radius=8)
    pygame.draw.circle(surface, puck_color, (int(env.ball_x), int(env.ball_y)), int(env.ball_radius))
    pygame.draw.circle(surface, paddle1_color, (int(env.paddle_x), int(env.paddle_y)), int(env.paddle_radius))
    pygame.draw.circle(surface, paddle2_color, (int(env.paddle2_x), int(env.paddle2_y)), int(env.paddle_radius))
    ep_txt = env.font.render(
        f"EP len={getattr(env, '_ep_length', 0)} r={getattr(env, '_ep_reward', 0.0):.2f} touches={getattr(env, '_ep_touches', 0)} consecutive: {getattr(env, '_ep_consecutive_touches', 0)}",
        True, (255,255,255))
    ep_rect = ep_txt.get_rect(center=(env.W // 2, 16))
    surface.blit(ep_txt, ep_rect)
    pygame.display.flip()
    env.clock.tick(60)
