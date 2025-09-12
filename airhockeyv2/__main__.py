#!/usr/bin/env python3
"""
Main entry point for airhockeyv2 package

Allows running the package directly with: python airhockeyv2
"""

import sys
import argparse
from envs.PhysicsAirHockey import PhysicsAirHockeyEnv


def main():
    """Launch the air hockey environment"""
    parser = argparse.ArgumentParser(
        description='Physics Air Hockey Environment v2',
        prog='python airhockeyv2'
    )
    parser.add_argument('--maximize', action='store_true', 
                        help='Start with maximized window')
    parser.add_argument('--width', type=int, default=800,
                        help='Window width (default: 800)')
    parser.add_argument('--height', type=int, default=400,
                        help='Window height (default: 400)')
    parser.add_argument('--train', action='store_true',
                        help='Launch training mode (coming soon)')
    parser.add_argument('--tune', action='store_true',
                        help='Launch hyperparameter tuning (coming soon)')
    
    args = parser.parse_args()
    
    if args.train:
        print("🚀 Training mode - Coming soon!")
        print("Use: python airhockeyv2/train.py")
        return
    
    if args.tune:
        print("🔧 Tuning mode - Coming soon!")
        print("Use: python airhockeyv2/tune.py")
        return
    
    # Default: Play mode
    print("🏒 Physics Air Hockey Environment v2")
    print("Controls:")
    print("- ESC: Quit")
    print("- SPACE: Reset ball")
    print("- D: Toggle debug physics drawing")
    print("- M: Toggle window maximize/restore")
    print()
    
    try:
        env = PhysicsAirHockeyEnv(
            width=args.width, 
            height=args.height, 
            maximize_window=args.maximize
        )
        env.run()
    except KeyboardInterrupt:
        print("\nGame interrupted by user")
    except Exception as e:
        print(f"Error starting environment: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()