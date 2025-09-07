"""Package entrypoint mirroring the previous top-level CLI.

Usage:
  python -m airhockey train --visible
  python -m airhockey visualize
"""
import runpy
import sys
import argparse


def main(argv=None):
    argv = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(prog="airhockey", description="Run project subcommands")
    parser.add_argument("subcommand", nargs="?", choices=["train", "visualize"], help="subcommand to run")
    parser.add_argument("rest", nargs=argparse.REMAINDER, help="arguments forwarded to subcommand")

    # If no args provided, show help
    if not argv:
        parser.print_help()
        return

    # Manually split: first token is the subcommand, forward the rest unchanged.
    cmd = argv[0]
    if cmd == "train":
        forwarded = argv[1:]
        sys.argv = ["airhockey.agents.train_agent"] + forwarded
        runpy.run_module("airhockey.agents.train_agent", run_name="__main__")
    elif cmd == "visualize":
        forwarded = argv[1:]
        sys.argv = ["main"] + forwarded
        runpy.run_module("main", run_name="__main__")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
