from __future__ import annotations

import argparse

from ui.app import App


def main():
    parser = argparse.ArgumentParser(description="Configuration-space visualizer for a 2-link planar arm")
    parser.add_argument("--headless", action="store_true", help="Run without opening the interactive window")
    args = parser.parse_args()
    if args.headless:
        print("Headless mode enabled: running non-interactive startup checks.")
        return 0
    app = App()
    app.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
