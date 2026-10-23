from __future__ import annotations

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    print(f"Project root: {root}")
    print("Headless startup check passed.")


if __name__ == "__main__":
    main()
