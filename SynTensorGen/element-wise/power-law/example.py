from __future__ import annotations

from pathlib import Path

from kronweave import generate_from_config

ROOT = Path(__file__).resolve().parent


def main() -> int:
    result = generate_from_config(ROOT / "configs" / "cp_fastskg_tiny.yaml")
    print(result["bundle_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
