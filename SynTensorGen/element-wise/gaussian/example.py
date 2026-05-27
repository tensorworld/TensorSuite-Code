from __future__ import annotations

from pathlib import Path

from random_generator import generate_from_config


def main() -> None:
    config_path = Path(__file__).parent / "configs" / "random_tiny.yaml"
    result = generate_from_config(config_path)
    print(f"Generated Random bundle: {result['bundle_dir']}")
    print(f"TNS: {result['tns']}")
    print(f"Metadata: {result['metadata']}")


if __name__ == "__main__":
    main()
