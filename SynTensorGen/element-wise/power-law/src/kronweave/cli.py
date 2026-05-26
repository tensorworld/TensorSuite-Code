from __future__ import annotations

import argparse
from pathlib import Path
import sys

from kronweave.api import generate_from_config, validate_bundle
from kronweave.config import load_and_validate_config


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Path to YAML config")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kronweave")
    sub = parser.add_subparsers(dest="command", required=True)
    p_generate = sub.add_parser("generate", help="Generate bundle from YAML config")
    _add_config_arg(p_generate)
    p_validate = sub.add_parser("validate-config", help="Validate YAML config")
    _add_config_arg(p_validate)
    p_smoke = sub.add_parser("smoke-test", help="Generate and validate a smoke-test bundle")
    _add_config_arg(p_smoke)
    p_validate_bundle = sub.add_parser("validate-bundle", help="Validate a generated bundle")
    p_validate_bundle.add_argument("--bundle", required=True, help="Path to generated bundle directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate-config":
            config = load_and_validate_config(args.config)
            print(f"Config valid: {args.config}")
            print(f"Generator: {config['generator']['type']}")
            return 0
        if args.command in {"generate", "smoke-test"}:
            result = generate_from_config(args.config)
            print(f"Generated bundle: {result['bundle_dir']}")
            print(f"TNS: {result['tns']}")
            print(f"TNSB: {result['tnsb']}")
            print(f"Metadata: {result['metadata']}")
            print(f"README: {result['readme']}")
            print(f"Config snapshot: {result['config']}")
            if args.command == "smoke-test":
                validate_bundle(result["bundle_dir"])
                print("Smoke test: PASS")
            return 0
        if args.command == "validate-bundle":
            validate_bundle(args.bundle)
            print(f"Bundle valid: {args.bundle}")
            return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
