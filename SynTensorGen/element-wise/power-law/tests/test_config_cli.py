from kronweave.cli import main
from kronweave.config import load_and_validate_config


def test_validate_tiny_config():
    cfg = load_and_validate_config("configs/zipf_tiny.yaml")
    assert cfg["generator"]["type"] == "zipf"


def test_cli_validate_config():
    assert main(["validate-config", "--config", "configs/zipf_tiny.yaml"]) == 0
