from pathlib import Path

import pytest

from cli import main
from config import load_config


def test_init_creates_private_valid_example(tmp_path: Path):
    path = tmp_path / "config.toml"

    assert main(["--config", str(path), "init"]) == 0

    settings = load_config(path)
    assert len(settings.nodes) == 2
    assert all("example.com" in node.vllm_url for node in settings.nodes)
    assert path.stat().st_mode & 0o777 == 0o600


def test_init_does_not_overwrite_existing_config(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text("keep me")

    assert main(["--config", str(path), "init"]) == 2
    assert path.read_text() == "keep me"


def test_help_does_not_require_configuration(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    assert "Terminal monitoring" in capsys.readouterr().out
