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


def test_themes_lists_default_and_tokyo_family(capsys):
    assert main(["themes"]) == 0
    out = capsys.readouterr().out
    assert "default theme: dgx-dark" in out
    for name in ("tokyo-night", "tokyo-night-storm", "tokyo-night-light", "nord"):
        assert name in out
    assert "tokyo-night-light (light)" in out
    assert "tokyo-night-storm" in out and "tokyo-night-storm (light)" not in out


def test_unknown_theme_errors_before_running(tmp_path: Path, capsys):
    path = tmp_path / "config.toml"
    path.write_text(
        """
[app]
poll_interval = 5
history_length = 40

[[nodes]]
label = "primary"
ssh_target = "primary"
vllm_url = "http://primary.example.com:8000"
"""
    )

    assert main(["--config", str(path), "--theme", "neon", "check"]) == 2
    assert "unknown theme" in capsys.readouterr().err
