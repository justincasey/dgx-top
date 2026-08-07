from pathlib import Path

import pytest

from config import ConfigError, load_config

VALID_CONFIG = """
[app]
poll_interval = 3
history_length = 60

[[nodes]]
label = "primary"
ssh_target = "spark-primary"
vllm_url = "http://spark-primary.example.com:8000/"
worker = false
"""


def _write(tmp_path: Path, text: str = VALID_CONFIG) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text)
    return path


def test_load_config_supports_ssh_alias_and_separate_vllm_url(tmp_path: Path):
    settings = load_config(_write(tmp_path))

    assert settings.poll_interval == 3
    assert settings.history_length == 60
    assert settings.nodes[0].ssh_target == "spark-primary"
    assert settings.nodes[0].vllm_url == "http://spark-primary.example.com:8000"


def test_missing_config_has_actionable_error(tmp_path: Path):
    with pytest.raises(ConfigError, match="dgx-top init"):
        load_config(tmp_path / "missing.toml")


@pytest.mark.parametrize(
    "replacement, message",
    [
        ('ssh_target = "spark-primary"', "ssh_target"),
        ('vllm_url = "http://spark-primary.example.com:8000/"', "vllm_url"),
    ],
)
def test_required_node_fields_are_validated(tmp_path: Path, replacement: str, message: str):
    text = VALID_CONFIG.replace(replacement, "")

    with pytest.raises(ConfigError, match=message):
        load_config(_write(tmp_path, text))


def test_more_than_two_nodes_is_rejected(tmp_path: Path):
    node = """
[[nodes]]
label = "extra"
ssh_target = "extra"
vllm_url = "http://extra.example.com:8000"
"""
    text = VALID_CONFIG + node + node.replace('"extra"', '"extra-2"', 1)

    with pytest.raises(ConfigError, match="one or two"):
        load_config(_write(tmp_path, text))


def test_default_theme_is_dgx_dark(tmp_path: Path):
    settings = load_config(_write(tmp_path))

    assert settings.theme == "dgx-dark"


def test_theme_option_is_loaded(tmp_path: Path):
    text = VALID_CONFIG.replace("poll_interval = 3", 'poll_interval = 3\ntheme = "nord"')
    settings = load_config(_write(tmp_path, text))

    assert settings.theme == "nord"


def test_unknown_theme_is_rejected_with_options(tmp_path: Path):
    text = VALID_CONFIG.replace("poll_interval = 3", 'poll_interval = 3\ntheme = "neon"')

    with pytest.raises(ConfigError, match=r"unknown theme 'neon'"):
        load_config(_write(tmp_path, text))


def test_theme_override_wins_over_config(tmp_path: Path):
    text = VALID_CONFIG.replace("poll_interval = 3", 'poll_interval = 3\ntheme = "nord"')
    settings = load_config(_write(tmp_path, text), theme="tokyo-night-storm")

    assert settings.theme == "tokyo-night-storm"


def test_theme_override_is_validated(tmp_path: Path):
    with pytest.raises(ConfigError, match=r"unknown theme 'nope'"):
        load_config(_write(tmp_path), theme="nope")
