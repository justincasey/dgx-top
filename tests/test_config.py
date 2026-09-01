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


def _write_cluster(tmp_path: Path, n: int) -> Path:
    lines = [
        "[app]",
        "poll_interval = 3",
        "history_length = 60",
    ]
    for i in range(1, n + 1):
        lines += [
            "[[nodes]]",
            f'label = "node-{i}"',
            f'ssh_target = "node-{i}"',
            f'vllm_url = "http://192.0.2.{i}:8000"',
        ]
        if i > 1:
            lines.append("worker = true")
    return _write(tmp_path, "\n".join(lines) + "\n")


def test_cluster_accepts_up_to_twelve_nodes(tmp_path: Path):
    settings = load_config(_write_cluster(tmp_path, 12))
    assert len(settings.nodes) == 12


def test_cluster_rejects_more_than_twelve_nodes(tmp_path: Path):
    with pytest.raises(ConfigError, match="supports up to 12"):
        load_config(_write_cluster(tmp_path, 13))


def test_simulate_builds_synthetic_nodes_without_config(tmp_path: Path):
    settings = load_config(tmp_path / "missing.toml", simulate=8)
    assert len(settings.nodes) == 8
    assert settings.nodes[0].label == "spark-1"
    assert settings.nodes[7].worker is True
    assert settings.poll_interval == 5  # defaults when no [app] block


def test_default_theme_is_dgx_aeon(tmp_path: Path):
    settings = load_config(_write(tmp_path))

    assert settings.theme == "dgx-aeon"


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


def test_empty_theme_override_is_rejected(tmp_path: Path):
    with pytest.raises(ConfigError, match="non-empty theme"):
        load_config(_write(tmp_path), theme=" ")


def test_theme_override_is_validated(tmp_path: Path):
    with pytest.raises(ConfigError, match=r"unknown theme 'nope'"):
        load_config(_write(tmp_path), theme="nope")


def _with_app_line(line: str) -> str:
    return VALID_CONFIG.replace("history_length = 60", f"history_length = 60\n{line}")


def test_meter_treatment_validation(tmp_path: Path):
    for treatment in ("gradient", "spark", "tick", "line"):
        path = tmp_path / f"{treatment}.toml"
        path.write_text(_with_app_line(f'meter_treatment = "{treatment}"'))
        assert load_config(path).meter_treatment == treatment

    path = tmp_path / "bogus.toml"
    path.write_text(_with_app_line('meter_treatment = "blocks"'))
    with pytest.raises(ConfigError, match="meter_treatment"):
        load_config(path)

    # default preserves the current gradient look
    assert load_config(_write(tmp_path)).meter_treatment == "gradient"


def test_quiet_validation(tmp_path: Path):
    path = tmp_path / "quiet.toml"
    path.write_text(_with_app_line("quiet = true"))
    assert load_config(path).quiet is True

    path = tmp_path / "notquiet.toml"
    path.write_text(_with_app_line("quiet = false"))
    assert load_config(path).quiet is False

    path = tmp_path / "bogus.toml"
    path.write_text(_with_app_line('quiet = "yes"'))
    with pytest.raises(ConfigError, match="quiet"):
        load_config(path)

    assert load_config(_write(tmp_path)).quiet is False


def test_configure_resets_simulate_flag_without_simulate(tmp_path: Path):
    """A non-simulate configure must clear a prior --simulate flag (no leak)."""
    import config as cfg
    from config import configure

    configure(_write(tmp_path), simulate=5)
    try:
        assert cfg.SIMULATION_NODES == 5
        configure(_write(tmp_path))  # real config, no simulate
        assert cfg.SIMULATION_NODES is None
    finally:
        configure(_write(tmp_path), simulate=None)  # ensure clean
