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


def test_chart_validation(tmp_path: Path):
    for chart in ("oscilloscope", "lines"):
        path = tmp_path / f"{chart}.toml"
        path.write_text(_with_app_line(f'chart = "{chart}"'))
        assert load_config(path).chart == chart

    path = tmp_path / "bogus.toml"
    path.write_text(_with_app_line('chart = "sparklines"'))
    with pytest.raises(ConfigError, match="app.chart"):
        load_config(path)

    # default is the oscilloscope
    assert load_config(_write(tmp_path)).chart == "oscilloscope"


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


def _with_node_line(extra: str, tmp_path: Path) -> Path:
    return _write(tmp_path, VALID_CONFIG.replace("worker = false", f"worker = false\n{extra}"))


def test_node_engine_key_is_optional_and_validated(tmp_path: Path):
    # Absent: the engine is inferred from the endpoint's own metrics.
    assert load_config(_write(tmp_path)).nodes[0].engine is None

    for engine in ("vllm", "sglang", "tensorfold"):
        path = _with_node_line(f'engine = "{engine}"', tmp_path)
        assert load_config(path).nodes[0].engine == engine

    # Case and surrounding whitespace are normalized away; a regression
    # that compares the raw value would reject valid mixed-case configs.
    for raw, expected in ((" SGLang ", "sglang"), ("VLLM", "vllm"), ("TensorFold", "tensorfold")):
        path = _with_node_line(f'engine = "{raw}"', tmp_path)
        assert load_config(path).nodes[0].engine == expected

    path = _with_node_line('engine = "tgi"', tmp_path)
    with pytest.raises(ConfigError, match="engine"):
        load_config(path)

    path = _with_node_line("engine = 3", tmp_path)
    with pytest.raises(ConfigError, match="engine"):
        load_config(path)


def test_engine_reaches_the_collector_units(tmp_path: Path):
    import config as cfg
    from config import configure

    configure(_with_node_line('engine = "sglang"', tmp_path))
    try:
        assert cfg.SPARK_UNITS[1]["engine"] == "sglang"
        assert cfg.SPARK_UNITS[1]["vllm_url"] == "http://spark-primary.example.com:8000"
    finally:
        configure(_write(tmp_path))


def test_declared_engines_match_the_collector_profiles():
    """``config.ENGINES`` gates the declared value and
    ``collector.ENGINE_PROFILES`` parses the same engine family, so a profile
    added without a declared value (or the reverse) leaves the two halves of
    one vocabulary out of step — the declared key would then name a row's
    engine while nothing could parse that engine's metrics. The collector
    imports config, so the two sets cannot be derived from one another — this
    is the pin that keeps them from drifting apart silently."""
    import collector
    import config as cfg

    assert cfg.ENGINES == set(collector.ENGINE_PROFILES)


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


# ─── save_node_endpoint (scan write-back) ────────────────────────────

SCAN_CONFIG = """# dgx-top cluster
[app]
poll_interval = 5  # seconds

# the head serves the model
[[nodes]]
label = "head"
ssh_target = "spark@head"
vllm_url = "http://head:8000"  # recipe port
engine = "vllm"

[[nodes]]
label = "worker"
ssh_target = "spark@worker"
vllm_url = "http://worker:8000"
worker = true
"""


def _scan_config(tmp_path: Path, mode: int = 0o600) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(SCAN_CONFIG)
    path.chmod(mode)
    return path


def test_save_edits_one_node_and_keeps_comments_and_other_keys(tmp_path: Path):
    from config import save_node_endpoint

    path = _scan_config(tmp_path)

    save_node_endpoint(path, 1, "head", "http://head:8001", "sglang")
    save_node_endpoint(path, 2, "worker", "http://worker:30000/", "sglang")

    text = path.read_text()
    for comment in ("# dgx-top cluster", "# seconds", "# the head serves the model"):
        assert comment in text
    assert 'vllm_url = "http://head:8001"  # recipe port' in text
    settings = load_config(path)
    head, worker = settings.nodes
    assert (head.vllm_url, head.engine, head.worker) == ("http://head:8001", "sglang", False)
    # No engine key is invented; worker is untouched.
    assert (worker.vllm_url, worker.engine, worker.worker) == ("http://worker:30000", None, True)
    assert "engine" not in text.split('label = "worker"')[1]


@pytest.mark.parametrize("mode", [0o600, 0o640])
def test_save_keeps_the_file_mode(tmp_path: Path, mode: int):
    from config import save_node_endpoint

    path = _scan_config(tmp_path, mode)

    save_node_endpoint(path, 1, "head", "http://head:8001", "vllm")

    assert path.stat().st_mode & 0o777 == mode


def test_save_refuses_a_node_whose_label_changed_and_leaves_bytes(tmp_path: Path):
    from config import save_node_endpoint

    path = _scan_config(tmp_path)
    before = path.read_bytes()

    with pytest.raises(ConfigError, match="no longer 'worker'"):
        save_node_endpoint(path, 1, "worker", "http://head:8001", "vllm")

    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["config.toml"]


def test_save_refuses_a_file_tomlkit_would_reformat_and_leaves_bytes(tmp_path: Path):
    """AC8: [[nodes]] split by another table does not round-trip through
    tomlkit (it would move [app] and its comments); refuse, never rewrite."""
    from config import save_node_endpoint

    path = tmp_path / "config.toml"
    path.write_text(
        "# first node\n"
        "[[nodes]]\n"
        'label = "head"\n'
        'ssh_target = "spark@head"\n'
        'vllm_url = "http://head:8000"\n'
        "\n"
        "# app section\n"
        "[app]\n"
        "poll_interval = 5\n"
        "\n"
        "# second node\n"
        "[[nodes]]\n"
        'label = "worker"\n'
        'ssh_target = "spark@worker"\n'
        'vllm_url = "http://worker:8000"\n'
    )
    assert [n.label for n in load_config(path).nodes] == ["head", "worker"]  # a valid file
    before = path.read_bytes()

    with pytest.raises(ConfigError, match=r"cannot be edited in place.*nodes\[1\] by hand"):
        save_node_endpoint(path, 1, "head", "http://head:8001", None)

    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["config.toml"]


def test_save_never_replaces_the_file_with_an_invalid_result(tmp_path: Path):
    from config import save_node_endpoint

    path = _scan_config(tmp_path)
    before = path.read_bytes()

    with pytest.raises(ConfigError, match="http"):
        save_node_endpoint(path, 1, "head", "ftp://head:8001", "vllm")

    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["config.toml"]


def test_save_through_a_symlink_edits_the_target(tmp_path: Path):
    from config import save_node_endpoint

    target = _scan_config(tmp_path)
    link = tmp_path / "link.toml"
    link.symlink_to(target)

    save_node_endpoint(link, 1, "head", "http://head:8001", "vllm")

    assert link.is_symlink()
    assert load_config(target).nodes[0].vllm_url == "http://head:8001"


def test_save_failure_to_write_is_a_config_error(tmp_path: Path):
    from config import save_node_endpoint

    directory = tmp_path / "ro"
    directory.mkdir()
    path = _scan_config(directory)
    before = path.read_bytes()
    directory.chmod(0o555)
    try:
        with pytest.raises(ConfigError, match="cannot write"):
            save_node_endpoint(path, 1, "head", "http://head:8001", "vllm")
    finally:
        directory.chmod(0o755)
    assert path.read_bytes() == before


def test_packaging_ships_every_src_module_and_tomlkit():
    """setuptools' ``py-modules`` is a hand-written allowlist: a module left
    out builds green and crashes non-editable installs at import."""
    from config import tomllib

    root = Path(__file__).resolve().parent.parent
    project = tomllib.loads((root / "pyproject.toml").read_text())
    shipped = set(project["tool"]["setuptools"]["py-modules"])
    modules = {p.stem for p in (root / "src").glob("*.py")}

    assert modules <= shipped, f"unshipped modules: {sorted(modules - shipped)}"
    assert any(dep.startswith("tomlkit") for dep in project["project"]["dependencies"])
