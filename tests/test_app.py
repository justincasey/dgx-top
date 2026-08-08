from pathlib import Path

from config import configure


def _config(path: Path, theme: str | None = None) -> None:
    theme_line = (
        f"""theme = "{theme}"
"""
        if theme
        else ""
    )
    path.write_text(
        f"""
[app]
poll_interval = 7
history_length = 25
{theme_line}
[[nodes]]
label = "primary"
ssh_target = "primary"
vllm_url = "http://primary.example.com:8000"
"""
    )


def test_app_uses_configured_poll_interval_and_node_count(tmp_path: Path):
    path = tmp_path / "config.toml"
    _config(path)
    settings = configure(path)

    from app import DGXTop

    app = DGXTop()
    assert app._current_interval() == 7
    assert len(settings.nodes) == 1


def test_default_theme_applies_through_app(tmp_path: Path):
    path = tmp_path / "config.toml"
    _config(path)  # no theme key -> default
    configure(path)

    from app import DGXTop

    app = DGXTop()
    assert app.theme == "dgx-dark"


def test_app_applies_configured_theme(tmp_path: Path):
    path = tmp_path / "config.toml"
    _config(path, theme="tokyo-night-storm")
    configure(path)

    from app import DGXTop

    app = DGXTop()
    assert app.theme == "tokyo-night-storm"
    assert app.current_theme.name == "tokyo-night-storm"


def test_custom_themes_are_registered(tmp_path: Path):
    path = tmp_path / "config.toml"
    _config(path)
    configure(path)

    from app import DGXTop

    app = DGXTop()
    assert app.get_theme("dgx-dark") is not None
    assert app.get_theme("tokyo-night-storm") is not None
    assert app.get_theme("tokyo-night-light") is not None
    assert app.get_theme("tokyo-night") is not None  # Textual built-in
