from pathlib import Path

from config import configure


def test_app_uses_configured_poll_interval_and_node_count(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text(
        """
[app]
poll_interval = 7
history_length = 25

[[nodes]]
label = "primary"
ssh_target = "primary"
vllm_url = "http://primary.example.com:8000"
"""
    )
    settings = configure(path)

    from app import DGXTop

    app = DGXTop()
    assert app._current_interval() == 7
    assert len(settings.nodes) == 1
