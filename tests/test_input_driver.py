from __future__ import annotations

import os
from pathlib import Path
from threading import Event, Thread

from textual import events
from textual._xterm_parser import XTermParser

from config import configure
from input_driver import ResilientLinuxDriver, _make_terminal_decoder


def _config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "[app]",
                "poll_interval = 5",
                "history_length = 25",
                "[[nodes]]",
                'label = "head"',
                'ssh_target = "head"',
                'vllm_url = "http://192.0.2.10:8000"',
            ]
        )
        + "\n"
    )


def test_terminal_decoder_replaces_invalid_bytes_and_preserves_valid_text():
    decode = _make_terminal_decoder()

    assert decode(b"a\x80b", final=True) == "a\ufffdb"


def test_terminal_decoder_preserves_split_utf8_and_finalizes_partial_input():
    decode = _make_terminal_decoder()
    assert decode(b"\xe2", final=False) == ""
    assert decode(b"\x82\xac", final=False) == "\u20ac"

    incomplete = _make_terminal_decoder()
    assert incomplete(b"\xe2", final=False) == ""
    assert incomplete(b"", final=True) == "\ufffd"


def test_replacement_does_not_join_fragments_into_control_sequence():
    decode = _make_terminal_decoder()
    parser = XTermParser(debug=False)

    parsed = list(parser.feed(decode(b"\x1b[\x80A", final=True)))
    parsed.extend(parser.feed(""))

    keys = [message.key for message in parsed if isinstance(message, events.Key)]
    assert "up" not in keys
    assert "replacement_character" in keys


def _run_driver_bytes(payload: bytes, expected_messages: int):
    read_fd, write_fd = os.pipe()
    driver = object.__new__(ResilientLinuxDriver)
    driver.fileno = read_fd
    driver.exit_event = Event()
    driver._debug = False
    messages = []
    received = Event()

    def capture(message):
        messages.append(message)
        if len(messages) >= expected_messages:
            received.set()

    driver.process_message = capture
    input_thread = Thread(target=driver.run_input_thread)
    input_thread.start()
    try:
        os.write(write_fd, payload)
        assert received.wait(1)
    finally:
        driver.exit_event.set()
        input_thread.join(1)
        os.close(write_fd)
        os.close(read_fd)

    assert not input_thread.is_alive()
    return messages


def test_driver_loop_dispatches_valid_text_around_invalid_byte():
    messages = _run_driver_bytes(b"a\x80b", expected_messages=3)

    characters = [message.character for message in messages if isinstance(message, events.Key)]
    assert characters == ["a", "\ufffd", "b"]


def test_driver_loop_preserves_valid_escape_sequence():
    messages = _run_driver_bytes(b"\x1b[A", expected_messages=1)

    assert [message.key for message in messages if isinstance(message, events.Key)] == ["up"]


def test_driver_loop_finalizes_buffered_utf8_on_shutdown(monkeypatch):
    read_fd, write_fd = os.pipe()
    driver = object.__new__(ResilientLinuxDriver)
    driver.fileno = read_fd
    driver.exit_event = Event()
    driver._debug = False
    messages = []
    driver.process_message = messages.append
    read_once = Event()
    real_read = os.read

    def tracked_read(fileno, size):
        data = real_read(fileno, size)
        read_once.set()
        return data

    monkeypatch.setattr("input_driver.os.read", tracked_read)
    input_thread = Thread(target=driver.run_input_thread)
    input_thread.start()
    try:
        os.write(write_fd, b"\xe2")
        assert read_once.wait(1)
    finally:
        driver.exit_event.set()
        input_thread.join(1)
        os.close(write_fd)
        os.close(read_fd)

    assert not input_thread.is_alive()
    characters = [message.character for message in messages if isinstance(message, events.Key)]
    assert characters == ["\ufffd"]


def test_dgxtop_selects_resilient_terminal_driver(tmp_path: Path):
    from app import DGXTop

    config_path = tmp_path / "config.toml"
    _config(config_path)
    configure(config_path)

    app = DGXTop()

    assert app.driver_class is ResilientLinuxDriver


def test_dgxtop_preserves_explicit_textual_driver(tmp_path: Path, monkeypatch):
    from textual import constants
    from textual.drivers.headless_driver import HeadlessDriver

    from app import DGXTop

    config_path = tmp_path / "config.toml"
    _config(config_path)
    configure(config_path)
    monkeypatch.setattr(
        constants,
        "DRIVER",
        "textual.drivers.headless_driver:HeadlessDriver",
    )

    app = DGXTop()

    assert app.driver_class is HeadlessDriver
