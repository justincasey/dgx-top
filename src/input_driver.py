from __future__ import annotations

import os
import selectors
from codecs import getincrementaldecoder

from textual._loop import loop_last
from textual._parser import ParseError
from textual._xterm_parser import XTermParser
from textual.drivers.linux_driver import LinuxDriver


def _make_terminal_decoder():
    """Build Textual's incremental UTF-8 decoder without fatal byte errors."""
    return getincrementaldecoder("utf-8")(errors="replace").decode


class ResilientLinuxDriver(LinuxDriver):
    """Textual 8.2.8 Linux driver that replaces malformed UTF-8 input."""

    def run_input_thread(self) -> None:
        """Wait for input and dispatch events."""
        selector = selectors.SelectSelector()
        selector.register(self.fileno, selectors.EVENT_READ)

        fileno = self.fileno
        event_read = selectors.EVENT_READ

        parser = XTermParser(self._debug)
        feed = parser.feed
        tick = parser.tick

        decode = _make_terminal_decoder()
        read = os.read

        def process_selector_events(
            selector_events: list[tuple[selectors.SelectorKey, int]],
            final: bool = False,
        ) -> None:
            """Process readable terminal input."""
            for last, (_selector_key, mask) in loop_last(selector_events):
                if mask & event_read:
                    unicode_data = decode(read(fileno, 1024 * 4), final=final and last)
                    if not unicode_data:
                        # This can occur if stdin is piped.
                        break
                    for event in feed(unicode_data):
                        self.process_message(event)
            for event in tick():
                self.process_message(event)

        try:
            while not self.exit_event.is_set():
                process_selector_events(selector.select(0.1))
            process_selector_events(selector.select(0.1), final=True)
            selector.unregister(self.fileno)
            unicode_data = decode(b"", final=True)
            if unicode_data:
                for event in feed(unicode_data):
                    self.process_message(event)
        finally:
            selector.close()
            try:
                for event in feed(""):
                    pass
            except (EOFError, ParseError):
                pass
