"""Optional tray icon showing the listening state (``pip install voice-forge[tray]``)."""

from __future__ import annotations

from typing import Any

from voice_forge.core import events as ev
from voice_forge.core.types import Status

COLORS = {
    Status.IDLE: (110, 110, 110),
    Status.LISTENING: (220, 40, 40),
    Status.TRANSCRIBING: (230, 190, 30),
    Status.THINKING: (180, 60, 200),
    Status.CONFIRMING: (240, 140, 0),
    Status.EXECUTING: (30, 180, 220),
}


def tray_available() -> bool:
    try:
        import PIL  # noqa: F401
        import pystray  # noqa: F401
    except Exception:
        return False
    return True


def _image(color: tuple[int, int, int]) -> Any:
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    ImageDraw.Draw(img).ellipse((8, 8, 56, 56), fill=(*color, 255))
    return img


class TrayStatus:
    """Bus subscriber that recolours a tray icon according to :class:`Status`."""

    def __init__(self) -> None:
        import pystray

        self._icon = pystray.Icon("voice-forge", _image(COLORS[Status.IDLE]), "voice-forge")

    def start(self) -> None:
        self._icon.run_detached()

    def stop(self) -> None:
        self._icon.stop()

    def __call__(self, event: ev.Event) -> None:
        if isinstance(event, ev.StatusChanged):
            self._icon.icon = _image(COLORS[event.status])
            self._icon.title = f"voice-forge · {event.status.value}"
