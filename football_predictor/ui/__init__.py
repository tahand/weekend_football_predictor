"""Kivy user interface. Call run() to start the desktop app."""

import sys

PREFERRED_SIZE = (1400, 860)
MINIMUM_SIZE = (1100, 700)
# Room for the title bar and window borders, which sit outside Kivy's client area.
FRAME_WIDTH = 16
FRAME_HEIGHT = 40


def _work_area() -> tuple[int, int, int, int] | None:
    """Return (left, top, width, height) of the desktop excluding the taskbar, if known."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    rect = wintypes.RECT()
    SPI_GETWORKAREA = 0x0030
    if not ctypes.windll.user32.SystemParametersInfoW(SPI_GETWORKAREA, 0, ctypes.byref(rect), 0):
        return None
    return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top


def _window_geometry() -> dict[str, int]:
    """Choose a window size and position that fits on the current screen."""
    width, height = PREFERRED_SIZE
    min_width, min_height = MINIMUM_SIZE
    area = _work_area()
    if area is None:
        return {"width": width, "height": height, "minimum_width": min_width, "minimum_height": min_height}
    left, top, area_width, area_height = area
    width = min(width, area_width - FRAME_WIDTH)
    height = min(height, area_height - FRAME_HEIGHT)
    return {
        "width": width,
        "height": height,
        "minimum_width": min(min_width, width),
        "minimum_height": min(min_height, height),
        "left": left + (area_width - width) // 2,
        "top": top + FRAME_HEIGHT - FRAME_WIDTH // 2 + (area_height - FRAME_HEIGHT - height) // 2,
    }


def run() -> None:
    import os

    os.environ.setdefault("KIVY_NO_ARGS", "1")
    from kivy.config import Config

    # Window settings must be applied before the window is created.
    geometry = _window_geometry()
    if "left" in geometry:
        Config.set("graphics", "position", "custom")
    for key, value in geometry.items():
        Config.set("graphics", key, str(value))
    Config.set("input", "mouse", "mouse,multitouch_on_demand")
    Config.set("kivy", "exit_on_escape", "0")

    from .app import PredictorApp

    PredictorApp().run()
