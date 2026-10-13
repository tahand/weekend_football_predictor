"""Kivy user interface. Call run() to start the desktop app."""


def run() -> None:
    import os

    os.environ.setdefault("KIVY_NO_ARGS", "1")
    from kivy.config import Config

    # Window settings must be applied before the window is created.
    Config.set("graphics", "width", "1400")
    Config.set("graphics", "height", "860")
    Config.set("graphics", "minimum_width", "1100")
    Config.set("graphics", "minimum_height", "700")
    Config.set("input", "mouse", "mouse,multitouch_on_demand")
    Config.set("kivy", "exit_on_escape", "0")

    from .app import PredictorApp

    PredictorApp().run()
