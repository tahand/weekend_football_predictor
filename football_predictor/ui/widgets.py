"""Reusable Kivy widgets: flat buttons, tables, bar charts and popups."""

from __future__ import annotations

import os
import string
from pathlib import Path
from typing import Callable, Sequence

from kivy.metrics import dp
from kivy.properties import BooleanProperty, ColorProperty, ListProperty, NumericProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.spinner import Spinner
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.widget import Widget

from . import theme


class FlatButton(Button):
    bg = ColorProperty(theme.PANEL_2)
    bg_down = ColorProperty(theme.ACCENT_DARK)


class PrimaryButton(FlatButton):
    pass


class DangerButton(FlatButton):
    pass


class NavButton(ToggleButton):
    pass


class ListToggle(ToggleButton):
    item_id = StringProperty("")


class Cell(Label):
    bg = ColorProperty((0, 0, 0, 0))


class Bar(Widget):
    value = NumericProperty(0)
    bar_color = ColorProperty(theme.ACCENT)


class DataTable(BoxLayout):
    """A simple non-virtualised table; fine for a few hundred rows."""

    row_height = NumericProperty(dp(28))
    zebra = BooleanProperty(True)

    def set_data(self, columns: Sequence[str], rows: Sequence[Sequence], weights: Sequence[float] | None = None,
                 aligns: Sequence[str] | None = None, row_colors: Sequence | None = None,
                 bold_rows: Sequence[bool] | None = None, empty_text: str = "Nothing to show") -> None:
        self.clear_widgets()
        n = len(columns)
        weights = list(weights or [1] * n)
        aligns = list(aligns or ["left"] * n)
        self.add_widget(self._row(columns, weights, aligns, theme.HEADER, bold=True))
        if not rows:
            self.add_widget(self._row([empty_text] + [""] * (n - 1), weights, aligns, (0, 0, 0, 0),
                                      color=theme.MUTED))
        for i, values in enumerate(rows):
            bg = row_colors[i] if row_colors and row_colors[i] else (
                theme.ROW_ALT if self.zebra and i % 2 else theme.ROW)
            bold = bool(bold_rows and bold_rows[i])
            self.add_widget(self._row(values, weights, aligns, bg, bold=bold))

    def _row(self, values, weights, aligns, bg, bold=False, color=None):
        row = BoxLayout(orientation="horizontal", size_hint_y=None, height=self.row_height)
        for value, weight, align in zip(values, weights, aligns):
            cell = Cell(text="" if value is None else str(value), size_hint_x=weight, halign=align,
                        bg=bg, bold=bold)
            if color:
                cell.color = color
            row.add_widget(cell)
        return row


class BarChart(BoxLayout):
    """Horizontal bars: each item is (label, fraction 0..1, value text, colour or None)."""

    def set_bars(self, items: Sequence[tuple], label_width: float = 0.32) -> None:
        self.clear_widgets()
        if not items:
            self.add_widget(Cell(text="Nothing to show yet", color=theme.MUTED, size_hint_y=None, height=dp(28)))
            return
        for label, fraction, text, colour in items:
            row = BoxLayout(size_hint_y=None, height=dp(26), spacing=dp(8))
            row.add_widget(Cell(text=str(label), size_hint_x=label_width, halign="right"))
            row.add_widget(Bar(value=max(0.0, min(1.0, fraction)), bar_color=colour or theme.ACCENT,
                               size_hint_x=1 - label_width - 0.14))
            row.add_widget(Cell(text=str(text), size_hint_x=0.14))
            self.add_widget(row)


class WrapLabel(Label):
    pass


class MutedLabel(WrapLabel):
    pass


class SectionTitle(Label):
    pass


# ---------------------------------------------------------------- popups

def _windows_drives() -> list[str]:
    if os.name != "nt":
        return ["/"]
    return [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]


def message(title: str, text: str, folder: Path | None = None) -> Popup:
    content = BoxLayout(orientation="vertical", padding=dp(12), spacing=dp(12))
    content.add_widget(WrapLabel(text=text))
    content.add_widget(Widget())
    buttons = BoxLayout(size_hint_y=None, height=dp(38), spacing=dp(8))
    popup = Popup(title=title, content=content, size_hint=(0.6, 0.45), separator_color=theme.ACCENT)
    if folder is not None:
        open_btn = FlatButton(text="Open folder")
        open_btn.bind(on_release=lambda *_: os.startfile(folder) if hasattr(os, "startfile") else None)
        buttons.add_widget(open_btn)
    ok = PrimaryButton(text="OK")
    ok.bind(on_release=lambda *_: popup.dismiss())
    buttons.add_widget(ok)
    content.add_widget(buttons)
    popup.open()
    return popup


def confirm(title: str, text: str, on_yes: Callable[[], None], yes_text: str = "Yes") -> Popup:
    content = BoxLayout(orientation="vertical", padding=dp(12), spacing=dp(12))
    content.add_widget(WrapLabel(text=text))
    content.add_widget(Widget())
    buttons = BoxLayout(size_hint_y=None, height=dp(38), spacing=dp(8))
    popup = Popup(title=title, content=content, size_hint=(0.5, 0.35), separator_color=theme.ACCENT)
    cancel = FlatButton(text="Cancel")
    cancel.bind(on_release=lambda *_: popup.dismiss())
    yes = DangerButton(text=yes_text)

    def _yes(*_):
        popup.dismiss()
        on_yes()

    yes.bind(on_release=_yes)
    buttons.add_widget(cancel)
    buttons.add_widget(yes)
    content.add_widget(buttons)
    popup.open()
    return popup


class ImportPopup(Popup):
    """CSV file picker with a drive selector and multi-select."""

    def __init__(self, start_path: str, on_files: Callable[[list[str]], None], **kwargs):
        super().__init__(title="Import CSV files", size_hint=(0.85, 0.9), separator_color=theme.ACCENT, **kwargs)
        self.callback = on_files
        start = start_path if start_path and os.path.isdir(start_path) else str(Path.home())

        root = BoxLayout(orientation="vertical", spacing=dp(8), padding=dp(8))
        top = BoxLayout(size_hint_y=None, height=dp(36), spacing=dp(8))
        drives = _windows_drives()
        current_drive = os.path.splitdrive(start)[0] + "\\" if os.name == "nt" else "/"
        self.drive = Spinner(text=current_drive if current_drive in drives else drives[0], values=drives,
                             size_hint_x=None, width=dp(80))
        self.drive.bind(text=lambda _, value: setattr(self.chooser, "path", value))
        up = FlatButton(text="Up", size_hint_x=None, width=dp(60))
        up.bind(on_release=lambda *_: self._go_up())
        self.path_label = Cell(text=start)
        top.add_widget(self.drive)
        top.add_widget(up)
        top.add_widget(self.path_label)
        root.add_widget(top)

        self.chooser = FileChooserListView(path=start, filters=["*.csv", "*.CSV"], multiselect=True)
        self.chooser.bind(path=lambda _, value: setattr(self.path_label, "text", value))
        self.chooser.bind(on_submit=lambda *_: self._import())
        root.add_widget(self.chooser)

        root.add_widget(MutedLabel(text="Click files to select several. Results and fixtures can be imported "
                                        "together; played matches are added to the history and unplayed "
                                        "matches replace the current fixture list."))
        buttons = BoxLayout(size_hint_y=None, height=dp(38), spacing=dp(8))
        select_all = FlatButton(text="Select all CSVs here")
        select_all.bind(on_release=lambda *_: self._select_all())
        cancel = FlatButton(text="Cancel")
        cancel.bind(on_release=lambda *_: self.dismiss())
        go = PrimaryButton(text="Import selected")
        go.bind(on_release=lambda *_: self._import())
        buttons.add_widget(select_all)
        buttons.add_widget(Widget())
        buttons.add_widget(cancel)
        buttons.add_widget(go)
        root.add_widget(buttons)
        self.content = root

    def _go_up(self):
        parent = str(Path(self.chooser.path).parent)
        self.chooser.path = parent

    def _select_all(self):
        folder = Path(self.chooser.path)
        self.chooser.selection = sorted(str(p) for p in folder.iterdir()
                                        if p.is_file() and p.suffix.lower() == ".csv")

    def _import(self):
        files = [f for f in self.chooser.selection if os.path.isfile(f)]
        if not files:
            return
        self.dismiss()
        self.callback(files)
