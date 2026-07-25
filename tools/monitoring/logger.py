import logging
import os
from contextlib import contextmanager
from datetime   import datetime
from typing     import Any, Mapping, Optional, Sequence

from rich.console import Console
from rich.logging import RichHandler
from rich.panel   import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.rule  import Rule
from rich.table import Table
from rich.text  import Text
from rich.theme import Theme


_THEME = Theme({
    "section":    "bold cyan",
    "subsection": "white",
    "key":        "bold magenta",
    "value":      "bright_white",
    "ok":         "bold green",
    "warn":       "bold yellow",
    "err":        "bold red",
    "muted":      "white",
    "logging.level.debug":    "white",
    "logging.level.info":     "white",
    "logging.level.warning":  "bold yellow",
    "logging.level.error":    "bold red",
    "logging.level.critical": "bold red",
})

_CONSOLE: Optional[Console] = None


def get_console() -> Console:

    global _CONSOLE
    if _CONSOLE is None:
        _CONSOLE = Console(
            theme=_THEME, 
            highlight=False, 
            soft_wrap=False, 
            force_terminal=True,
            color_system="truecolor",
            legacy_windows=False,
            no_color=False
        )
    return _CONSOLE


def _make_progress(console: Console, transient: bool = False) -> Progress:

    return Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=transient,
        refresh_per_second=8,
    )


class Logger:

    LOG_LEVELS = {name: getattr(logging, name) for name in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")}

    def __init__(self, log_dir: str = "logs", name: str = "experiment", level: str = "INFO", config: Any = None) -> None:
        self.log_dir    = log_dir
        self.name       = name
        self.start_time = datetime.now()
        self.config     = config

        if log_dir:
            os.makedirs(self.log_dir, exist_ok=True)

        self.console: Console = get_console()
        self.logger           = logging.getLogger(name)
        self.logger.propagate = False
        if self.logger.hasHandlers():
            for handler in list(self.logger.handlers):
                handler.close()
                self.logger.removeHandler(handler)

        log_level = self.LOG_LEVELS.get(str(level).upper(), logging.INFO)
        self.logger.setLevel(log_level)

        rich_handler = RichHandler(
            console            = self.console,
            level              = log_level,
            show_time          = True,
            show_level         = True,
            show_path          = False,
            markup             = True,
            rich_tracebacks    = True,
            log_time_format    = "[%H:%M:%S]",
        )
        rich_handler.setLevel(log_level)
        self.logger.addHandler(rich_handler)


        self._file_handler: Optional[logging.FileHandler] = None
        if log_dir:
            file_formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            file_handler   = logging.FileHandler(os.path.join(self.log_dir, f'{name}.log'), mode='w', encoding='utf-8')
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(log_level)
            self.logger.addHandler(file_handler)
            self._file_handler = file_handler

    def _to_file(self, message: str, level: int = logging.INFO) -> None:
        if self._file_handler is None:
            return
        self._file_handler.handle(self.logger.makeRecord(self.name, level, "", 0, message, None, None))

    def section(self, title: str) -> None:
        text = str(title).upper()
        self.console.print()
        self.console.print(Rule(Text(text, style="section"), style="cyan"))
        self._to_file(f">>> {text}")

    def subsection(self, title: str) -> None:
        line = f"  [cyan]>[/cyan] {title}"
        self.console.print(line, style="bold white")
        self._to_file(f"  > {title}")

    def debug(self, message: str) -> None:    
        self.logger.debug(message)
    
    def info(self, message: str) -> None:     
        self.logger.info(message)
    
    def warning(self, message: str) -> None:  
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        self.logger.error(message)

    def critical(self, message: str) -> None:
        self.logger.critical(message)

    def ok(self, message: str) -> None:
        self.console.print(f"  [ok]+[/ok] {message}")
        self._to_file(f"  + {message}")

    @staticmethod
    def _fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    def render(self, renderable: Any) -> None:
        self.console.print(renderable)

    def panel(self, body: Any, title: Optional[str] = None, style: str = "cyan") -> None:
        self.console.print(Panel(body, title=title, border_style=style))

    def rule(self, title: str = "", style: str = "cyan") -> None:
        self.console.print(Rule(title, style=style))

    def kv_table(self, data: Mapping[str, Any], title: Optional[str] = None, key_header: str = "Field", value_header: str = "Value") -> None:
        tbl = Table(title=title, show_header=True, header_style="bold cyan", expand=False)
        tbl.add_column(key_header, style="key", no_wrap=True)
        tbl.add_column(value_header, style="value")

        for k, v in data.items():
            tbl.add_row(str(k), self._fmt(v))

        self.console.print(tbl)

        if not data:
            return

        if title:
            self._to_file(f"  > {title}")

        key_width = max(len(str(k)) for k in data)
        for k, v in data.items():
            self._to_file(f"      {str(k):<{key_width}} : {self._fmt(v)}")

    def metrics_table(self, rows: Sequence[Mapping[str, Any]], columns: Sequence[str], title: Optional[str] = None, column_styles: Optional[Mapping[str, str]] = None,) -> None:
        styles = column_styles or {}
        tbl    = Table(title=title, show_header=True, header_style="bold cyan", expand=False)

        for col in columns:
            tbl.add_column(col, style=styles.get(col, "value"))

        for row in rows:
            tbl.add_row(*[self._fmt(row.get(c, "")) for c in columns])

        self.console.print(tbl)

        if title:
            self._to_file(f"  > {title}")

        cells  = [[self._fmt(row.get(column, "")) for column in columns] for row in rows]
        widths = [max(len(str(columns[index])), *(len(row[index]) for row in cells)) if cells else len(str(columns[index])) for index in range(len(columns))]

        self._to_file("      " + "  ".join(str(columns[index]).ljust(widths[index]) for index in range(len(columns))))
        for row in cells:
            self._to_file("      " + "  ".join(row[index].ljust(widths[index]) for index in range(len(columns))))

    @contextmanager
    def timer(self, label: str):
        start = datetime.now()
        yield
        elapsed = (datetime.now() - start).total_seconds()
        self.info(f"{label} completed in {elapsed:.2f}s")

    @contextmanager
    def track(self, transient: bool = False):
        progress = _make_progress(self.console, transient=transient)
        with progress:
            yield progress

    progress_bar = track

    def close(self) -> None:
        elapsed = datetime.now() - self.start_time
        hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)

        self.logger.info(f"[End] Duration: {hours:02d}:{minutes:02d}:{seconds:02d}")
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
        self._file_handler = None

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

class NullLogger:
    def __getattr__(self, name: str):
        return lambda *args, **kwargs: None
