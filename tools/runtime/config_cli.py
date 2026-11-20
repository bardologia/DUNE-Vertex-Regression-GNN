from __future__ import annotations

import argparse
import ast
import json
import os
import signal
import subprocess
import sys
from dataclasses import fields, is_dataclass
from datetime    import datetime
from pathlib     import Path

from configuration.shared    import IoNames, LogSettings
from configuration.tools     import CliSettings
from tools.monitoring.logger import Logger


class Detacher:

    def __init__(self, log_dir: str = IoNames.logs_dir) -> None:
        self.log_dir = Path(log_dir)

    def requested(self, argv: list[str] | None = None) -> bool:
        argv = sys.argv[1:] if argv is None else argv
        return any(flag in argv for flag in CliSettings.detach_flags)

    def active(self) -> bool:
        return os.environ.get(CliSettings.detach_env_flag) == "1"

    def ensure(self) -> None:
        if self.active():
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
            return

        if not self.requested():
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)

        stamp    = datetime.now().strftime(IoNames.run_tag_format)
        name     = Path(sys.argv[0]).stem
        log_path = self.log_dir / f"{IoNames.detached_log.format(script=name, stamp=stamp)}{IoNames.output_log_suffix}"

        env                              = dict(os.environ)
        env[CliSettings.detach_env_flag] = "1"
        env["PYTHONUNBUFFERED"]          = "1"
        env.setdefault("FORCE_COLOR", "1")
        env.setdefault("COLUMNS", str(LogSettings.terminal_columns))

        with open(log_path, "ab") as sink:
            process = subprocess.Popen(
                [sys.executable, CliSettings.unbuffered_flag, *sys.argv],
                cwd               = os.getcwd(),
                stdout            = sink,
                stderr            = subprocess.STDOUT,
                stdin             = subprocess.DEVNULL,
                env               = env,
                start_new_session = True,
            )

        logger = Logger(log_dir="", name=name)
        logger.info(f"detached {name} as pid {process.pid}, immune to hangup")
        logger.info(f"output: {log_path}")
        logger.info(f"follow: tail -f {log_path}")

        raise SystemExit(0)


class ConfigCli:

    def __init__(self, config, description: str | None = None) -> None:
        self.config    = config
        self.overrides : dict = {}
        self.parser    = argparse.ArgumentParser(description=description, add_help=False, allow_abbrev=False)

        self.parser.add_argument(CliSettings.help_config_flag, action="store_true", dest="_help_config")
        self.parser.add_argument(*CliSettings.detach_flags, action="store_true", dest="_detach")

        for path, value in self._leaves(config):
            if value is not None and not isinstance(value, CliSettings.supported_types):
                continue

            options = [f"--{path}"]
            dashed  = f"--{path.replace('_', '-')}"
            if dashed not in options:
                options.append(dashed)

            self.parser.add_argument(*options, dest=path, type=str, default=None)

    def apply(self, argv: list[str] | None = None):
        args, leftover = self.parser.parse_known_args(argv)

        self._reject_unknown_options(leftover)

        if getattr(args, "_help_config", False):
            self._log_config_help()
            raise SystemExit(0)

        if getattr(args, "_detach", False):
            Detacher().ensure()

        for path, current in list(self._leaves(self.config)):
            raw = getattr(args, path, None)
            if raw is None:
                continue

            value = self._coerce(raw, current)
            self.set_path(self.config, path, value)
            self.overrides[path] = value

        return self.config

    def _reject_unknown_options(self, leftover: list[str]) -> None:
        offenders = []

        for token in leftover:
            if not token.startswith("--") and not token.startswith("-"):
                continue

            name = token.split("=", 1)[0]
            if name in CliSettings.bootstrap_flags:
                continue

            offenders.append(name)

        if offenders:
            keys = ", ".join(sorted(set(offenders)))
            raise ValueError(f"Unrecognized override option(s): {keys}. Known overrides: --<path> from {type(self.config).__name__}; bootstrap flags: {', '.join(CliSettings.bootstrap_flags)}")

    @classmethod
    def _leaves(cls, config, prefix: str = ""):
        for f in fields(config):
            value = getattr(config, f.name)
            path  = f"{prefix}{f.name}"

            if is_dataclass(value):
                yield from cls._leaves(value, prefix=f"{path}.")
            else:
                yield path, value

    def _coerce(self, raw: str, current):
        if isinstance(current, bool):
            lowered = raw.strip().lower()
            if lowered in CliSettings.true_words:
                return True
            if lowered in CliSettings.false_words:
                return False
            raise ValueError(f"Cannot parse boolean from '{raw}'")

        if isinstance(current, int):
            return int(raw)
        if isinstance(current, float):
            return float(raw)
        if isinstance(current, Path):
            return Path(raw)
        if isinstance(current, list):
            try:
                parsed = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                return [token.strip() for token in raw.split(",") if token.strip()]
            return list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]

        if isinstance(current, dict):
            return ast.literal_eval(raw)

        if isinstance(current, tuple):
            parsed = ast.literal_eval(raw)
            return tuple(parsed) if isinstance(parsed, (list, tuple)) else (parsed,)

        if current is None:
            try:
                return ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                return raw

        return raw

    def _log_config_help(self) -> None:
        rows   = [{"Option": f"--{path}", "Type": type(value).__name__ if value is not None else "any", "Default": repr(value)} for path, value in self._leaves(self.config)]
        logger = Logger(log_dir="", name=Path(sys.argv[0]).stem)

        logger.metrics_table(rows, list(rows[0]), title=f"Configuration overrides for {type(self.config).__name__} (pass as --<path> <value>)")
        logger.kv_table({"--detach (alias --nohup)": "relaunch detached from the terminal, output to logs/<script>_<stamp>.out"}, title="Execution flags", key_header="Flag", value_header="Effect")

    @staticmethod
    def set_path(config, path: str, value) -> None:
        parts  = path.split(".")
        target = config
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], value)

    @classmethod
    def apply_overrides(cls, config, overrides: dict):
        for path, value in overrides.items():
            cls.set_path(config, path, value)
        return config

    @classmethod
    def to_mapping(cls, config) -> dict:
        mapping = {}
        for path, value in cls._leaves(config):
            if isinstance(value, Path):
                mapping[path] = str(value)
            elif isinstance(value, tuple):
                mapping[path] = list(value)
            elif value is None or isinstance(value, CliSettings.supported_types):
                mapping[path] = value
        return mapping

    @classmethod
    def save_resolved(cls, config, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cls.to_mapping(config), f, indent=IoNames.json_indent)
        return path

    @classmethod
    def load_resolved(cls, config, path: Path):
        with open(path, "r", encoding="utf-8") as f:
            mapping = json.load(f)

        known   = {leaf for leaf, _ in cls._leaves(config)}
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise KeyError(f"Unknown key(s) in resolved config {path}: {', '.join(unknown)}. Known keys belong to {type(config).__name__}")

        missing = sorted(known - set(mapping))
        if missing:
            raise KeyError(f"Resolved config {path} is missing key(s): {', '.join(missing)}. It predates the current {type(config).__name__} schema; re-run the training that produced it.")

        for leaf, current in list(cls._leaves(config)):
            value = mapping[leaf]
            if isinstance(current, Path) and isinstance(value, str):
                value = Path(value)
            elif isinstance(current, tuple) and isinstance(value, list):
                value = tuple(value)

            cls.set_path(config, leaf, value)

        return config

    @staticmethod
    def to_argv(overrides: dict) -> list[str]:
        argv = []
        for path, value in overrides.items():
            if isinstance(value, bool):
                rendered = "true" if value else "false"
            elif isinstance(value, tuple):
                rendered = str(list(value))
            else:
                rendered = str(value)
            argv += [f"--{path}", rendered]
        return argv
