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

SUPPORTED_TYPES = (bool, int, float, str, Path, list, tuple, dict)


class Detacher:

    ENV_FLAG = "DUNE_GNN_DETACHED"
    FLAGS    = ("--detach", "--nohup")

    def __init__(self, log_dir: str = "logs") -> None:
        self.log_dir = Path(log_dir)

    def requested(self, argv: list[str] | None = None) -> bool:
        argv = sys.argv[1:] if argv is None else argv
        return any(flag in argv for flag in self.FLAGS)

    def active(self) -> bool:
        return os.environ.get(self.ENV_FLAG) == "1"

    def ensure(self) -> None:
        if self.active():
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
            return

        if not self.requested():
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)

        stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
        name     = Path(sys.argv[0]).stem
        log_path = self.log_dir / f"{name}_{stamp}.out"

        env                     = dict(os.environ)
        env[self.ENV_FLAG]      = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("FORCE_COLOR", "1")
        env.setdefault("COLUMNS", "120")

        with open(log_path, "ab") as sink:
            process = subprocess.Popen(
                [sys.executable, "-u", *sys.argv],
                cwd               = os.getcwd(),
                stdout            = sink,
                stderr            = subprocess.STDOUT,
                stdin             = subprocess.DEVNULL,
                env               = env,
                start_new_session = True,
            )

        print(f"detached {name} as pid {process.pid}, immune to hangup")
        print(f"output: {log_path}")
        print(f"follow: tail -f {log_path}")

        raise SystemExit(0)


class ConfigCli:

    BOOTSTRAP_FLAGS = (
        "-h", "--help",
        "--help-config",
        "--detach", "--nohup",
        "--gpu",
    )

    def __init__(self, config, description: str | None = None) -> None:
        self.config    = config
        self.overrides : dict = {}
        self.parser    = argparse.ArgumentParser(description=description, add_help=False, allow_abbrev=False)

        self.parser.add_argument("--help-config", action="store_true", dest="_help_config")
        self.parser.add_argument("--detach", "--nohup", action="store_true", dest="_detach")

        for path, value in self._leaves(config):
            if value is not None and not isinstance(value, SUPPORTED_TYPES):
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
            self._print_config_help()
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
            if name in self.BOOTSTRAP_FLAGS:
                continue

            offenders.append(name)

        if offenders:
            keys = ", ".join(sorted(set(offenders)))
            raise ValueError(f"Unrecognized override option(s): {keys}. Known overrides: --<path> from {type(self.config).__name__}; bootstrap flags: {', '.join(self.BOOTSTRAP_FLAGS)}")

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
            if lowered in ("true", "1", "yes", "on"):
                return True
            if lowered in ("false", "0", "no", "off"):
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

    def _print_config_help(self) -> None:
        rows  = [(path, type(value).__name__ if value is not None else "any", repr(value)) for path, value in self._leaves(self.config)]
        width = max(len(path) for path, _, _ in rows)

        print(f"Configuration overrides for {type(self.config).__name__} (pass as --<path> <value>):")
        for path, type_name, default in rows:
            print(f"  --{path:<{width}}  {type_name:<6}  default: {default}")
        print("Execution flags:")
        print("  --detach (alias --nohup)  relaunch detached from the terminal, output to logs/<script>_<stamp>.out")

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
            elif value is None or isinstance(value, SUPPORTED_TYPES):
                mapping[path] = value
        return mapping

    @classmethod
    def save_resolved(cls, config, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cls.to_mapping(config), f, indent=2)
        return path

    @classmethod
    def load_resolved(cls, config, path: Path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Resolved config not found at {path}. The run directory is missing metadata/resolved_config.json and cannot be evaluated.")

        with open(path, "r", encoding="utf-8") as f:
            mapping = json.load(f)

        known   = {leaf for leaf, _ in cls._leaves(config)}
        unknown = sorted(key for key in mapping if key not in known)
        if unknown:
            raise KeyError(f"Unknown key(s) in resolved config {path}: {', '.join(unknown)}. Known keys belong to {type(config).__name__}")

        for leaf, current in list(cls._leaves(config)):
            if leaf not in mapping:
                continue

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
