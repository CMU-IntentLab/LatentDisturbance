"""YAML configs with `base:` inheritance and `--key value` CLI overrides."""

import argparse
import pathlib

from ruamel.yaml import YAML

from .dreamer.tools import args_type

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _merge(base, update):
	for k, v in update.items():
		if isinstance(v, dict) and isinstance(base.get(k), dict):
			_merge(base[k], v)
		else:
			base[k] = v
	return base


def load_yaml(path):
	path = pathlib.Path(path)
	if not path.is_absolute() and not path.exists():
		path = ROOT / path
	data = YAML(typ="safe", pure=True).load(path.read_text()) or {}
	cfg = {}
	for base in data.pop("base", []) or []:
		_merge(cfg, load_yaml(path.parent / base))
	return _merge(cfg, data)


def parse(argv=None, defaults=None):
	"""`--config a.yaml [b.yaml ...]` followed by any `--key value` override."""
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", nargs="+", required=True)
	args, rest = parser.parse_known_args(argv)
	cfg = dict(defaults or {})
	for path in args.config:
		_merge(cfg, load_yaml(path))
	parser = argparse.ArgumentParser()
	for k, v in sorted(cfg.items()):
		parser.add_argument(f"--{k}", type=args_type(v), default=args_type(v)(v))
	ns = parser.parse_args(rest)
	ns.config = args.config
	return ns


def resolve(path):
	"""Paths in configs are relative to the repository root."""
	if path is None:
		return None
	path = pathlib.Path(str(path)).expanduser()
	return path if path.is_absolute() else ROOT / path
