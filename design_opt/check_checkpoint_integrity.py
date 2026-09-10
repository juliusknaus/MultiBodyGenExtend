#!/usr/bin/env python3
"""Check BodyGen checkpoint pickle files for truncation/load errors.

Example:
    python3 design_opt/check_checkpoint_integrity.py --root_dir final_runs/stacking
"""

import argparse
import csv
import glob
import os
import pickle
import re
import sys

import numpy as np
import yaml
from omegaconf import OmegaConf

sys.path.append(os.getcwd())

from design_opt.eval_morphology import resolve_train_run_dir
from design_opt.utils.config import Config


EPOCH_PATTERN = re.compile(r"epoch_\d+\.p$")
SUMMARY_FIELDNAMES = [
    "run_label", "train_run_dir", "model_dir", "checkpoint", "path", "size_bytes",
    "status", "error_type", "error_message",
]


def is_numpy_core_compat_error(exc):
    if not isinstance(exc, ModuleNotFoundError):
        return False
    missing_name = getattr(exc, "name", "") or ""
    message = str(exc)
    return missing_name.startswith("numpy._core") or "numpy._core" in message


def load_pickle_compat(path):
    try:
        with open(path, "rb") as stream:
            return pickle.load(stream)
    except ModuleNotFoundError as exc:
        if not is_numpy_core_compat_error(exc):
            raise

        prev_core = sys.modules.get("numpy._core")
        prev_multiarray = sys.modules.get("numpy._core.multiarray")
        sys.modules["numpy._core"] = np.core
        sys.modules["numpy._core.multiarray"] = np.core.multiarray
        try:
            with open(path, "rb") as stream:
                return pickle.load(stream)
        finally:
            if prev_core is None:
                sys.modules.pop("numpy._core", None)
            else:
                sys.modules["numpy._core"] = prev_core

            if prev_multiarray is None:
                sys.modules.pop("numpy._core.multiarray", None)
            else:
                sys.modules["numpy._core.multiarray"] = prev_multiarray


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check checkpoint .p files under training runs for truncation/load errors."
    )
    parser.add_argument(
        "--root_dir", required=True,
        help="Folder containing training run directories, e.g. final_runs/stacking."
    )
    parser.add_argument(
        "--include_best", action="store_true",
        help="Also check best.p and best_agent_*.p in each models directory."
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Discover runs recursively below --root_dir instead of only direct children."
    )
    parser.add_argument(
        "--summary_csv", default=None,
        help="Optional CSV path for the full checkpoint status table."
    )
    return parser.parse_args()


def discover_train_runs(root_dir, recursive=False):
    root_dir = os.path.abspath(os.path.expanduser(root_dir))
    if os.path.isfile(os.path.join(root_dir, ".hydra", "config.yaml")):
        return [{"label": os.path.basename(root_dir), "train_run_dir": root_dir}]

    if not recursive:
        runs = []
        for name in sorted(os.listdir(root_dir)):
            candidate = os.path.join(root_dir, name)
            if not os.path.isdir(candidate):
                continue
            try:
                train_run_dir = resolve_train_run_dir(candidate)
            except FileNotFoundError:
                continue
            runs.append({"label": name, "train_run_dir": train_run_dir})
        return runs

    runs = []
    seen = set()
    for current_dir, dirnames, filenames in os.walk(root_dir):
        if ".hydra" in dirnames and os.path.isfile(os.path.join(current_dir, ".hydra", "config.yaml")):
            train_run_dir = os.path.abspath(current_dir)
            if train_run_dir not in seen:
                label = os.path.relpath(train_run_dir, root_dir)
                runs.append({"label": label, "train_run_dir": train_run_dir})
                seen.add(train_run_dir)
            dirnames[:] = []
    return sorted(runs, key=lambda run: run["label"])


def resolve_model_dir(train_run_dir, project_path):
    config_path = os.path.join(train_run_dir, ".hydra", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as stream:
        flags = OmegaConf.create(yaml.safe_load(stream))
    cfg = Config(flags, project_path, base_dir=train_run_dir)
    return cfg.model_dir


def checkpoint_paths(model_dir, include_best=False):
    paths = sorted(
        path for path in glob.glob(os.path.join(model_dir, "*.p"))
        if EPOCH_PATTERN.search(os.path.basename(path))
    )
    if include_best:
        best_paths = sorted(glob.glob(os.path.join(model_dir, "best*.p")))
        paths = sorted(set(paths + best_paths))
    return paths


def check_checkpoint(path):
    row = {
        "checkpoint": os.path.basename(path),
        "path": path,
        "size_bytes": os.path.getsize(path),
        "status": "OK",
        "error_type": "",
        "error_message": "",
    }
    try:
        load_pickle_compat(path)
    except EOFError as exc:
        row.update({
            "status": "TRUNCATED",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        })
    except pickle.UnpicklingError as exc:
        message = str(exc)
        row.update({
            "status": "TRUNCATED" if "truncated" in message.lower() else "ERROR",
            "error_type": type(exc).__name__,
            "error_message": message,
        })
    except Exception as exc:
        row.update({
            "status": "ERROR",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        })
    return row


def write_summary_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_integrity_check(root_dir, include_best=False, recursive=False, summary_csv=None):
    project_path = os.getcwd()
    runs = discover_train_runs(root_dir, recursive=recursive)
    if not runs:
        raise FileNotFoundError(f"No training runs with .hydra/config.yaml found under {root_dir}")

    rows = []
    for run in runs:
        label = run["label"]
        train_run_dir = run["train_run_dir"]
        try:
            model_dir = resolve_model_dir(train_run_dir, project_path)
        except Exception as exc:
            print(f"[{label}] ERROR resolving model dir: {type(exc).__name__}: {exc}")
            continue

        paths = checkpoint_paths(model_dir, include_best=include_best)
        if not paths:
            print(f"[{label}] no matching checkpoint files in {model_dir}")
            continue

        run_rows = []
        for path in paths:
            row = check_checkpoint(path)
            row.update({
                "run_label": label,
                "train_run_dir": train_run_dir,
                "model_dir": model_dir,
            })
            run_rows.append(row)
            rows.append(row)

        bad_rows = [row for row in run_rows if row["status"] != "OK"]
        print(f"[{label}] checked {len(run_rows)} checkpoint(s): {len(bad_rows)} problem(s)")
        for row in bad_rows:
            print(
                f"  {row['status']}: {row['checkpoint']} "
                f"({row['size_bytes']} bytes) {row['error_type']}: {row['error_message']}"
            )

    if summary_csv is not None:
        write_summary_csv(summary_csv, rows)
        print(f"Checkpoint integrity summary: {summary_csv}")

    bad_count = sum(1 for row in rows if row["status"] != "OK")
    print(f"Total: checked {len(rows)} checkpoint(s), {bad_count} problem(s)")
    return rows


def main():
    args = parse_args()
    rows = run_integrity_check(
        root_dir=args.root_dir,
        include_best=args.include_best,
        recursive=args.recursive,
        summary_csv=args.summary_csv,
    )
    if any(row["status"] != "OK" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()