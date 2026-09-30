"""Evaluate the pretrained STEERER image counter on MDC++ frames.

This deliberately bypasses the repository's video-counting models. It loads only the
pretrained global STEERER counter, evaluates individual frames, and writes restart-safe
per-image results suitable for later error analysis.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MDC_MEAN = np.asarray([117 / 255.0, 110 / 255.0, 105 / 255.0], dtype=np.float32)
MDC_STD = np.asarray([67.10 / 255.0, 65.45 / 255.0, 66.23 / 255.0], dtype=np.float32)
CSV_FIELDS = [
    "sample_id",
    "image_relative_path",
    "annotation_relative_path",
    "scene",
    "clip",
    "frame_number",
    "annotation_frame_index",
    "ground_truth_count",
    "predicted_count",
    "signed_error",
    "absolute_error",
    "squared_error",
    "absolute_relative_error",
    "original_width",
    "original_height",
    "input_width",
    "input_height",
    "padded_width",
    "padded_height",
    "density_width",
    "density_height",
    "inference_seconds",
    "density_relative_path",
    "run_signature",
]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def natural_key(path: Path) -> tuple[int, int | str, str]:
    try:
        return (0, int(path.stem), path.name)
    except ValueError:
        return (1, path.stem, path.name)


def annotation_counts(path: Path) -> dict[int, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {path}")
    counts: Counter[int] = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if not row or not any(value.strip() for value in row):
                continue
            try:
                frame_index = int(float(row[0]))
            except (IndexError, ValueError) as error:
                raise ValueError(f"Invalid frame index at {path}:{line_number}: {row}") from error
            counts[frame_index] += 1
    return dict(counts)


def clip_paths_for_entry(data_root: Path, entry: str) -> list[Path]:
    relative = Path(*entry.replace("\\", "/").split("/"))
    frame_entry = data_root / "frames" / relative
    if not frame_entry.is_dir():
        raise FileNotFoundError(f"Split entry does not exist under frames/: {entry}")
    direct_images = [
        path for path in frame_entry.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if direct_images:
        return [relative]
    clip_dirs = [path for path in frame_entry.iterdir() if path.is_dir()]
    if not clip_dirs:
        raise RuntimeError(f"No image files or clip directories found for split entry: {entry}")
    return [relative / path.name for path in sorted(clip_dirs, key=natural_key)]


def build_manifest(
    data_root: Path,
    split_file: str,
    frame_stride: int = 1,
    max_samples: int = 0,
) -> list[dict[str, Any]]:
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1")
    split_path = data_root / split_file
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    entries = [
        line.strip() for line in split_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not entries:
        raise RuntimeError(f"Split file is empty: {split_path}")

    manifest: list[dict[str, Any]] = []
    seen_clips: set[str] = set()
    seen_samples: set[str] = set()
    for entry in entries:
        for clip_relative in clip_paths_for_entry(data_root, entry):
            clip_key = clip_relative.as_posix()
            if clip_key in seen_clips:
                raise RuntimeError(f"Clip occurs more than once in {split_file}: {clip_key}")
            seen_clips.add(clip_key)
            frame_dir = data_root / "frames" / clip_relative
            annotation_relative = Path("annotations") / clip_relative.parent / f"{clip_relative.name}.csv"
            counts = annotation_counts(data_root / annotation_relative)
            images = sorted(
                (
                    path for path in frame_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                ),
                key=natural_key,
            )
            if not images:
                raise RuntimeError(f"No images found in {frame_dir}")
            for image_path in images[::frame_stride]:
                try:
                    frame_number = int(image_path.stem)
                except ValueError as error:
                    raise ValueError(f"MDC++ frame filename is not numeric: {image_path}") from error
                annotation_frame_index = frame_number - 1
                image_relative = image_path.relative_to(data_root)
                sample_id = image_relative.relative_to("frames").as_posix()
                if sample_id in seen_samples:
                    raise RuntimeError(f"Duplicate sample in manifest: {sample_id}")
                seen_samples.add(sample_id)
                parts = clip_relative.parts
                manifest.append(
                    {
                        "sample_id": sample_id,
                        "image_path": image_path,
                        "image_relative_path": image_relative.as_posix(),
                        "annotation_relative_path": annotation_relative.as_posix(),
                        "scene": parts[0],
                        "clip": "/".join(parts[1:]) if len(parts) > 1 else "",
                        "frame_number": frame_number,
                        "annotation_frame_index": annotation_frame_index,
                        "ground_truth_count": counts.get(annotation_frame_index, 0),
                    }
                )
                if max_samples and len(manifest) >= max_samples:
                    return manifest
    return manifest


def manifest_summary(manifest: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    counts = np.asarray([sample["ground_truth_count"] for sample in manifest], dtype=np.float64)
    return {
        "created_utc": utc_now(),
        "data_root": str(args.data_root),
        "split_file": args.split_file,
        "frame_stride": args.frame_stride,
        "max_samples": args.max_samples,
        "selected_samples": len(manifest),
        "selected_clips": len({(item["scene"], item["clip"]) for item in manifest}),
        "ground_truth_total": int(counts.sum()) if len(counts) else 0,
        "ground_truth_min": int(counts.min()) if len(counts) else None,
        "ground_truth_max": int(counts.max()) if len(counts) else None,
        "ground_truth_mean": float(counts.mean()) if len(counts) else None,
        "first_sample": manifest[0]["sample_id"] if manifest else None,
        "last_sample": manifest[-1]["sample_id"] if manifest else None,
    }


def repository_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def unwrap_state_dict(checkpoint: Any) -> dict[str, Any]:
    state = checkpoint
    if isinstance(state, dict):
        for key in ("state_dict", "model", "net"):
            candidate = state.get(key)
            if isinstance(candidate, dict):
                state = candidate
                break
    if not isinstance(state, dict) or not state:
        raise TypeError("Checkpoint does not contain a non-empty state dictionary")
    clean: dict[str, Any] = {}
    for key, value in state.items():
        clean_key = str(key)
        while clean_key.startswith("module."):
            clean_key = clean_key[7:]
        clean[clean_key] = value
    return clean


def load_model(checkpoint_path: Path, device: Any) -> tuple[Any, int]:
    import torch
    from mmcv import Config

    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from model.density_estimator.STEERER.build_counter import Baseline_Counter

    config = Config.fromfile(str(repo_root / "model/density_estimator/STEERER/configs/MDC.py"))
    # The complete counter checkpoint contains the backbone. Avoid an unnecessary attempt
    # to load the stale relative pretraining path embedded in the original config.
    config.network.pretrained_backbone = ""
    model = Baseline_Counter(
        config.network,
        config.dataset.den_factor,
        config.train.route_size,
        device=device,
    )
    try:
        raw_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        raw_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = unwrap_state_dict(raw_checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return model, parameter_count


def preprocess_image(path: Path, max_long: int, max_short: int) -> tuple[Any, dict[str, int]]:
    import torch
    import torch.nn.functional as functional

    with Image.open(path) as opened:
        image = opened.convert("RGB")
        original_width, original_height = image.size
        long_side = max(original_width, original_height)
        short_side = min(original_width, original_height)
        scale = min(max_long / long_side, max_short / short_side, 1.0)
        input_width = max(1, int(original_width * scale))
        input_height = max(1, int(original_height * scale))
        if (input_width, input_height) != image.size:
            resampling = getattr(Image, "Resampling", Image).LANCZOS
            image = image.resize((input_width, input_height), resampling)
        array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - MDC_MEAN) / MDC_STD
    tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).unsqueeze(0)
    pad_height = (-input_height) % 32
    pad_width = (-input_width) % 32
    tensor = functional.pad(tensor, (0, pad_width, 0, pad_height), mode="constant", value=0.0)
    metadata = {
        "original_width": original_width,
        "original_height": original_height,
        "input_width": input_width,
        "input_height": input_height,
        "padded_width": input_width + pad_width,
        "padded_height": input_height + pad_height,
    }
    return tensor, metadata


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CSV_FIELDS:
            raise RuntimeError(
                f"Unexpected columns in resumable CSV {path}; use a new RUN_NAME. "
                f"Expected {CSV_FIELDS}, found {reader.fieldnames}."
            )
        rows = list(reader)
    identifiers = [row["sample_id"] for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError(f"Duplicate sample_id values found in {path}")
    return rows


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def calculate_summary(
    rows: list[dict[str, str]],
    expected_ids: set[str],
    run_config: dict[str, Any],
    started_at: float,
    peak_reserved_mb: float | None,
) -> dict[str, Any]:
    row_ids = {row["sample_id"] for row in rows}
    missing = sorted(expected_ids - row_ids)
    unexpected = sorted(row_ids - expected_ids)
    absolute = np.asarray([float(row["absolute_error"]) for row in rows], dtype=np.float64)
    squared = np.asarray([float(row["squared_error"]) for row in rows], dtype=np.float64)
    signed = np.asarray([float(row["signed_error"]) for row in rows], dtype=np.float64)
    ground_truth = np.asarray([float(row["ground_truth_count"]) for row in rows], dtype=np.float64)
    predictions = np.asarray([float(row["predicted_count"]) for row in rows], dtype=np.float64)
    inference_times = np.asarray([float(row["inference_seconds"]) for row in rows], dtype=np.float64)
    positive = ground_truth > 0
    nae = float(np.mean(absolute[positive] / ground_truth[positive])) if positive.any() else float("nan")
    complete = not missing and not unexpected and len(rows) == len(expected_ids)
    return {
        "status": "complete" if complete else "incomplete",
        "created_utc": utc_now(),
        "run_signature": run_config["run_signature"],
        "expected_samples": len(expected_ids),
        "completed_samples": len(rows),
        "missing_samples": len(missing),
        "unexpected_samples": len(unexpected),
        "first_missing_samples": missing[:20],
        "first_unexpected_samples": unexpected[:20],
        "mae": float(absolute.mean()) if len(rows) else None,
        "rmse": float(np.sqrt(squared.mean())) if len(rows) else None,
        "mean_signed_error": float(signed.mean()) if len(rows) else None,
        "nae_nonzero_ground_truth": finite_or_none(nae),
        "ground_truth_total": float(ground_truth.sum()) if len(rows) else 0.0,
        "predicted_total": float(predictions.sum()) if len(rows) else 0.0,
        "mean_inference_seconds": float(inference_times.mean()) if len(rows) else None,
        "total_inference_seconds": float(inference_times.sum()) if len(rows) else 0.0,
        "script_wall_seconds_this_invocation": time.time() - started_at,
        "cuda_peak_reserved_mb_this_invocation": peak_reserved_mb,
        "checkpoint_sha256": run_config["checkpoint_sha256"],
        "repo_commit": run_config["repo_commit"],
        "preprocessing": run_config["preprocessing"],
    }


def sync_csv(local_csv: Path, durable_csv: Path) -> None:
    durable_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = durable_csv.with_name(durable_csv.name + ".tmp")
    shutil.copy2(local_csv, temporary)
    os.replace(temporary, durable_csv)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    started_at = time.time()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args.data_root, args.split_file, args.frame_stride, args.max_samples)
    expected_ids = {sample["sample_id"] for sample in manifest}
    if not manifest:
        raise RuntimeError("No samples selected for evaluation")
    if not args.checkpoint or not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    split_path = args.data_root / args.split_file
    checkpoint_sha = sha256_file(args.checkpoint)
    core_config = {
        "format_version": 1,
        "repo_commit": repository_commit(),
        "checkpoint_path_at_run_time": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "split_file": args.split_file,
        "split_file_sha256": sha256_file(split_path),
        "frame_stride": args.frame_stride,
        "max_samples": args.max_samples,
        "selected_samples": len(manifest),
        "use_amp": args.use_amp,
        "save_density_maps": args.save_density_maps,
        "density_map_limit": args.density_map_limit,
        "preprocessing": {
            "max_long": args.max_long,
            "max_short": args.max_short,
            "pad_multiple": 32,
            "pad_side": "right_and_bottom",
            "mean": MDC_MEAN.tolist(),
            "std": MDC_STD.tolist(),
            "density_factor": 100,
        },
    }
    signature_payload = json.dumps(core_config, sort_keys=True, separators=(",", ":"))
    run_signature = hashlib.sha256(signature_payload.encode("utf-8")).hexdigest()
    run_config = {**core_config, "run_signature": run_signature, "created_utc": utc_now()}
    config_path = output_dir / "run_config.json"
    if config_path.is_file():
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        if existing_config.get("run_signature") != run_signature:
            raise RuntimeError(
                "This output directory contains results from a different configuration. "
                "Choose a new RUN_NAME or restore the original settings."
            )
    else:
        write_json(config_path, run_config)

    durable_csv = output_dir / "predictions.csv"
    existing_rows = load_rows(durable_csv)
    if existing_rows and not args.resume:
        raise RuntimeError(f"Results already exist at {durable_csv}; pass --resume or use a new RUN_NAME")
    for row in existing_rows:
        if row["run_signature"] != run_signature:
            raise RuntimeError(f"Existing CSV was produced by a different configuration: {durable_csv}")
    completed_ids = {row["sample_id"] for row in existing_rows}
    unexpected_existing = completed_ids - expected_ids
    if unexpected_existing:
        raise RuntimeError(f"Existing CSV contains unexpected samples: {sorted(unexpected_existing)[:10]}")
    pending = [sample for sample in manifest if sample["sample_id"] not in completed_ids]

    args.work_dir.mkdir(parents=True, exist_ok=True)
    local_csv = args.work_dir / f"predictions_{run_signature[:12]}.csv"
    if durable_csv.is_file():
        shutil.copy2(durable_csv, local_csv)
    else:
        with local_csv.open("w", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=CSV_FIELDS).writeheader()

    progress_path = output_dir / "progress.json"
    write_json(
        progress_path,
        {
            "status": "starting" if pending else "already_complete",
            "updated_utc": utc_now(),
            "expected_samples": len(manifest),
            "completed_samples": len(existing_rows),
            "pending_samples": len(pending),
            "run_signature": run_signature,
        },
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. In Colab select a GPU runtime.")
    if args.use_amp and device.type != "cuda":
        raise ValueError("--use-amp is supported only with CUDA")

    model = None
    parameter_count = None
    processed_this_invocation = 0
    peak_reserved_mb: float | None = None
    density_dir = output_dir / "density_maps"
    saved_density_count = len(list(density_dir.glob("*.npz"))) if density_dir.is_dir() else 0
    failure: BaseException | None = None
    try:
        if pending:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(args.seed)
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
                torch.cuda.reset_peak_memory_stats(device)
            print(f"Loading STEERER checkpoint: {args.checkpoint}")
            model, parameter_count = load_model(args.checkpoint, device)
            print(f"Model parameters: {parameter_count:,}")
            print(f"Samples: {len(manifest)} total, {len(existing_rows)} already complete, {len(pending)} pending")

            with local_csv.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                for pending_index, sample in enumerate(pending, start=1):
                    input_tensor, image_meta = preprocess_image(
                        sample["image_path"], args.max_long, args.max_short
                    )
                    input_tensor = input_tensor.to(device, non_blocking=device.type == "cuda")
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    inference_start = time.perf_counter()
                    with torch.inference_mode():
                        with torch.autocast(
                            device_type=device.type,
                            dtype=torch.float16,
                            enabled=args.use_amp,
                        ):
                            density = model(input_tensor)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    inference_seconds = time.perf_counter() - inference_start
                    if density.ndim != 4 or density.shape[0] != 1 or density.shape[1] != 1:
                        raise RuntimeError(
                            f"STEERER returned an unexpected density shape for {sample['sample_id']}: "
                            f"{tuple(density.shape)}"
                        )
                    predicted_count = float(density.float().sum().item())
                    if not math.isfinite(predicted_count):
                        raise RuntimeError(
                            f"STEERER returned a non-finite count for {sample['sample_id']}: "
                            f"{predicted_count}"
                        )
                    ground_truth_count = float(sample["ground_truth_count"])
                    signed_error = predicted_count - ground_truth_count
                    density_relative = ""
                    should_save_density = args.save_density_maps and (
                        args.density_map_limit == 0
                        or saved_density_count < args.density_map_limit
                    )
                    if should_save_density:
                        density_dir.mkdir(parents=True, exist_ok=True)
                        safe_stem = Path(sample["sample_id"]).with_suffix("").as_posix().replace("/", "__")
                        density_path = density_dir / f"{safe_stem}.npz"
                        np.savez_compressed(
                            density_path,
                            density=density.detach().float().cpu().numpy()[0, 0],
                        )
                        density_relative = density_path.relative_to(output_dir).as_posix()
                        saved_density_count += 1
                    row = {
                        **{key: sample[key] for key in (
                            "sample_id", "image_relative_path", "annotation_relative_path",
                            "scene", "clip", "frame_number", "annotation_frame_index",
                            "ground_truth_count",
                        )},
                        "predicted_count": f"{predicted_count:.10f}",
                        "signed_error": f"{signed_error:.10f}",
                        "absolute_error": f"{abs(signed_error):.10f}",
                        "squared_error": f"{signed_error ** 2:.10f}",
                        "absolute_relative_error": (
                            f"{abs(signed_error) / ground_truth_count:.10f}"
                            if ground_truth_count > 0 else ""
                        ),
                        **image_meta,
                        "density_width": int(density.shape[-1]),
                        "density_height": int(density.shape[-2]),
                        "inference_seconds": f"{inference_seconds:.6f}",
                        "density_relative_path": density_relative,
                        "run_signature": run_signature,
                    }
                    writer.writerow(row)
                    handle.flush()
                    processed_this_invocation += 1
                    completed_total = len(existing_rows) + processed_this_invocation
                    if processed_this_invocation % args.sync_every == 0 or pending_index == len(pending):
                        sync_csv(local_csv, durable_csv)
                        write_json(
                            progress_path,
                            {
                                "status": "running" if pending_index < len(pending) else "finishing",
                                "updated_utc": utc_now(),
                                "expected_samples": len(manifest),
                                "completed_samples": completed_total,
                                "pending_samples": len(manifest) - completed_total,
                                "last_sample": sample["sample_id"],
                                "run_signature": run_signature,
                            },
                        )
                    if pending_index == 1 or pending_index % args.print_every == 0 or pending_index == len(pending):
                        print(
                            f"[{completed_total}/{len(manifest)}] {sample['sample_id']} "
                            f"GT={ground_truth_count:.0f} pred={predicted_count:.2f} "
                            f"abs={abs(signed_error):.2f} time={inference_seconds:.2f}s"
                        )
                    del input_tensor, density
            if device.type == "cuda":
                peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    except BaseException as error:
        failure = error
        raise
    finally:
        if local_csv.is_file():
            sync_csv(local_csv, durable_csv)
        if failure is not None:
            write_json(
                progress_path,
                {
                    "status": "failed",
                    "updated_utc": utc_now(),
                    "expected_samples": len(manifest),
                    "completed_samples": len(load_rows(durable_csv)),
                    "error_type": type(failure).__name__,
                    "error": str(failure),
                    "run_signature": run_signature,
                },
            )

    final_rows = load_rows(durable_csv)
    summary = calculate_summary(final_rows, expected_ids, run_config, started_at, peak_reserved_mb)
    previous_summary_path = output_dir / "summary.json"
    previous_model_parameters = None
    if previous_summary_path.is_file():
        previous_model_parameters = json.loads(
            previous_summary_path.read_text(encoding="utf-8")
        ).get("model_parameters")
    summary["model_parameters"] = (
        parameter_count if parameter_count is not None else previous_model_parameters
    )
    summary["processed_this_invocation"] = processed_this_invocation
    write_json(output_dir / "summary.json", summary)
    write_json(
        progress_path,
        {
            "status": summary["status"],
            "updated_utc": utc_now(),
            "expected_samples": len(manifest),
            "completed_samples": len(final_rows),
            "pending_samples": len(manifest) - len(final_rows),
            "run_signature": run_signature,
        },
    )
    print(json.dumps(summary, indent=2))
    return summary


def parse_path(value: str) -> Path:
    return Path(value).expanduser()


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=parse_path, required=True)
    parser.add_argument("--split-file", default="test.txt")
    parser.add_argument("--checkpoint", type=parse_path)
    parser.add_argument("--output-dir", type=parse_path, required=True)
    parser.add_argument("--work-dir", type=parse_path, default=Path("/tmp/steerer_mdc_work"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-long", type=int, default=1920)
    parser.add_argument("--max-short", type=int, default=1080)
    parser.add_argument("--seed", type=int, default=3035)
    parser.add_argument("--sync-every", type=int, default=25)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--density-map-limit", type=int, default=20)
    parser.add_argument("--save-density-maps", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--inspect-only", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    args.data_root = args.data_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.work_dir = args.work_dir.resolve()
    if args.checkpoint:
        args.checkpoint = args.checkpoint.resolve()
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be at least 1")
    if args.max_samples < 0:
        raise ValueError("--max-samples cannot be negative")
    if args.max_long < 32 or args.max_short < 32:
        raise ValueError("--max-long and --max-short must be at least 32")
    if args.sync_every < 1 or args.print_every < 1:
        raise ValueError("--sync-every and --print-every must be at least 1")
    if args.density_map_limit < 0:
        raise ValueError("--density-map-limit cannot be negative")


def main(argv: Iterable[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.inspect_only:
        manifest = build_manifest(args.data_root, args.split_file, args.frame_stride, args.max_samples)
        summary = manifest_summary(manifest, args)
        write_json(args.output_dir / "dataset_inspection.json", summary)
        print(json.dumps(summary, indent=2))
        return 0
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
