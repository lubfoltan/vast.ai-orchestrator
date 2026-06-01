"""Convert a NORMAL/PNEUMONIA chest X-ray dataset into three class folders.

The script supports either:
  - /path/to/data/NORMAL and /path/to/data/PNEUMONIA
  - /path/to/data/train|val|test/NORMAL and PNEUMONIA

Pneumonia subtype is inferred from filename/path tokens containing bacteria or virus.
"""

from __future__ import annotations

import argparse
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
TARGET_CLASSES = ("NORMAL", "PNEUMONIA_BACTERIAL", "PNEUMONIA_VIRUS")
SPLIT_NAMES = ("train", "val", "validation", "test")
OUTPUT_SPLITS = ("train", "val", "test")


def infer_class_name(path: Path) -> str | None:
    tokens = [part.lower() for part in path.parts]
    filename = path.name.lower()

    if any(part in {"normal", "normal2"} for part in tokens) or "normal" in filename:
        return "NORMAL"
    if "bacteria" in filename or "bacterial" in filename or any("bacteria" in part or "bacterial" in part for part in tokens):
        return "PNEUMONIA_BACTERIAL"
    if "virus" in filename or "viral" in filename or any("virus" in part or "viral" in part for part in tokens):
        return "PNEUMONIA_VIRUS"
    if filename.startswith("im-") or filename.startswith("normal2-im-"):
        return "NORMAL"
    return None


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    suffix = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{suffix}{path.suffix}")
        if not candidate.exists():
            return candidate
        suffix += 1


def group_key(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"_aug_\d+$", "", stem, flags=re.I)
    match = re.search(r"(person\d+)", stem, re.I)
    if match:
        return match.group(1).lower()
    match = re.search(r"(NORMAL2-IM-\d+|IM-\d+)", stem, re.I)
    if match:
        return match.group(1).lower()
    match = re.search(r"((?:BACTERIA|VIRUS)-\d+)", stem, re.I)
    if match:
        return match.group(1).lower()
    return stem.lower()


def split_contexts(root: Path) -> list[Path]:
    contexts = [root / name for name in SPLIT_NAMES if (root / name).is_dir()]
    if contexts:
        return contexts
    return [root]


def image_files_for_context(context: Path) -> list[Path]:
    files: list[Path] = []
    for path in context.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if any(part.startswith(".") for part in path.relative_to(context).parts):
            continue
        files.append(path)
    return sorted(files)


def convert_context(context: Path, dry_run: bool) -> Counter[str]:
    counts: Counter[str] = Counter()
    skipped: list[Path] = []

    for source_path in image_files_for_context(context):
        target_class = infer_class_name(source_path.relative_to(context))
        if target_class is None:
            skipped.append(source_path)
            continue

        target_dir = context / target_class
        target_path = unique_path(target_dir / source_path.name)
        already_correct = source_path.parent == target_dir
        counts[target_class] += 1

        if dry_run or already_correct:
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_path), str(target_path))

    if not dry_run:
        for empty_name in ("PNEUMONIA", "pneumonia"):
            empty_dir = context / empty_name
            if empty_dir.is_dir() and not any(empty_dir.iterdir()):
                empty_dir.rmdir()

    print(f"## {context}")
    for class_name in TARGET_CLASSES:
        print(f"{class_name}: {counts[class_name]}")
    if skipped:
        print(f"Skipped unknown subtype/class: {len(skipped)}")
        for path in skipped[:20]:
            print(f"  {path}")
    return counts


def split_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    ratio_sum = sum(ratios.values())
    if ratio_sum <= 0:
        raise ValueError("Split ratios must sum to a positive value.")

    normalized = [ratios[name] / ratio_sum for name in OUTPUT_SPLITS]
    positive = [index for index, ratio in enumerate(normalized) if ratio > 0]
    if total < len(positive):
        raise ValueError(f"Only {total} groups/images available for {len(positive)} non-empty splits.")

    raw = [total * ratio for ratio in normalized]
    counts = [math.floor(value) for value in raw]
    remainder = total - sum(counts)
    order = sorted(range(len(raw)), key=lambda index: (raw[index] - counts[index], normalized[index]), reverse=True)
    for index in order[:remainder]:
        counts[index] += 1
    for index in positive:
        if counts[index] == 0:
            donor = max((candidate for candidate in positive if counts[candidate] > 1), key=lambda candidate: counts[candidate])
            counts[donor] -= 1
            counts[index] += 1
    return dict(zip(OUTPUT_SPLITS, counts))


def split_groups_for_class(class_name: str, files: list[Path], ratios: dict[str, float], seed: int) -> dict[str, list[Path]]:
    groups_by_key: dict[str, list[Path]] = defaultdict(list)
    for source_path in files:
        groups_by_key[group_key(source_path)].append(source_path)

    positive_splits = [name for name in OUTPUT_SPLITS if ratios[name] > 0]
    if len(groups_by_key) < len(positive_splits):
        raise ValueError(
            f"Class {class_name} has only {len(groups_by_key)} patient/source groups, "
            f"but {len(positive_splits)} non-empty splits are requested."
        )

    groups = sorted(groups_by_key.items(), key=lambda item: (len(item[1]), item[0]), reverse=True)
    target_counts = split_counts(len(files), ratios)
    assigned = {name: [] for name in OUTPUT_SPLITS}
    assigned_counts = {name: 0 for name in OUTPUT_SPLITS}

    ordered_splits = sorted(positive_splits, key=lambda name: target_counts[name], reverse=True)
    if ordered_splits:
        split_name, (_, paths) = ordered_splits[0], groups.pop(0)
        assigned[split_name].extend(paths)
        assigned_counts[split_name] += len(paths)
        for split_name in sorted(ordered_splits[1:], key=lambda name: target_counts[name]):
            _, paths = groups.pop()
            assigned[split_name].extend(paths)
            assigned_counts[split_name] += len(paths)

    rng = random.Random(f"{seed}:{class_name}:groups")
    rng.shuffle(groups)
    groups.sort(key=lambda item: len(item[1]), reverse=True)
    for _, paths in groups:
        best_split = max(
            positive_splits,
            key=lambda name: (
                (target_counts[name] - assigned_counts[name]) / max(target_counts[name], 1),
                target_counts[name] - assigned_counts[name],
            ),
        )
        assigned[best_split].extend(paths)
        assigned_counts[best_split] += len(paths)

    print(f"{class_name}: {len(files)} images in {len(groups_by_key)} groups -> {assigned_counts}")
    return assigned


def collect_files_by_class(root: Path) -> dict[str, list[Path]]:
    files_by_class: dict[str, list[Path]] = defaultdict(list)
    for context in split_contexts(root):
        for source_path in image_files_for_context(context):
            class_name = infer_class_name(source_path.relative_to(context))
            if class_name in TARGET_CLASSES:
                files_by_class[class_name].append(source_path)
    return files_by_class


def validate_no_train_overlap(root: Path) -> None:
    groups_by_class_split: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for split_name in OUTPUT_SPLITS:
        split_dir = root / split_name
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            for source_path in image_files_for_context(class_dir):
                groups_by_class_split[class_dir.name][split_name].add(group_key(source_path))

    for class_name in TARGET_CLASSES:
        train_groups = groups_by_class_split[class_name]["train"]
        for split_name in ("val", "test"):
            overlap = groups_by_class_split[class_name][split_name] & train_groups
            print(f"{class_name} {split_name} overlap with train groups: {len(overlap)}")


def resplit_dataset(root: Path, ratios: dict[str, float], seed: int, dry_run: bool) -> None:
    files_by_class = collect_files_by_class(root)
    missing = [class_name for class_name in TARGET_CLASSES if not files_by_class.get(class_name)]
    if missing:
        raise ValueError(f"Missing target classes before re-split: {missing}")

    assignments = {
        class_name: split_groups_for_class(class_name, files_by_class[class_name], ratios, seed)
        for class_name in TARGET_CLASSES
    }
    if dry_run:
        print("Dry-run only: group-safe re-split was not written.")
        return

    tmp_root = root.with_name(f"{root.name}_group_safe_tmp")
    shutil.rmtree(tmp_root, ignore_errors=True)
    for split_name in OUTPUT_SPLITS:
        (tmp_root / split_name).mkdir(parents=True, exist_ok=True)

    for class_name, class_assignments in assignments.items():
        for split_name, source_paths in class_assignments.items():
            target_dir = tmp_root / split_name / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for source_path in source_paths:
                shutil.move(str(source_path), str(unique_path(target_dir / source_path.name)))

    for entry in root.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    for split_name in OUTPUT_SPLITS:
        shutil.move(str(tmp_root / split_name), str(root / split_name))
    shutil.rmtree(tmp_root, ignore_errors=True)

    print("Group-safe re-split written.")
    validate_no_train_overlap(root)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a three-class chest X-ray dataset.")
    parser.add_argument("root", nargs="?", default="/workspace/data", help="Dataset root directory")
    parser.add_argument("--dry-run", action="store_true", help="Only print the planned class counts")
    parser.add_argument("--resplit", action="store_true", help="Pool existing data and write a group-safe train/val/test split")
    parser.add_argument("--train-split", type=float, default=0.70)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=27)
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"ERROR: Dataset root does not exist: {root}")
        return 1

    totals: Counter[str] = Counter()
    for context in split_contexts(root):
        totals.update(convert_context(context, args.dry_run))

    print("## TOTAL")
    for class_name in TARGET_CLASSES:
        print(f"{class_name}: {totals[class_name]}")
    if any(totals[class_name] == 0 for class_name in TARGET_CLASSES):
        print("ERROR: At least one target class has zero images.")
        return 1
    if args.resplit:
        ratios = {"train": args.train_split, "val": args.val_split, "test": args.test_split}
        resplit_dataset(root, ratios, args.seed, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())