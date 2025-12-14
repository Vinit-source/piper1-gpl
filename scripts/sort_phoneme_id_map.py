#!/usr/bin/env python3
"""Sort the `phoneme_id_map` in a JSON config by ascending phoneme id.

Usage examples:
  # Overwrite file (creates a .bak backup by default)
  python scripts/sort_phoneme_id_map.py configs/en_Us_hfc_female.json

  # Write output to a separate file
  python scripts/sort_phoneme_id_map.py configs/en_Us_hfc_female.json --output configs/en_Us_hfc_female.sorted.json

  # Dry-run: print resulting JSON
  python scripts/sort_phoneme_id_map.py configs/en_Us_hfc_female.json --dry-run

This script preserves Unicode characters and writes pretty JSON with 4-space indent.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from typing import Any, Dict


def sort_phoneme_id_map(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of obj with `phoneme_id_map` entries ordered by numeric id.

    The ordering key for each phoneme entry is the minimum integer value found
    in its associated array. If a value is not a non-empty list of ints, it is
    placed at the end.
    """
    if 'phoneme_id_map' not in obj:
        raise KeyError("missing key: phoneme_id_map")

    ph = obj['phoneme_id_map']
    if not isinstance(ph, dict):
        raise TypeError("phoneme_id_map must be an object/dict")

    def entry_key(item):
        _, v = item
        if not isinstance(v, list) or len(v) == 0:
            return float('inf')
        try:
            return min(int(x) for x in v)
        except Exception:
            return float('inf')

    items = sorted(ph.items(), key=entry_key)
    # Build ordered mapping
    new_ph = {k: v for k, v in items}
    obj['phoneme_id_map'] = new_ph
    return obj


def load_json(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(path: str, data: Dict[str, Any]):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        f.write('\n')


def main(argv=None):
    p = argparse.ArgumentParser(description='Sort phoneme_id_map by id')
    p.add_argument('file', help='JSON config file (e.g. configs/en_Us_hfc_female.json)')
    p.add_argument('--output', '-o', help='Write output to this file instead of overwriting')
    p.add_argument('--no-backup', action='store_true', help='Do not create a .bak backup when overwriting')
    p.add_argument('--dry-run', action='store_true', help='Print resulting JSON to stdout instead of writing')
    args = p.parse_args(argv)

    try:
        data = load_json(args.file)
    except Exception as exc:
        print(f'Error reading JSON: {exc}', file=sys.stderr)
        return 2

    try:
        result = sort_phoneme_id_map(data)
    except Exception as exc:
        print(f'Error processing file: {exc}', file=sys.stderr)
        return 3

    if args.dry_run:
        print(json.dumps(result, ensure_ascii=False, indent=4))
        return 0

    out_path = args.output if args.output else args.file

    # When overwriting original, create backup unless explicitly disabled
    if out_path == args.file and not args.no_backup:
        shutil.copy2(args.file, args.file + '.bak')

    try:
        write_json(out_path, result)
    except Exception as exc:
        print(f'Error writing JSON: {exc}', file=sys.stderr)
        return 4

    print(f'Wrote: {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
