#!/usr/bin/env python3
"""
Re-encode text files to UTF-8 from CP949 (or keep if already UTF-8).

Safety:
- Creates a backup copy under backup_encoding/<relative_path> before writing.
- Only rewrites when decoding with UTF-8 fails but CP949 succeeds.

Usage:
  py -X utf8 scripts/reencode_utf8.py path1 path2 ...
  (If no args, it does nothing.)
"""
import sys
from pathlib import Path


def decode_try(data: bytes):
    try:
        return data.decode('utf-8'), 'utf-8'
    except UnicodeDecodeError:
        pass
    try:
        return data.decode('cp949'), 'cp949'
    except UnicodeDecodeError:
        return None, None


def ensure_backup(root: Path, file_path: Path):
    backup_root = root / 'backup_encoding'
    backup_path = backup_root / file_path
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    if not backup_path.exists():
        try:
            backup_path.write_bytes(file_path.read_bytes())
        except Exception:
            pass


def main(argv):
    if len(argv) < 1:
        print('No files specified.')
        return 0

    cwd = Path.cwd()
    changed = 0
    for p in argv:
        file_path = Path(p)
        if not file_path.exists() or not file_path.is_file():
            print(f'SKIP (not file): {p}')
            continue
        try:
            data = file_path.read_bytes()
        except Exception as e:
            print(f'ERROR read {p}: {e}')
            continue

        text, enc = decode_try(data)
        if text is None:
            print(f'SKIP (unknown encoding): {p}')
            continue

        if enc == 'utf-8':
            print(f'OK (utf-8): {p}')
            continue

        # Backup then write UTF-8
        try:
            ensure_backup(cwd, file_path)
            file_path.write_text(text, encoding='utf-8', newline='\n')
            changed += 1
            print(f'FIXED ({enc} -> utf-8): {p}')
        except Exception as e:
            print(f'ERROR write {p}: {e}')

    print(f'Done. Changed files: {changed}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))

