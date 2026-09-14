"""
Create a tar.gz archive of a folder, keeping all files except that
.wav/.flac files are only included if their filename contains a given
pattern (default: "p225").

Usage:
    python make_archive_p225.py \
        --src /home/jupyter-tailleur-m/speechConcealerExp \
        --out speechConcealerExp_p225.tar.gz \
        --pattern p225
"""

import argparse
import os
import tarfile

AUDIO_EXTENSIONS = (".wav", ".flac")


def should_include(filename: str, pattern: str) -> bool:
    if filename.lower().endswith(AUDIO_EXTENSIONS):
        return pattern.lower() in filename.lower()
    return True


def make_archive(src: str, out: str, pattern: str) -> None:
    src = os.path.abspath(src)
    parent_dir = os.path.dirname(src.rstrip(os.sep))

    n_included = 0
    n_skipped = 0

    with tarfile.open(out, "w:gz") as tar:
        for root, _, files in os.walk(src):
            for filename in files:
                if not should_include(filename, pattern):
                    n_skipped += 1
                    continue
                full_path = os.path.join(root, filename)
                arcname = os.path.relpath(full_path, parent_dir)
                tar.add(full_path, arcname=arcname)
                n_included += 1

    print(f"Archive written to {out}")
    print(f"Included: {n_included} files, skipped (audio without '{pattern}'): {n_skipped} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        default="/home/jupyter-tailleur-m/speechConcealerExp",
        help="Source folder to archive",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output tar.gz file path (default: <parent of --src>/<basename of --src>_p225.tar.gz)",
    )
    parser.add_argument(
        "--pattern",
        default="p225",
        help="Substring that .wav/.flac filenames must contain to be included",
    )
    args = parser.parse_args()

    out = args.out
    if out is None:
        src_abs = os.path.abspath(args.src)
        base_dir = os.path.basename(src_abs.rstrip(os.sep))
        parent_dir = os.path.dirname(src_abs.rstrip(os.sep))
        out = os.path.join(parent_dir, f"{base_dir}_p225.tar.gz")

    make_archive(args.src, out, args.pattern)
