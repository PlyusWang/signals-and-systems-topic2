"""Package the reusable project and complete Windows portable application."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = "SignalScope-Topic2"
DEFAULT_OUTPUT = ROOT / "release/SignalScope-Topic2-reusable-win-x64.zip"
SOURCE_FILES = (".gitignore", "README.md", "requirements.txt", "conftest.py")
SOURCE_TREES = (
    "shared", "task1/analyzer", "task1/scripts", "task1/tests",
    "task1/samples", "task1/screenshots", "task2", "documents",
)
SOURCE_EXTENSIONS = {
    ".py", ".ps1", ".md", ".txt", ".spec", ".wav", ".npy", ".npz",
    ".csv", ".json", ".png", ".pdf", ".tex", ".bib", ".pptx",
}
SCREENSHOTS = {
    "gui_zh_fft.png", "gui_zh_stft.png", "gui_zh_cwt.png", "gui_zh_corr.png",
}
LAUNCHER = (
    '@echo off\r\n'
    'setlocal\r\n'
    'set "app_dir=%~dp0task1\\release\\portable\\SignalScope"\r\n'
    'if not exist "%app_dir%\\SignalScope.exe" (\r\n'
    '  echo Extract the complete ZIP before starting SignalScope.\r\n'
    '  pause\r\n'
    '  exit /b 1\r\n'
    ')\r\n'
    'pushd "%app_dir%"\r\n'
    'start "" "%app_dir%\\SignalScope.exe" "%app_dir%\\samples\\clean_plus_N1_N2.wav"\r\n'
    'popd\r\n'
).encode("ascii")


def collect_files() -> dict[str, Path]:
    files = {}

    def add(path: Path) -> None:
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        files[path.relative_to(ROOT).as_posix()] = path

    for relative in (*SOURCE_FILES, "task1/__init__.py", "task1/SignalScope.spec",
                     "task1/requirements-pack.txt"):
        add(ROOT / relative)
    for relative in SOURCE_TREES:
        for path in (ROOT / relative).rglob("*"):
            if not path.is_file() or path.suffix.lower() not in SOURCE_EXTENSIONS:
                continue
            if any(part.startswith((".", "_minted")) or part == "__pycache__"
                   for part in path.relative_to(ROOT).parts[:-1]):
                continue
            if path.name.startswith("~$") or path.name.endswith("~"):
                continue
            if path.name == "long_cwt_smoke.wav":
                continue
            if relative == "task1/screenshots" and path.name not in SCREENSHOTS:
                continue
            add(path)

    # A directory-mode PyInstaller program needs its entire runtime, including
    # bundled library ZIPs and package resources. Do not filter it by extension.
    portable = ROOT / "task1/release/portable/SignalScope"
    for required in (portable / "SignalScope.exe", portable / "_internal/python312.dll",
                     portable / "samples/clean_plus_N1_N2.wav"):
        if not required.is_file():
            raise FileNotFoundError(required)
    for path in portable.rglob("*"):
        if path.is_file():
            add(path)
    return dict(sorted(files.items()))


def build(output: Path) -> None:
    output = output.resolve()
    files = collect_files()
    if output in files.values():
        raise ValueError("The output archive must not overwrite a project asset.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".building.zip")
    digests = {}
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED,
                             compresslevel=6) as archive:
            for relative, path in files.items():
                digest = hashlib.sha256()
                with path.open("rb") as source, archive.open(
                    f"{PACKAGE_ROOT}/{relative}", "w", force_zip64=True,
                ) as destination:
                    while block := source.read(1024 * 1024):
                        destination.write(block)
                        digest.update(block)
                digests[relative] = digest.hexdigest()
            archive.writestr(f"{PACKAGE_ROOT}/Start-SignalScope.cmd", LAUNCHER)
            digests["Start-SignalScope.cmd"] = hashlib.sha256(LAUNCHER).hexdigest()
            manifest = "".join(f"{digest}  {name}\n"
                               for name, digest in sorted(digests.items()))
            archive.writestr(f"{PACKAGE_ROOT}/MANIFEST.sha256", manifest)

        # Reading every archived member checks both ZIP CRC and its original
        # SHA-256, including the runtime resources needed by the executable.
        with zipfile.ZipFile(temporary) as archive:
            for relative, expected in digests.items():
                digest = hashlib.sha256()
                with archive.open(f"{PACKAGE_ROOT}/{relative}") as source:
                    while block := source.read(1024 * 1024):
                        digest.update(block)
                if digest.hexdigest() != expected:
                    raise RuntimeError(f"Archive verification failed: {relative}")
        temporary.replace(output)
        digest = hashlib.sha256()
        with output.open("rb") as source:
            while block := source.read(1024 * 1024):
                digest.update(block)
        checksum = output.with_name(output.stem + ".sha256")
        checksum.write_text(f"{digest.hexdigest()}  {output.name}\n", encoding="utf-8")
        print(f"Archive: {output}")
        print(f"Files: {len(digests) + 1}")
        print(f"Size: {output.stat().st_size / 2**20:.2f} MiB")
        print(f"Verified SHA-256 for {len(digests)} files")
        print(f"Checksum: {checksum}")
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    build(parser.parse_args().output)
