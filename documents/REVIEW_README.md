# Topic 2 Review Bundle

The archive uses the same structure as the local project. Run commands from its root directory.

- `task1/analyzer/`, `task1/scripts/`, `task1/tests/`: SignalScope, GUI verification, CWT benchmark and Windows packaging.
- `task1/screenshots/`: Chinese GUI views from source and portable checks.
- `task2/communication_experiment/`, `task2/scripts/`, `task2/tests/`: acoustic 2-FSK modem, noise models, experiment runner and tests.
- `task2/audio/`: clean, noisy and PSD-Wiener sample WAVs.
- `task2/experiments/`: measured CSV/JSON results, including 680 single-frame evaluation records and 78 repeated-reception records.
- `task2/figures/`: scientific figures in PNG and PDF formats.
- `shared/` and `conftest.py`: shared signal-processing modules, tests and Qt test setup.
- `documents/report/`: XeLaTeX source, Biber bibliography, report images and compiled A4 PDF.
- `documents/presentation/`: editable 14-slide editable PPTX and generator; the PPTX contains 11 native Office Math equations.
- `documents/scripts/`: report assets and review-archive construction.

Virtual environments, build caches, portable binaries, bulk WAVs and repeated-frame binary arrays are omitted from this compact archive. The portable ZIP is distributed separately under `task1/release/` in the full project. The included experiment records are existing measurements; the experiment runner regenerates them when explicitly run.

## Run and test

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest task1\tests task2\tests shared\tests -q
.\.venv\Scripts\python.exe -m task1.analyzer.app task2\audio\clean_plus_N1_N2.wav
```

The test suite contains 82 tests. To regenerate communication experiments, run `python -m task2.communication_experiment.runner` from the root. The scripts in `task1/scripts/` and `task2/scripts/` use the root `.venv`.

## Documents

From `documents/report/`, compile with XeLaTeX (shell escape enabled for minted), Biber, and two further XeLaTeX passes. The figure and code references resolve to `task2/figures/` and `shared/`.

From the root, run `python documents/presentation/build_presentation.py` to rebuild the PPTX. The generator preserves existing native Office equations; creating them without an existing deck uses Windows PowerPoint and pywin32. `--refresh-analysis` updates the selected scientific figures and repairs the native SNR equation. `--refresh-gui` updates GUI screenshots. PowerPoint provides PDF export and MathZone validation.

`documents/scripts/build_review_bundle.ps1` builds the review ZIP directly from the current files, preserving the task directory structure.
