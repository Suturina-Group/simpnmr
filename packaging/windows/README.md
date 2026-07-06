# Windows desktop app packaging

This directory packages the SimpNMR GUI ([`simpnmr/gui/app.py`](../../simpnmr/gui/app.py))
into a **double-click Windows application** so users never need to touch the
command line.

| File | Purpose |
| --- | --- |
| `simpnmr.spec` | PyInstaller spec — freezes the GUI into `dist/SimpNMR/SimpNMR.exe`. |
| `simpnmr.iss` | Inno Setup script — wraps the frozen app in an installer with a Start-menu shortcut and uninstaller. |
| `build.ps1` | One-command build: install deps → PyInstaller → Inno Setup. |
| `simpnmr.ico` | Application icon (copied from the docs favicon). |

## Important: this must be built on Windows

PyInstaller is **not** a cross-compiler. A Windows `.exe` can only be produced
on Windows — you cannot build it from macOS or Linux. Use one of:

1. **GitHub Actions (recommended, free).** The workflow at
   [`.github/workflows/windows-build.yml`](../../.github/workflows/windows-build.yml)
   builds the installer on GitHub's hosted `windows-latest` runner on every
   `v*` tag (and on manual dispatch). On a tag build it then **uploads the
   installer back to the GitLab release** (via `upload-gitlab-release.ps1`), so
   users download it from the GitLab Releases page that the docs link to.
   Because this project is hosted on GitLab, mirror it to GitHub (Settings →
   Repository → Mirroring repositories, or a one-off push) so the workflow can
   run. GitLab.com's free tier does **not** offer hosted Windows runners.

   **Required GitHub secret:** add a repository secret named `GITLAB_TOKEN`
   (Settings → Secrets and variables → Actions) containing a GitLab Project or
   Personal Access Token with the `api` scope, belonging to a user who can
   create releases on the project. This lets the workflow attach the installer
   to the GitLab release. The GitLab release itself is created by the
   semantic-release CI job when the tag is pushed; the tag mirrors to GitHub,
   triggering this build, which then adds the installer as a release asset.

2. **A local Windows PC.** See below.

3. **A self-hosted GitLab Windows runner.** If your group registers a Windows
   runner, add a job to `.gitlab-ci.yml` that calls `build.ps1` and tag it for
   that runner. It is intentionally left out of the active pipeline so that
   pipelines don't hang waiting for a Windows runner that isn't there. On such
   a runner the upload can use the built-in CI job token — no PAT needed:

   ```powershell
   packaging\windows\upload-gitlab-release.ps1 -Token $env:CI_JOB_TOKEN -TokenHeader JOB-TOKEN
   ```

## Uploading a build to the GitLab release manually

If you build on a local Windows PC (or want to re-upload), attach the installer
to an existing GitLab release with:

```powershell
$env:GITLAB_TOKEN = "glpat-..."   # a token with the api scope
packaging\windows\upload-gitlab-release.ps1 -Version 2.0.0
```

It uploads the `.exe` to the project's Generic Packages registry and adds it as
an asset link on the `v2.0.0` release. You can also just drag-and-drop the
`.exe` onto the release in the GitLab **Releases** UI.

## Building on a local Windows PC

Prerequisites:

- **Python 3.10+ (64-bit)** on `PATH` — <https://www.python.org/downloads/>
- **Inno Setup 6** (`iscc.exe`) on `PATH` — <https://jrsoftware.org/isdl.php>
  (only needed for the installer; the frozen app is built either way)

From the repository root, in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File packaging\windows\build.ps1
```

Outputs:

- `dist\SimpNMR\SimpNMR.exe` — the frozen application (a folder you can zip and
  share as a portable build).
- `packaging\windows\Output\SimpNMR-Setup-<version>.exe` — the installer.

## How the frozen app runs pipelines

The GUI normally launches workflows by spawning a separate Python process
(`python -c "from simpnmr.cli.main import interface; interface()"`). A frozen
app has no separate `python.exe`, so `simpnmr/gui/app.py`:

- re-invokes **its own executable** with a `--run-pipeline` flag to act as the
  CLI worker (`_run_frozen_worker_if_requested`), and
- calls `multiprocessing.freeze_support()` (and the `multiprocess` fork used by
  `pathos`) at startup so the fitting pipeline's worker processes don't
  recursively relaunch the GUI.

Both behaviours are guarded by `getattr(sys, "frozen", False)`, so the normal
`pip`-installed workflow is unchanged.

## Notes / things to verify on the first Windows build

- **WebEngine size.** The bundle includes Qt WebEngine (Chromium) for the
  3Dmol.js viewer, so expect ~300–500 MB. This is why a one-folder (not
  one-file) build is used.
- **Icon.** `simpnmr.ico` is currently a single 32×32 image. For crisp
  high-DPI shortcuts, regenerate it as a multi-resolution `.ico`
  (16/32/48/256).
- **Smoke test after building:** launch `SimpNMR.exe`, load an example
  `run.yml`, confirm the 3D molecule viewer renders, and run a `fit_susc`
  workflow end-to-end (this exercises the `--run-pipeline` worker path and the
  `pathos` multiprocessing under freezing).
