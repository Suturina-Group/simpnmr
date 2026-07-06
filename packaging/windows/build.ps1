<#
.SYNOPSIS
    Build the SimpNMR Windows desktop application and installer.

.DESCRIPTION
    Runs the full packaging pipeline on a Windows machine:
      1. installs SimpNMR (with GUI extras) plus PyInstaller into the current
         Python environment,
      2. freezes the GUI into dist\SimpNMR\ with PyInstaller,
      3. compiles a double-click installer with Inno Setup (if iscc is found).

    Run from the repository root:

        powershell -ExecutionPolicy Bypass -File packaging\windows\build.ps1

    Requirements on the build machine:
      * Python 3.10+ (64-bit) on PATH
      * Inno Setup 6 (iscc.exe) on PATH — only needed for the installer step;
        the frozen app in dist\SimpNMR\ is produced regardless.

.NOTES
    This script is host-agnostic: it is used both by developers on a local
    Windows PC and by the CI Windows build job.
#>
[CmdletBinding()]
param(
    # Version string embedded in the installer filename and metadata.
    # Defaults to the value in simpnmr/__version__.py.
    [string]$Version
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# Resolve repository root (two levels up from this script).
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepoRoot
Write-Host "==> Repository root: $RepoRoot"

if (-not $Version) {
    $versionLine = Select-String -Path "simpnmr\__version__.py" -Pattern '__version__\s*=\s*"([^"]+)"'
    $Version = $versionLine.Matches[0].Groups[1].Value
}
Write-Host "==> Building SimpNMR $Version"

Write-Host "==> Installing SimpNMR (with GUI extras) and PyInstaller"
python -m pip install --upgrade pip
python -m pip install ".[gui]" pyinstaller

Write-Host "==> Cleaning previous build output"
if (Test-Path "build\SimpNMR") { Remove-Item -Recurse -Force "build\SimpNMR" }
if (Test-Path "dist\SimpNMR")  { Remove-Item -Recurse -Force "dist\SimpNMR" }

Write-Host "==> Running PyInstaller"
pyinstaller "packaging\windows\simpnmr.spec" --noconfirm

if (-not (Test-Path "dist\SimpNMR\SimpNMR.exe")) {
    throw "PyInstaller did not produce dist\SimpNMR\SimpNMR.exe"
}
Write-Host "==> Frozen app ready at dist\SimpNMR\SimpNMR.exe"

$iscc = Get-Command iscc.exe -ErrorAction SilentlyContinue
if ($iscc) {
    Write-Host "==> Compiling installer with Inno Setup"
    & $iscc.Source "/DAppVersion=$Version" "packaging\windows\simpnmr.iss"
    $installer = "packaging\windows\Output\SimpNMR-Setup-$Version.exe"
    if (Test-Path $installer) {
        Write-Host "==> Installer ready: $installer"
    } else {
        throw "Inno Setup ran but the installer was not found at $installer"
    }
} else {
    Write-Warning "iscc.exe not found on PATH — skipping installer step."
    Write-Warning "Install Inno Setup 6 to produce SimpNMR-Setup-$Version.exe."
}

Write-Host "==> Done."
