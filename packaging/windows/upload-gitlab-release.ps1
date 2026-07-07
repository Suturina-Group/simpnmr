<#
.SYNOPSIS
    Upload the SimpNMR Windows installer to a GitLab release.

.DESCRIPTION
    Attaches the built installer to the GitLab release for a tag so it can be
    downloaded from the project's Releases page (where the documentation links
    to). Two API calls:

      1. upload the file to the project's Generic Packages registry, which
         gives it a stable, public download URL;
      2. add an asset link on the release pointing at that URL.

    The GitLab release for the tag is expected to already exist - on this
    project it is created by the semantic-release CI job when the tag is pushed.

.PARAMETER Version
    Package version, e.g. "2.0.0". Defaults to simpnmr/__version__.py.

.PARAMETER InstallerPath
    Path to the installer .exe. Defaults to the newest
    packaging\windows\Output\SimpNMR-Setup-*.exe.

.PARAMETER Token
    GitLab token with the `api` scope (a Personal or Project Access Token, or a
    CI job token). Defaults to the $env:GITLAB_TOKEN environment variable.

.PARAMETER Tag
    Release tag. Defaults to "v<Version>".

.PARAMETER ProjectPath
    URL path of the project. Defaults to "suturina-group/simpnmr".

.PARAMETER GitlabHost
    Base URL. Defaults to "https://gitlab.com".

.PARAMETER TokenHeader
    Auth header to use: "PRIVATE-TOKEN" (default, for PATs/PrATs) or
    "JOB-TOKEN" (when running inside GitLab CI with $CI_JOB_TOKEN).

.EXAMPLE
    $env:GITLAB_TOKEN = "glpat-..."
    powershell -File packaging\windows\upload-gitlab-release.ps1 -Version 2.0.0
#>
[CmdletBinding()]
param(
    [string]$Version,
    [string]$InstallerPath,
    [string]$Token = $env:GITLAB_TOKEN,
    [string]$Tag,
    [string]$ProjectPath = "suturina-group/simpnmr",
    [string]$GitlabHost = "https://gitlab.com",
    [ValidateSet("PRIVATE-TOKEN", "JOB-TOKEN")]
    [string]$TokenHeader = "PRIVATE-TOKEN"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepoRoot

if (-not $Token) {
    throw "No GitLab token provided. Set -Token or the GITLAB_TOKEN environment variable."
}

if (-not $Version) {
    $versionLine = Select-String -Path "simpnmr\__version__.py" -Pattern '__version__\s*=\s*"([^"]+)"'
    $Version = $versionLine.Matches[0].Groups[1].Value
}
if (-not $Tag) { $Tag = "v$Version" }

if (-not $InstallerPath) {
    $exe = Get-ChildItem "packaging\windows\Output\SimpNMR-Setup-*.exe" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $exe) { throw "No installer found under packaging\windows\Output\." }
    $InstallerPath = $exe.FullName
}
if (-not (Test-Path $InstallerPath)) { throw "Installer not found: $InstallerPath" }

$fileName    = Split-Path $InstallerPath -Leaf
$projectEnc  = [uri]::EscapeDataString($ProjectPath)
$tagEnc      = [uri]::EscapeDataString($Tag)
$packageName = "simpnmr-windows"
$headers     = @{ $TokenHeader = $Token }

$packageUrl = "$GitlabHost/api/v4/projects/$projectEnc/packages/generic/$packageName/$Version/$fileName"

Write-Host "==> Uploading $fileName to the generic package registry"
Write-Host "    $packageUrl"
Invoke-RestMethod -Method Put -Uri $packageUrl -Headers $headers -InFile $InstallerPath | Out-Null

Write-Host "==> Linking the package to release $Tag"
$linkUri  = "$GitlabHost/api/v4/projects/$projectEnc/releases/$tagEnc/assets/links"
$linkBody = @{
    name      = "Windows installer ($fileName)"
    url       = $packageUrl
    link_type = "package"
}
try {
    Invoke-RestMethod -Method Post -Uri $linkUri -Headers $headers -Body $linkBody | Out-Null
    Write-Host "==> Done. Installer attached to release $Tag."
} catch {
    # A re-run of the same version would otherwise fail on the duplicate link.
    if ($_.Exception.Response -and $_.Exception.Response.StatusCode.value__ -eq 400) {
        Write-Warning "Asset link may already exist for $Tag; the package upload succeeded."
    } else {
        throw
    }
}
