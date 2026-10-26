<#
.SYNOPSIS
    Thin launcher: runs find_best_model.py in the micromamba environment and
    passes every argument through unchanged. Options and defaults come only
    from find_best_model.py (argparse).

.EXAMPLE
    .\run_fbm.ps1 --help
    .\run_fbm.ps1                                   # find_best_model.py asks for each setting
    .\run_fbm.ps1 --gauge 02317620 --n 500 --start-year 2021 --end-year 2025 --n-procs 8 --skip-snow
    .\run_fbm.ps1 --gauge 02317620 --models "7,11,13" --precip-source nasa

.NOTES
    Quote lists such as --models "7,11,13"; unquoted, PowerShell turns 7,11,13 into an array.
    Set $env:FBM_ENV to use a different environment path.
#>

$ErrorActionPreference = "Stop"
$envPath = if ($env:FBM_ENV) { $env:FBM_ENV } else { "D:\micromamba\envs\findbestmodel" }

if (-not (Get-Command micromamba -ErrorAction SilentlyContinue)) {
    throw "micromamba was not found on PATH."
}
if (-not (Test-Path $envPath)) {
    throw "Environment not found: $envPath. Create it with: micromamba create -p $envPath -f environment.yml"
}

Push-Location $PSScriptRoot
try {
    # Python's stderr (argparse errors, warnings) must not become a PowerShell error
    $ErrorActionPreference = "Continue"
    # -u: unbuffered output, so progress and prompts show immediately
    & micromamba run -p $envPath python -u find_best_model.py @args
    $code = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $code
