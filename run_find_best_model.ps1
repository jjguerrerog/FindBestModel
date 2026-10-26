<#
.SYNOPSIS
    Runs find_best_model.py inside the micromamba environment.

.DESCRIPTION
    Every setting of find_best_model.py is exposed as a parameter. Edit the
    defaults below, or override them on the command line.

.EXAMPLE
    .\run_find_best_model.ps1
    .\run_find_best_model.ps1 -Gauge 02317620 -N 500 -StartYear 2021 -EndYear 2025 -NProcs 8
    .\run_find_best_model.ps1 -Gauge 02317620 -Models "7,11,13,28,34" -PrecipSource nasa -NasaSampling basin
    .\run_find_best_model.ps1 -SkipSnow:$false     # include the 12 snow models
    .\run_find_best_model.ps1 -DryRun              # only print the command
#>
param(
    [string]$Gauge = "02249500",
    [int]$N = 200,
    [int]$StartYear = 2015,
    [int]$EndYear = 2025,
    [int]$NProcs = [Environment]::ProcessorCount,
    [int]$WarmupYears = 1,

    [ValidateSet("aorc", "nasa")]
    [string]$PrecipSource = "aorc",

    [ValidateSet("point", "basin")]
    [string]$NasaSampling = "point",

    [ValidateSet("Penman", "Hargreaves-Samani", "Blaney-Criddle", "Priestley-Taylor", "Turc")]
    [string]$EtMethod = "Turc",

    [string]$Models = "",          # e.g. "7,8,10-12"; empty = all 46 models
    [int]$Seed = 1,
    [switch]$SkipSnow = $true,     # use -SkipSnow:$false to include snow models

    [string]$EnvPath = "D:\micromamba\envs\findbestmodel",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot

if (-not (Get-Command micromamba -ErrorAction SilentlyContinue)) {
    throw "micromamba was not found on PATH."
}
if (-not (Test-Path $EnvPath)) {
    throw "Environment not found: $EnvPath. Create it with: micromamba create -p $EnvPath -f environment.yml"
}

$pyArgs = @(
    "find_best_model.py",
    "--gauge", $Gauge,
    "--n", $N,
    "--start-year", $StartYear,
    "--end-year", $EndYear,
    "--n-procs", $NProcs,
    "--warmup-years", $WarmupYears,
    "--precip-source", $PrecipSource,
    "--nasa-sampling", $NasaSampling,
    "--et-method", $EtMethod,
    "--seed", $Seed
)
if ($Models) { $pyArgs += @("--models", $Models) }
if ($SkipSnow) { $pyArgs += "--skip-snow" }

Write-Host "Environment: $EnvPath"
Write-Host "Command:     python $($pyArgs -join ' ')"
if ($DryRun) { return }

$start = Get-Date
Push-Location $repo
try {
    # -u: unbuffered output, so progress shows as it happens
    & micromamba run -p $EnvPath python -u @pyArgs
    $code = $LASTEXITCODE
}
finally {
    Pop-Location
}

$elapsed = (Get-Date) - $start
Write-Host ("Finished with exit code {0} in {1:hh\:mm\:ss}" -f $code, $elapsed)
Write-Host "Results: $(Join-Path $repo "performance\$Gauge")"
exit $code
