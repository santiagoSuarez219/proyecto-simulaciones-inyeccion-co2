param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$Args
)

$ErrorActionPreference = "Stop"

# Ensure the local `src/` layout is importable as a package.
$env:PYTHONPATH = (Join-Path $PSScriptRoot "src")

python -m cmg2tensor @Args

