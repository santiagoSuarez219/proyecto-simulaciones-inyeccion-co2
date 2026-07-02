$root = (Get-Location).Path
# $sims = @(
#     "28")
$sims = @(
    "29","30","31","32","33",
    "102","103","104","168",
    "305","306","307","308","309","310","311"
)

$total = $sims.Count
$i = 0

foreach ($sim in $sims) {
    if (Test-Path "$root/data/processed/$sim/layer_cubes_report.json") {
        Write-Host "[SKIP] $sim ya procesada"
        continue
    }

    $i++
    Write-Host "[$i/$total] Lanzando $sim - $(Get-Date -Format 'HH:mm:ss')"

    Start-Job -Name $sim -ScriptBlock {
        param($s, $r)
        Set-Location $r
        $arglist = @(
            "-m", "fno_co2.etl",
            "--sf-path",        "data/raw/$s/SF.txt",
            "--vd-path",        "data/raw/$s/VD.txt",
            "--cohesion-path",  "data/raw/$s/cohesion.txt",
            "--afi-path",       "data/raw/$s/friction_angle.txt",
            "--injection-path", "data/raw/$s/inyeccion.xlsx",
            "--nz", "97", "--nj", "50", "--ni", "50",
            "--no-normalize",
            "--output-dir", "data/processed/$s"
        )
        & "$r\.venv\Scripts\python.exe" @arglist
    } -ArgumentList $sim, $root | Out-Null

    if ($i -lt $total) {
        Start-Sleep -Seconds 180
    }
}

Write-Host "Todos lanzados. Esperando que terminen..."
Get-Job | Wait-Job | Out-Null
Get-Job | ForEach-Object {
    Write-Host "$($_.Name) - $($_.State)"
    Remove-Job $_
}
Write-Host "Listo."
