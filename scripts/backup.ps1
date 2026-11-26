# Stop app writes, copy SQLite via tar from the named volume, then resume.
$ErrorActionPreference='Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$backupDir = Join-Path (Get-Location) 'backups'
New-Item -ItemType Directory -Force $backupDir | Out-Null
Write-Host 'Stopping only Streamlit app to make SQLite consistent...'
docker compose stop app
try {
    $vol = (docker volume ls --format '{{.Name}}' | Select-String 'app_data$' | Select-Object -First 1).ToString().Trim()
    if (-not $vol) { throw 'Cannot identify app_data volume' }
    $out = Join-Path $backupDir "rag_data_$stamp.tar.gz"
    docker run --rm -v "${vol}:/source:ro" -v "${backupDir}:/backup" alpine sh -c "tar -czf /backup/rag_data_$stamp.tar.gz -C /source ."
    Write-Host "Backup written: $out"
} finally {
    docker compose start app
}
