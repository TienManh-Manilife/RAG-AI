# Create a fresh local .env without printing the generated password into commands/history.
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if (Test-Path '.env') { Write-Host '.env already exists; not overwriting.'; exit }
Add-Type -AssemblyName System.Security
$bytes = New-Object byte[] 32
[System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
$secret = [Convert]::ToBase64String($bytes).Replace('+','A').Replace('/','B')
$config = @"
BOOTSTRAP_ADMIN_NAME=Quản trị An Phát
BOOTSTRAP_ADMIN_EMAIL=admin@anphat.local
BOOTSTRAP_ADMIN_PASSWORD=$secret
RAG_MIN_RELEVANCE=0.38
"@
[System.IO.File]::WriteAllText((Join-Path (Get-Location) '.env'), $config, (New-Object System.Text.UTF8Encoding($false)))
Write-Host 'Local .env created. Save this initial admin password securely:'
Write-Host $secret
Write-Host 'Email: admin@anphat.local'
