param([switch]$Setup, [switch]$Status, [switch]$StopAfterShot, [switch]$Foreground, [switch]$ResumeAtLogon)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$manifest = 'output\manifest.json'
if ($ResumeAtLogon) {
    if ((Test-Path .local/STOP) -or !(Test-Path $manifest)) { return }
    $saved = Get-Content $manifest -Raw | ConvertFrom-Json
    if ($saved.status -eq 'validated' -and (Test-Path output/LNMA_Live_Rap_Metal_Official_Music_Video.mp4)) { return }
}
if ($Status) {
    if (Test-Path $manifest) {
        $state = Get-Content $manifest -Raw | ConvertFrom-Json
        $done = @($state.completed.PSObject.Properties | Where-Object MemberType -eq NoteProperty).Count
        [pscustomobject]@{ Status=$state.status; Completed=$done; Remaining=$state.total-$done; Current=$state.current; WorkerPID=$state.pid; WorkerAlive=[bool](Get-Process -Id $state.pid -ErrorAction SilentlyContinue) }
    }
    return
}
if ($StopAfterShot) {
    New-Item -ItemType File -Force .local/STOP | Out-Null
    return
}
function Invoke-Checked([string]$Exe, [string[]]$Arguments) {
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Exe exited with $LASTEXITCODE" }
}
if ($Setup) {
    Invoke-Checked uv @('venv','--allow-existing','--python','3.12','.venv')
    Invoke-Checked uv @('pip','install','-r','requirements.txt')
    Invoke-Checked $python @('bootstrap.py')
    Invoke-Checked $python @('analyse.py')
    Invoke-Checked $python @('plan.py')
}
if (Test-Path .local/STOP) { Remove-Item -LiteralPath .local/STOP }
if ($Foreground) {
    Invoke-Checked $python @('-u','render.py')
} else {
    $stamp = Get-Date -Format yyyyMMdd-HHmmss
    $log = Join-Path $PSScriptRoot ".local\logs\worker-$stamp"
    Start-Process -FilePath $python -ArgumentList '-u','render.py' -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput "$log.out" -RedirectStandardError "$log.err" -PassThru | Select-Object Id
}
