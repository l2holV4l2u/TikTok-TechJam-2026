$pattern = 'wait_' + 'and_' + 'run'
$mine = $PID
$procs = Get-CimInstance Win32_Process -Filter "Name='bash.exe'" |
    Where-Object { $_.CommandLine -like "*$pattern*" -and $_.ProcessId -ne $mine -and
                   $_.CommandLine -notlike '*stop_watcher*' }
foreach ($p in $procs) {
    Write-Output ("killing {0}" -f $p.ProcessId)
    & taskkill /F /T /PID $p.ProcessId 2>&1 | Out-Null
}
Start-Sleep -Seconds 2
$left = @(Get-CimInstance Win32_Process -Filter "Name='bash.exe'" |
    Where-Object { $_.CommandLine -like "*$pattern*" -and $_.CommandLine -notlike '*stop_watcher*' })
Write-Output ("remaining: " + $left.Count)
