# scripts/archive_debug_logs.sh を Windows タスクスケジューラへ登録する。
# 毎日 04:00 に走るが、スクリプト側が当月・前月を除外するため実際に圧縮が動くのは月替わり後の初回だけ。
# PC が停止していて実行を逃した場合は StartWhenAvailable で次回起動時に拾う。
#
# 実行: powershell -ExecutionPolicy Bypass -File scripts\register_archive_task.ps1
# 解除: Unregister-ScheduledTask -TaskName 'Chotgor debug log archive' -Confirm:$false

$action = New-ScheduledTaskAction `
    -Execute 'C:\Program Files\Git\bin\bash.exe' `
    -Argument '-lc "/c/Users/seamo/Chotgor/scripts/archive_debug_logs.sh"'

$trigger = New-ScheduledTaskTrigger -Daily -At 4am

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -DontStopIfGoingOnBatteries `
    -AllowStartIfOnBatteries

Register-ScheduledTask `
    -TaskName 'Chotgor debug log archive' `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description 'Chotgor: debug/ の生ログを月単位で tar.gz へ退避する（当月・前月は残す）。毎日走るが対象月が無ければ即終了する。' `
    -Force | Out-Null

Get-ScheduledTask -TaskName 'Chotgor debug log archive' |
    Select-Object TaskName, State |
    Format-List
