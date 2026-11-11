# Copies runctl to %USERPROFILE%\tichu-runctl (outside the repo, so branch switches don't touch it)
# and puts a "Tichu runs" shortcut on the desktop. Rerun after pulling a newer version.
$src = $PSScriptRoot
$dst = Join-Path $env:USERPROFILE 'tichu-runctl'
New-Item -ItemType Directory -Force $dst | Out-Null
if ((Resolve-Path $src).Path -ne (Resolve-Path $dst).Path) {
    Copy-Item (Join-Path $src '*') $dst -Force -Exclude '__pycache__'
}
$pyw = Join-Path $env:USERPROFILE 'projects\tichu-ai\.venv\Scripts\pythonw.exe'
$lnk = Join-Path ([Environment]::GetFolderPath('Desktop')) 'Tichu runs.lnk'
$s = (New-Object -ComObject WScript.Shell).CreateShortcut($lnk)
$s.TargetPath = $pyw
$s.Arguments = '"' + (Join-Path $dst 'runctl_gui.pyw') + '"'
$s.WorkingDirectory = $dst
$s.IconLocation = "$env:SystemRoot\System32\shell32.dll,24"
$s.Description = 'Pause, resume, stop and start Tichu training runs'
$s.Save()
"installed to $dst; shortcut: $lnk"
