# runctl: pause, resume, stop and start training runs

A small window ("Tichu runs" on the desktop) and a command line (`runctl.cmd`) for the
training jobs on Dave's PC. Stdlib Python only (tkinter + ctypes), Windows only.

**Install / update:** `powershell -ExecutionPolicy Bypass -File tools\runctl\install.ps1`.
It copies these files to `%USERPROFILE%\tichu-runctl` (outside the repo, so switching
branches never removes the tool) and creates the desktop shortcut.

## What it sees

- **Runs** are directories with a `run.cmd`, by default `tichu-ai\runs\<name>\run.cmd`
  (more roots: `"roots"` in `%LOCALAPPDATA%\tichu-runctl\config.json`). A run's processes
  are found by command line: anything whose command line names the run directory (the
  `run.cmd` path, `--dir runs\<name>`, ...) plus all training processes below it in the
  process tree. So a run shows up however it was started: scheduled task, Claude thread,
  or by hand.
- **Other** training processes (any `tichu*.exe`, or python/cmd running `train\...`,
  `dmc.py`, `iterate.py`, `train.py`, `calls.py`, `now.py`, `run.cmd` or something under a
  `runs\` dir) that match no run dir are listed as their own group and can be paused and
  stopped too, but not restarted.
- Scheduled tasks named `tichu*`, with Enable/Disable.

## Controls

| | |
|---|---|
| Pause | Suspends every process of the run (instant, nothing lost). It keeps its RAM and GPU memory. Optional timer: 30 min to 4 h, or until 9 pm; the run resumes by itself then. |
| Resume | Undo pause. |
| Stop | Kills the run. Work since its last checkpoint is lost; Start resumes from it (dmc.py: last finished chunk; iterate.py: last finished generation). |
| Start | Runs `run.cmd` through `runctl.py launch`, which keeps Windows awake while the run is running (not while paused) and does timed resumes even with the window closed. |
| Low priority | Idle CPU priority for the whole run; new worker processes inherit it. Doesn't change GPU sharing. |
| Hold new runs | `runctl start` refuses while it's on. Scheduled tasks still fire: disable them separately. |

## For threads/scripts starting runs

1. Put the run in `tichu-ai\runs\<name>\` with a `run.cmd` that `cd`s to the repo, runs the
   job with `--dir runs\<name>` and appends to `runs\<name>\console.txt`, ending with
   `echo ==== <script> exit %ERRORLEVEL% ...` (that line marks finished/failed). Make the job
   resumable by rerunning the same `run.cmd`.
2. Check `runctl status` first: don't start while `hold` is on, and don't resume or restart
   a run someone paused or stopped.
3. Start it with `runctl start <name>` (not `start`/`pythonw now.py`), so it gets keep-awake
   and timed resume.

```
%USERPROFILE%\tichu-runctl\runctl.cmd status [--json]
%USERPROFILE%\tichu-runctl\runctl.cmd start <name> [--low]
%USERPROFILE%\tichu-runctl\runctl.cmd pause <name>|--all [--for 2h]
%USERPROFILE%\tichu-runctl\runctl.cmd resume|stop|low <name>|--all
%USERPROFILE%\tichu-runctl\runctl.cmd hold on|off
```

State (pauses, hold) lives in `%LOCALAPPDATA%\tichu-runctl\state.json`.
