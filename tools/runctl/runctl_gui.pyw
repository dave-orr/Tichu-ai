"""Tichu runs: a small window to pause, resume, stop and start training runs. See runctl.py."""
import ctypes
import ctypes.wintypes as wt
import json
import os
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runctl  # noqa: E402

REFRESH_MS = 2000
OLD_DAYS = 3
NO_WINDOW = 0x08000000
NCPU = os.cpu_count() or 1
STATE_COLORS = dict(running="#1a7f37", paused="#b35900", stopped="#6e7781", finished="#0969da", failed="#cf222e")
PAUSE_CHOICES = [("Until I resume", None), ("For 30 minutes", 1800), ("For 1 hour", 3600),
                 ("For 2 hours", 7200), ("For 4 hours", 14400), ("Until 9 pm", "21:00")]


def until_from(choice):
    if choice is None:
        return None
    if isinstance(choice, str):
        h, m = map(int, choice.split(":"))
        t = time.localtime()
        target = time.mktime((t.tm_year, t.tm_mon, t.tm_mday, h, m, 0, 0, 0, -1))
        return target if target > time.time() else target + 86400
    return time.time() + choice


def ago(t):
    s = time.time() - t
    if s < 90:
        return "just now"
    if s < 5400:
        return f"{int(s / 60)} min ago"
    if s < 2 * 86400:
        return f"{s / 3600:.1f} h ago"
    return f"{int(s / 86400)} days ago"


def hhmm(t):
    return time.strftime("%H:%M", time.localtime(t)) if t else ""


class SystemTimes:
    def __init__(self):
        self.prev = None

    def cpu_pct(self):
        idle, kern, user = (wt.FILETIME() for _ in range(3))
        ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user))
        v = [(f.dwHighDateTime << 32) | f.dwLowDateTime for f in (idle, kern, user)]
        pct = None
        if self.prev:
            di, dk, du = (a - b for a, b in zip(v, self.prev))
            if dk + du > 0:
                pct = 100.0 * (dk + du - di) / (dk + du)
        self.prev = v
        return pct


class Poller(threading.Thread):
    """Runs a slow command in the background every `every` seconds; .value holds the last result."""

    def __init__(self, fn, every):
        super().__init__(daemon=True)
        self.fn, self.every, self.value = fn, every, None
        self.kick = threading.Event()

    def run(self):
        while True:
            try:
                self.value = self.fn()
            except Exception as e:  # keep polling
                self.value = e
            self.kick.wait(self.every)
            self.kick.clear()


def gpu_info():
    out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True,
                         creationflags=NO_WINDOW, timeout=10).stdout.strip().splitlines()
    if not out:
        return None
    u, used, total = (float(x) for x in out[0].split(","))
    return u, used / 1024, total / 1024


def sched_tasks():
    ps = ("Get-ScheduledTask -TaskName 'tichu*' -ErrorAction SilentlyContinue | ForEach-Object { "
          "$i = $_ | Get-ScheduledTaskInfo; [pscustomobject]@{name=$_.TaskName; state=[string]$_.State; "
          "next=if ($i.NextRunTime) { $i.NextRunTime.ToString('ddd HH:mm') } else { '' }; "
          "action=($_.Actions | ForEach-Object { $_.Execute + ' ' + $_.Arguments }) -join '; '} } | ConvertTo-Json")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                         creationflags=NO_WINDOW, timeout=60).stdout.strip()
    if not out:
        return []
    v = json.loads(out)
    return v if isinstance(v, list) else [v]


class RunRow:
    def __init__(self, app, parent, g):
        self.app, self.key = app, g["key"]
        f = self.frame = ttk.Frame(parent, padding=(8, 6), relief="groove", borderwidth=1)
        top = ttk.Frame(f)
        top.pack(fill="x")
        self.name = ttk.Label(top, text=g["name"], font=("Segoe UI", 11, "bold"))
        self.name.pack(side="left")
        self.state = tk.Label(top, text="", fg="white", font=("Segoe UI", 9, "bold"), padx=6)
        self.state.pack(side="left", padx=8)
        self.info = ttk.Label(top, text="", foreground="#57606a")
        self.info.pack(side="left")
        self.logline = ttk.Label(f, text="", foreground="#24292f", font=("Consolas", 9))
        self.logline.pack(fill="x", anchor="w", pady=(3, 3))
        b = ttk.Frame(f)
        b.pack(fill="x")
        self.b_start = ttk.Button(b, text="Start", width=8, command=self.start)
        self.b_pause = ttk.Menubutton(b, text="Pause", width=8)
        m = tk.Menu(self.b_pause, tearoff=False)
        for label, choice in PAUSE_CHOICES:
            m.add_command(label=label, command=lambda c=choice: self.pause(c))
        self.b_pause["menu"] = m
        self.b_resume = ttk.Button(b, text="Resume", width=8, command=self.resume)
        self.b_stop = ttk.Button(b, text="Stop", width=8, command=self.stop)
        self.low = tk.BooleanVar()
        self.b_low = ttk.Checkbutton(b, text="Low priority", variable=self.low, command=self.set_low)
        self.b_log = ttk.Button(b, text="Log", width=6, command=self.open_log)
        self.b_dir = ttk.Button(b, text="Folder", width=7, command=self.open_dir)
        for w in (self.b_start, self.b_pause, self.b_resume, self.b_stop):
            w.pack(side="left", padx=(0, 4))
        self.b_low.pack(side="left", padx=(8, 0))
        self.b_dir.pack(side="right")
        self.b_log.pack(side="right", padx=(0, 4))
        self.g = g

    def update(self, g, cpu):
        self.g = g
        s = g["state"]
        label = s.upper()
        if s == "paused" and g["until"]:
            label += " UNTIL " + hhmm(g["until"])
        if s == "failed":
            label += f" ({g.get('rc')})"
        self.state.config(text=label, bg=STATE_COLORS[s])
        bits = []
        if g["procs"]:
            bits.append(f"{len(g['procs'])} processes")
            if cpu is not None:
                bits.append(f"CPU {cpu:.0f}%")
            if g["started"]:
                bits.append("started " + time.strftime("%a %H:%M", time.localtime(g["started"])))
        if g["log"]:
            try:
                bits.append("log updated " + ago(os.path.getmtime(g["log"])))
            except OSError:
                pass
        if g.get("other"):
            bits.append("not a run dir")
        self.info.config(text="  ·  ".join(bits))
        last = runctl.tail(g["log"], 1) if g["log"] else []
        text = last[0] if last else ""
        self.logline.config(text=text[:140])
        live = bool(g["procs"])
        can_start = not live and bool(g.get("dir")) and os.path.isfile(os.path.join(g["dir"], "run.cmd"))
        self.b_start.state(["!disabled"] if can_start else ["disabled"])
        self.b_pause.state(["!disabled"] if live else ["disabled"])
        self.b_resume.state(["!disabled"] if s == "paused" else ["disabled"])
        self.b_stop.state(["!disabled"] if live else ["disabled"])
        self.b_low.state(["!disabled"] if live else ["disabled"])
        self.low.set(bool(g["low"]) and live)
        self.b_log.state(["!disabled"] if g["log"] else ["disabled"])
        self.b_dir.state(["!disabled"] if g.get("dir") else ["disabled"])

    def start(self):
        st = runctl.load_state()
        force = False
        if st["hold"]:
            if not messagebox.askyesno("Hold is on", f"Hold is on. Start {self.g['name']} anyway?"):
                return
            force = True
        self.app.act(lambda: runctl.start(self.g, force=force))

    def pause(self, choice):
        self.app.act(lambda: runctl.pause(self.g, until_from(choice)))

    def resume(self):
        self.app.act(lambda: runctl.resume(self.g))

    def stop(self):
        msg = (f"Stop {self.g['name']}? Its processes are killed. Work since its last checkpoint is lost; "
               "Start resumes from that checkpoint (for runs that support resuming, like dmc.py and iterate.py).")
        if messagebox.askyesno("Stop run", msg, icon="warning"):
            self.app.act(lambda: runctl.stop(self.g))

    def set_low(self):
        on = self.low.get()
        self.app.act(lambda: runctl.set_low(self.g, on))

    def open_log(self):
        if self.g["log"]:
            os.startfile(self.g["log"])

    def open_dir(self):
        if self.g.get("dir"):
            os.startfile(self.g["dir"])


class App:
    def __init__(self, root):
        self.root = root
        root.title("Tichu runs")
        root.geometry("760x560")
        root.minsize(620, 300)
        try:
            ttk.Style().theme_use("vista")
        except tk.TclError:
            pass
        self.sys = SystemTimes()
        self.prev_cpu = {}
        self.gpu = Poller(gpu_info, 3)
        self.gpu.start()
        self.tasks = Poller(sched_tasks, 30)
        self.tasks.start()

        bar = ttk.Frame(root, padding=(10, 8))
        bar.pack(fill="x")
        self.sysinfo = ttk.Label(bar, text="", font=("Segoe UI", 10))
        self.sysinfo.pack(side="left")
        self.hold = tk.BooleanVar()
        ttk.Checkbutton(bar, text="Hold new runs", variable=self.hold, command=self.set_hold).pack(side="right")
        ttk.Button(bar, text="Resume all", command=self.resume_all).pack(side="right", padx=6)
        pa = ttk.Menubutton(bar, text="Pause all")
        m = tk.Menu(pa, tearoff=False)
        for label, choice in PAUSE_CHOICES:
            m.add_command(label=label, command=lambda c=choice: self.pause_all(c))
        pa["menu"] = m
        pa.pack(side="right")
        self.msg = ttk.Label(root, text="", foreground="#cf222e", padding=(10, 0))
        self.msg.pack(fill="x")

        # scrollable list
        outer = ttk.Frame(root)
        outer.pack(fill="both", expand=True, padx=10, pady=(4, 0))
        self.canvas = tk.Canvas(outer, highlightthickness=0)
        sb = ttk.Scrollbar(outer, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self.win, width=e.width))
        self.canvas.configure(yscrollcommand=sb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        root.bind_all("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-e.delta / 120), "units"))
        self.runs_box = ttk.Frame(self.inner)
        self.runs_box.pack(fill="x")
        self.show_old = tk.BooleanVar(value=False)
        opt = ttk.Frame(self.inner)
        opt.pack(fill="x", pady=(4, 6))
        ttk.Checkbutton(opt, text=f"Show runs idle for more than {OLD_DAYS} days", variable=self.show_old,
                        command=self.refresh).pack(side="left")
        ttk.Label(self.inner, text="Scheduled tasks", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(6, 2))
        self.tasks_box = ttk.Frame(self.inner)
        self.tasks_box.pack(fill="x")
        ttk.Label(self.inner, foreground="#57606a", wraplength=700, justify="left", padding=(0, 10),
                  text="Pause freezes a run instantly and keeps its place; it still holds its memory, including "
                       "GPU memory. Stop frees everything; Start later picks up from the run's last checkpoint. "
                       "Hold new runs makes runctl refuse to start runs, but scheduled tasks still fire: disable "
                       "those here too.").pack(anchor="w")
        self.rows = {}
        self.task_sig = None
        self.refresh()

    def act(self, fn):
        try:
            fn()
            self.msg.config(text="")
        except SystemExit as e:
            self.msg.config(text=str(e))
        except Exception as e:
            self.msg.config(text=f"{type(e).__name__}: {e}")
        self.refresh(reschedule=False)

    def set_hold(self):
        st = runctl.load_state()
        st["hold"] = self.hold.get()
        runctl.save_json(runctl.STATE_PATH, st)

    def pause_all(self, choice):
        u = until_from(choice)
        self.act(lambda: [runctl.pause(g, u) for g in runctl.groups() if g["procs"]])

    def resume_all(self):
        self.act(lambda: [runctl.resume(g) for g in runctl.groups() if g["procs"]])

    def visible(self, g):
        if g["procs"] or self.show_old.get():
            return True
        try:
            return time.time() - os.path.getmtime(g["log"] or os.path.join(g["dir"], "run.cmd")) < OLD_DAYS * 86400
        except (OSError, TypeError):
            return False

    def refresh(self, reschedule=True):
        try:
            runctl.tick()
            gs, st = runctl.status()
            self.hold.set(st["hold"])
            now = time.time()
            shown = [g for g in gs if self.visible(g)]
            keys = [g["key"] for g in shown]
            if keys != list(self.rows):
                for r in self.rows.values():
                    r.frame.destroy()
                self.rows = {}
                for g in shown:
                    r = RunRow(self, self.runs_box, g)
                    r.frame.pack(fill="x", pady=3)
                    self.rows[g["key"]] = r
            if not shown:
                if not getattr(self, "empty", None):
                    self.empty = ttk.Label(self.runs_box, text="No runs.", foreground="#57606a")
                    self.empty.pack(anchor="w")
            elif getattr(self, "empty", None):
                self.empty.destroy()
                self.empty = None
            for g in shown:
                cpu_s = sum(p["cpu"] for p in g["procs"])
                prev = self.prev_cpu.get(g["key"])
                pct = None
                if prev and g["procs"] and now > prev[1]:
                    pct = max(0.0, (cpu_s - prev[0]) / (now - prev[1]) / NCPU * 100)
                self.prev_cpu[g["key"]] = (cpu_s, now)
                self.rows[g["key"]].update(g, pct)
            self.update_sysinfo()
            self.update_tasks()
        except Exception as e:
            self.msg.config(text=f"refresh failed: {type(e).__name__}: {e}")
        if reschedule:
            self.root.after(REFRESH_MS, self.refresh)

    def update_sysinfo(self):
        bits = []
        c = self.sys.cpu_pct()
        if c is not None:
            bits.append(f"CPU {c:.0f}%")
        g = self.gpu.value
        if isinstance(g, tuple):
            bits.append(f"GPU {g[0]:.0f}%  ·  GPU memory {g[1]:.1f} / {g[2]:.0f} GB")
        self.sysinfo.config(text="   ".join(bits))

    def update_tasks(self):
        ts = self.tasks.value
        sig = json.dumps(ts, default=str)
        if sig == self.task_sig:
            return
        self.task_sig = sig
        for w in self.tasks_box.winfo_children():
            w.destroy()
        if isinstance(ts, Exception):
            ttk.Label(self.tasks_box, text=f"couldn't list tasks: {ts}").pack(anchor="w")
            return
        if ts is None:
            ttk.Label(self.tasks_box, text="loading…", foreground="#57606a").pack(anchor="w")
            return
        if not ts:
            ttk.Label(self.tasks_box, text="No tichu* scheduled tasks.", foreground="#57606a").pack(anchor="w")
            return
        for t in ts:
            row = ttk.Frame(self.tasks_box, padding=(8, 3))
            row.pack(fill="x")
            on = t["state"] != "Disabled"
            ttk.Label(row, text=t["name"], width=22, font=("Segoe UI", 10, "bold")).pack(side="left")
            ttk.Label(row, text=t["state"] + (f", next {t['next']}" if on and t["next"] else ""),
                      width=24).pack(side="left")
            ttk.Button(row, text="Disable" if on else "Enable", width=8,
                       command=lambda n=t["name"], e=not on: self.toggle_task(n, e)).pack(side="right")

    def toggle_task(self, name, enable):
        r = subprocess.run(["schtasks", "/change", "/tn", name, "/enable" if enable else "/disable"],
                           capture_output=True, text=True, creationflags=NO_WINDOW)
        self.msg.config(text="" if r.returncode == 0 else (r.stderr or r.stdout).strip())
        self.tasks.kick.set()


def main():
    ctypes.windll.shcore.SetProcessDpiAwareness(1) if hasattr(ctypes.windll, "shcore") else None
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
