"""runctl: see, pause, resume, stop and start training runs on this PC.

A "run" is a directory with a run.cmd in it (by default each tichu-ai/runs/<name>/run.cmd).
runctl finds a run's processes by their command lines (anything mentioning the run's
directory), so runs started any way at all (a scheduled task, a Claude thread, by hand)
show up. Training processes that belong to no run dir are listed too, as "other".

    runctl status [--json]            what is running, paused, stopped
    runctl start RUN [--force]        start RUN/run.cmd in the background (refused while held)
    runctl pause RUN|--all [--for 2h] suspend every process of the run (instant; GPU memory stays allocated)
    runctl resume RUN|--all           undo pause
    runctl stop RUN|--all             kill the run; `start` later resumes from its last checkpoint
    runctl low RUN|--all [--off]      idle CPU priority (the PC stays responsive), or back to normal
    runctl hold on|off                while on, `start` refuses (so no new runs begin)

RUN is a run name (dmc1), a run directory, or a key shown by `status` (other groups).
Stdlib only (ctypes); Windows only.
"""
import argparse
import ctypes
import ctypes.wintypes as wt
import datetime
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STATE_DIR = os.environ.get("RUNCTL_STATE") or os.path.join(os.environ.get("LOCALAPPDATA", HERE), "tichu-runctl")
STATE_PATH = os.path.join(STATE_DIR, "state.json")
CONFIG_PATH = os.path.join(STATE_DIR, "config.json")
DEFAULT_REPO = os.path.join(os.path.expanduser("~"), "projects", "tichu-ai")

# ---------------------------------------------------------------- Win32 plumbing

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
ntdll = ctypes.WinDLL("ntdll")

TH32CS_SNAPPROCESS = 0x2
PROCESS_TERMINATE = 0x0001
PROCESS_SUSPEND_RESUME = 0x0800
PROCESS_SET_INFORMATION = 0x0200
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
IDLE_PRIORITY_CLASS = 0x40
BELOW_NORMAL_PRIORITY_CLASS = 0x4000
NORMAL_PRIORITY_CLASS = 0x20
CREATE_NO_WINDOW = 0x08000000
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [("dwSize", wt.DWORD), ("cntUsage", wt.DWORD), ("th32ProcessID", wt.DWORD),
                ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wt.DWORD),
                ("cntThreads", wt.DWORD), ("th32ParentProcessID", wt.DWORD),
                ("pcPriClassBase", ctypes.c_long), ("dwFlags", wt.DWORD),
                ("szExeFile", ctypes.c_wchar * 260)]


class UNICODE_STRING(ctypes.Structure):
    _fields_ = [("Length", ctypes.c_ushort), ("MaximumLength", ctypes.c_ushort),
                ("Buffer", ctypes.c_void_p)]


k32.CreateToolhelp32Snapshot.restype = wt.HANDLE
k32.CreateToolhelp32Snapshot.argtypes = [wt.DWORD, wt.DWORD]
k32.Process32FirstW.argtypes = [wt.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
k32.Process32NextW.argtypes = [wt.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
k32.OpenProcess.restype = wt.HANDLE
k32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
k32.CloseHandle.argtypes = [wt.HANDLE]
k32.GetProcessTimes.argtypes = [wt.HANDLE] + [ctypes.POINTER(ctypes.c_ulonglong)] * 4
k32.TerminateProcess.argtypes = [wt.HANDLE, wt.UINT]
k32.GetPriorityClass.argtypes = [wt.HANDLE]
k32.GetPriorityClass.restype = wt.DWORD
k32.SetPriorityClass.argtypes = [wt.HANDLE, wt.DWORD]
k32.SetThreadExecutionState.argtypes = [wt.DWORD]
k32.SetThreadExecutionState.restype = wt.DWORD
ntdll.NtQueryInformationProcess.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.ULONG,
                                            ctypes.POINTER(wt.ULONG)]
ntdll.NtQueryInformationProcess.restype = ctypes.c_long
ntdll.NtSuspendProcess.argtypes = [wt.HANDLE]
ntdll.NtSuspendProcess.restype = ctypes.c_long
ntdll.NtResumeProcess.argtypes = [wt.HANDLE]
ntdll.NtResumeProcess.restype = ctypes.c_long


def _open(pid, access):
    h = k32.OpenProcess(access, False, pid)
    return h or None


def _cmdline(h):
    size = wt.ULONG(0)
    ntdll.NtQueryInformationProcess(h, 60, None, 0, ctypes.byref(size))
    if not size.value:
        return ""
    buf = ctypes.create_string_buffer(size.value)
    if ntdll.NtQueryInformationProcess(h, 60, buf, size.value, ctypes.byref(size)) != 0:
        return ""
    us = UNICODE_STRING.from_buffer(buf)
    return ctypes.wstring_at(us.Buffer, us.Length // 2) if us.Buffer else ""


def _times(h):
    c, e, kt, ut = (ctypes.c_ulonglong() for _ in range(4))
    if not k32.GetProcessTimes(h, ctypes.byref(c), ctypes.byref(e), ctypes.byref(kt), ctypes.byref(ut)):
        return 0, 0.0
    return c.value, (kt.value + ut.value) / 1e7


def ft_unix(ft):
    """FILETIME (100 ns since 1601) to a Unix timestamp."""
    return ft / 1e7 - 11644473600 if ft else 0


# names whose command lines we read (others are only used to link the tree)
CANDIDATE_NAMES = ("python.exe", "pythonw.exe", "cmd.exe")


def snapshot():
    """{pid: dict(pid, ppid, name, cmd, ctime, cpu, prio)} for every process we can see."""
    snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    procs = {}
    e = PROCESSENTRY32W()
    e.dwSize = ctypes.sizeof(e)
    ok = k32.Process32FirstW(snap, ctypes.byref(e))
    while ok:
        procs[e.th32ProcessID] = dict(pid=e.th32ProcessID, ppid=e.th32ParentProcessID,
                                      name=e.szExeFile.lower(), cmd="", ctime=0, cpu=0.0, prio=0)
        ok = k32.Process32NextW(snap, ctypes.byref(e))
    k32.CloseHandle(snap)
    for p in procs.values():
        n = p["name"]
        if n in CANDIDATE_NAMES or n.startswith("tichu"):
            h = _open(p["pid"], PROCESS_QUERY_LIMITED_INFORMATION)
            if h:
                try:
                    p["cmd"] = _cmdline(h)
                    p["ctime"], p["cpu"] = _times(h)
                    p["prio"] = k32.GetPriorityClass(h)
                finally:
                    k32.CloseHandle(h)
    return procs


def _suspend(pid, ctime, resume=False):
    h = _open(pid, PROCESS_SUSPEND_RESUME | PROCESS_QUERY_LIMITED_INFORMATION)
    if not h:
        return False
    try:
        if ctime and _times(h)[0] != ctime:
            return False  # pid reused by another process
        return (ntdll.NtResumeProcess(h) if resume else ntdll.NtSuspendProcess(h)) == 0
    finally:
        k32.CloseHandle(h)


def _terminate(pid, ctime):
    h = _open(pid, PROCESS_TERMINATE | PROCESS_QUERY_LIMITED_INFORMATION)
    if not h:
        return False
    try:
        if ctime and _times(h)[0] != ctime:
            return False
        return bool(k32.TerminateProcess(h, 1))
    finally:
        k32.CloseHandle(h)


def _set_priority(pid, ctime, cls):
    h = _open(pid, PROCESS_SET_INFORMATION | PROCESS_QUERY_LIMITED_INFORMATION)
    if not h:
        return False
    try:
        if ctime and _times(h)[0] != ctime:
            return False
        return bool(k32.SetPriorityClass(h, cls))
    finally:
        k32.CloseHandle(h)


# ---------------------------------------------------------------- config and state

def load_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".%d.tmp" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    for _ in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # another runctl is reading it
            time.sleep(0.05)
    os.replace(tmp, path)


def config():
    cfg = load_json(CONFIG_PATH, {})
    cfg.setdefault("repo", os.environ.get("TICHU_AI", DEFAULT_REPO))
    cfg.setdefault("roots", [os.path.join(cfg["repo"], "runs")])
    return cfg


def load_state():
    st = load_json(STATE_PATH, {})
    st.setdefault("hold", False)
    st.setdefault("paused", {})  # key -> {pids: [[pid, ctime]], at, until}
    return st


def norm(path):
    return os.path.normcase(os.path.abspath(path))


def known_runs(cfg=None):
    """[(name, dir)] for every <root>/<name>/run.cmd."""
    cfg = cfg or config()
    out = []
    for root in cfg["roots"]:
        try:
            names = sorted(os.listdir(root))
        except OSError:
            continue
        for n in names:
            d = os.path.join(root, n)
            if os.path.isfile(os.path.join(d, "run.cmd")):
                out.append((n, d))
    return out


# ---------------------------------------------------------------- grouping processes into runs

TRAIN_HINTS = re.compile(r"(?i)(\btrain[\\/]|dmc\.py|iterate\.py|train\.py|calls\.py|now\.py|"
                         r"run\.cmd|runctl\.py\"?\s+launch|[\\/]runs[\\/])")
def is_training(p):
    n, cmd = p["name"], p["cmd"]
    if n.startswith("tichu") and n.endswith(".exe"):
        return True
    if not cmd or ("runctl" in cmd.lower() and not is_manager(p)):
        return False  # runctl itself (the window, a status call)
    if n in ("python.exe", "pythonw.exe", "cmd.exe"):
        return bool(TRAIN_HINTS.search(cmd))
    return False


def is_manager(p):
    """The runctl launcher: never suspended (it holds keep-awake and does timed resumes)."""
    return bool(re.search(r"(?i)runctl\.py\"?\s+launch\b", p["cmd"]))


def _parent(procs, p):
    q = procs.get(p["ppid"])
    if q is None or q["pid"] == p["pid"]:
        return None
    if q["ctime"] and p["ctime"] and q["ctime"] > p["ctime"]:
        return None  # ppid was reused
    return q


def _mentions(cmd, d):
    c = cmd.replace("/", "\\").lower()
    dl = norm(d).lower()
    if dl in c:
        return True
    parent, name = os.path.split(dl)
    if os.path.basename(parent) == "runs":
        return re.search(r"(^|[\s\"'=\\])runs\\" + re.escape(name) + r"($|[\s\"'\\])", c) is not None
    return False


def groups(procs=None, cfg=None):
    """List of run groups: dict(key, name, dir, procs=[...]) for known runs (running or not),
    plus one group per tree of training processes not tied to a known run."""
    procs = snapshot() if procs is None else procs
    cfg = cfg or config()
    train = {pid: p for pid, p in procs.items() if is_training(p) and pid != os.getpid()}
    # roots: climb while the parent is also a training process
    trees = {}
    for p in train.values():
        r = p
        while True:
            q = _parent(procs, r)
            if q is None or q["pid"] not in train:
                break
            r = q
        trees.setdefault(r["pid"], []).append(p)
    runs = known_runs(cfg)
    out = [dict(key=norm(d), name=n, dir=d, procs=[]) for n, d in runs]
    by_key = {g["key"]: g for g in out}
    for rootpid, members in sorted(trees.items()):
        members.sort(key=lambda p: p["ctime"])
        hit = None
        for g in out:
            if any(_mentions(p["cmd"], g["dir"]) for p in members):
                hit = g
                break
        if hit is not None:
            hit["procs"].extend(members)
        else:
            out.append(dict(key="pid%d" % rootpid, name=_label(members), dir=_guess_dir(members),
                            procs=members, other=True))
    for g in out:
        g["procs"].sort(key=lambda p: p["ctime"])
    return out


def _guess_dir(members):
    for p in members:
        m = re.search(r"--(?:dir|out)[\s=]+\"?([^\s\"]+)", p["cmd"])
        if m:
            d = m.group(1)
            return d if os.path.isdir(d) else os.path.dirname(d)
    return None


def _label(members):
    root = members[0]
    cmd = root["cmd"]
    m = re.search(r"(?i)([\w.\-]+\.py)\b", cmd)
    tool = m.group(1) if m else root["name"]
    d = _guess_dir(members)
    return f"{tool} ({os.path.basename(os.path.normpath(d))})" if d else f"{tool} (pid {root['pid']})"


# ---------------------------------------------------------------- status

def last_exit(d):
    """('exit', rc, when) from the last '==== ... exit N' line run.cmd appends to console.txt."""
    p = os.path.join(d, "console.txt")
    try:
        with open(p, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 4000))
            tail = f.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return None
    for line in reversed(tail):
        if line.startswith("===="):
            m = re.search(r"exit (-?\d+)", line)
            return int(m.group(1)) if m else None
    return None


def log_path(d):
    if not d:
        return None
    for n in ("log.txt", "console.txt", "console.log"):
        p = os.path.join(d, n)
        if os.path.isfile(p):
            return p
    return None


def tail(path, n=1):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 8000))
            lines = [l for l in f.read().decode("utf-8", "replace").splitlines() if l.strip()]
        return lines[-n:]
    except OSError:
        return []


def describe(g, st):
    """Fill in state: running / paused / stopped / finished / failed."""
    live = {(p["pid"], p["ctime"]) for p in g["procs"]}
    pz = st["paused"].get(g["key"])
    if g["procs"]:
        paused = bool(pz) and any(tuple(x) in live for x in pz["pids"])
        g["state"] = "paused" if paused else "running"
        g["until"] = pz.get("until") if paused else None
        g["low"] = all(p["prio"] in (IDLE_PRIORITY_CLASS, BELOW_NORMAL_PRIORITY_CLASS)
                       for p in g["procs"] if p["prio"])
    else:
        rc = last_exit(g["dir"]) if g.get("dir") else None
        g["state"] = "stopped" if rc is None else ("finished" if rc == 0 else "failed")
        g["rc"] = rc
        g["until"], g["low"] = None, False
    g["log"] = log_path(g.get("dir"))
    g["started"] = ft_unix(min((p["ctime"] for p in g["procs"] if p["ctime"]), default=0))
    return g


def status(procs=None):
    st = load_state()
    gs = [describe(g, st) for g in groups(procs)]
    # forget pause records of groups that are gone
    keys = {g["key"] for g in gs if g["procs"]}
    stale = [k for k in st["paused"] if k not in keys]
    if stale:
        for k in stale:
            del st["paused"][k]
        save_json(STATE_PATH, st)
    return gs, st


# ---------------------------------------------------------------- actions

def find(gs, which):
    w = which.lower()
    for g in gs:
        if w in (g["name"].lower(), g["key"].lower()) or (g.get("dir") and os.path.isdir(which)
                                                           and norm(which) == norm(g["dir"])):
            return g
    raise SystemExit(f"no run called {which!r}; see `runctl status`")


def pause(g, until=None):
    """Suspend every process in the run (except the launcher). Re-pausing just updates `until`."""
    st = load_state()
    rec = st["paused"].get(g["key"], {"pids": [], "at": time.time()})
    have = {tuple(x) for x in rec["pids"]}
    for _ in range(2):  # a second pass catches a child spawned during the first
        for p in g["procs"]:
            if is_manager(p) or (p["pid"], p["ctime"]) in have:
                continue
            if _suspend(p["pid"], p["ctime"]):
                have.add((p["pid"], p["ctime"]))
        fresh = describe_procs(g)
        if all((p["pid"], p["ctime"]) in have or is_manager(p) for p in fresh):
            break
        g = dict(g, procs=fresh)
    rec["pids"] = [list(x) for x in sorted(have)]
    rec["until"] = until
    st = load_state()
    st["paused"][g["key"]] = rec
    save_json(STATE_PATH, st)
    return len(have)


def describe_procs(g):
    for h in groups():
        if h["key"] == g["key"]:
            return h["procs"]
    return []


def resume(g):
    st = load_state()
    rec = st["paused"].pop(g["key"], None)
    save_json(STATE_PATH, st)
    n = 0
    if rec:
        for pid, ctime in rec["pids"]:
            n += _suspend(pid, ctime, resume=True)
    return n


def stop(g):
    """Freeze the whole tree first (so nothing reacts to its children dying), then kill it."""
    procs = g["procs"]
    for p in procs:
        _suspend(p["pid"], p["ctime"])
    n = 0
    for p in reversed(procs):  # launcher (oldest) last
        n += _terminate(p["pid"], p["ctime"])
    st = load_state()
    st["paused"].pop(g["key"], None)
    save_json(STATE_PATH, st)
    return n


def set_low(g, on=True):
    cls = IDLE_PRIORITY_CLASS if on else NORMAL_PRIORITY_CLASS
    return sum(_set_priority(p["pid"], p["ctime"], cls) for p in g["procs"])


def python_exe(windowless=True):
    exe = sys.executable
    d = os.path.dirname(exe)
    w = os.path.join(d, "pythonw.exe")
    return w if windowless and os.path.exists(w) else exe


def start(g, force=False, low=False):
    st = load_state()
    if st["hold"] and not force:
        raise SystemExit("hold is on: not starting new runs (runctl hold off, or --force)")
    if g["procs"]:
        raise SystemExit(f"{g['name']} is already running")
    if not g.get("dir") or not os.path.isfile(os.path.join(g["dir"], "run.cmd")):
        raise SystemExit(f"{g['name']} has no run.cmd")
    args = [python_exe(), os.path.abspath(__file__), "launch", g["dir"]]
    if low:
        args.append("--low")
    flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
    try:
        p = subprocess.Popen(args, creationflags=flags | CREATE_BREAKAWAY_FROM_JOB, close_fds=True,
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:  # breakaway not allowed in this job
        p = subprocess.Popen(args, creationflags=flags, close_fds=True,
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return p.pid


def launch(d, low=False):
    """Run d/run.cmd, keeping Windows awake while the run is running (not while it is paused),
    and resuming a timed pause when it is due even if no runctl window is open."""
    d = os.path.abspath(d)
    key = norm(d)
    logf = os.path.join(d, "runctl.txt")

    def log(msg):
        with open(logf, "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {msg}\n")

    cwd = os.path.dirname(os.path.dirname(d)) if os.path.basename(os.path.dirname(d)).lower() == "runs" else d
    flags = CREATE_NO_WINDOW | (IDLE_PRIORITY_CLASS if low else 0)
    log(f"launcher pid {os.getpid()}: starting run.cmd")
    child = subprocess.Popen(["cmd.exe", "/c", os.path.join(d, "run.cmd")], cwd=cwd, creationflags=flags)
    awake = None
    try:
        while child.poll() is None:
            st = load_state()
            rec = st["paused"].get(key)
            if rec and rec.get("until") and time.time() >= rec["until"]:
                gs = [g for g in groups() if g["key"] == key]
                if gs:
                    resume(gs[0])
                    log("timed pause over: resumed")
                rec = None
            want = rec is None
            if want != awake:
                k32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if want else 0))
                awake = want
            time.sleep(5)
    finally:
        k32.SetThreadExecutionState(ES_CONTINUOUS)
    log(f"run.cmd exited with {child.returncode}")
    return child.returncode


def tick():
    """Resume timed pauses that are due (the GUI calls this; launchers do it for their own run)."""
    st = load_state()
    due = [k for k, r in st["paused"].items() if r.get("until") and time.time() >= r["until"]]
    if not due:
        return []
    done = []
    for g in groups():
        if g["key"] in due:
            resume(g)
            done.append(g["name"])
    return done


def parse_duration(s):
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([smhd]?)", s.strip().lower())
    if not m:
        raise SystemExit(f"bad duration {s!r} (try 30m, 2h)")
    return float(m.group(1)) * {"": 60, "s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]


# ---------------------------------------------------------------- CLI

def _fmt_time(t):
    return datetime.datetime.fromtimestamp(t).strftime("%H:%M") if t else ""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status")
    s.add_argument("--json", action="store_true")
    for name in ("pause", "resume", "stop", "low"):
        s = sub.add_parser(name)
        s.add_argument("run", nargs="?")
        s.add_argument("--all", action="store_true")
        if name == "pause":
            s.add_argument("--for", dest="dur")
        if name == "low":
            s.add_argument("--off", action="store_true")
    s = sub.add_parser("start")
    s.add_argument("run")
    s.add_argument("--force", action="store_true", help="start even while hold is on")
    s.add_argument("--low", action="store_true", help="start at idle CPU priority")
    s = sub.add_parser("hold")
    s.add_argument("onoff", choices=["on", "off"], nargs="?")
    s = sub.add_parser("launch")
    s.add_argument("dir")
    s.add_argument("--low", action="store_true")
    sub.add_parser("tick")
    sub.add_parser("gui")
    a = ap.parse_args(argv)

    if a.cmd == "launch":
        return launch(a.dir, a.low)
    if a.cmd == "gui":
        import runctl_gui
        return runctl_gui.main()
    if a.cmd == "tick":
        print(" ".join(tick()))
        return 0
    if a.cmd == "hold":
        st = load_state()
        if a.onoff:
            st["hold"] = a.onoff == "on"
            save_json(STATE_PATH, st)
        print("hold", "on" if st["hold"] else "off")
        return 0

    gs, st = status()
    if a.cmd == "status":
        if a.json:
            print(json.dumps(dict(hold=st["hold"], runs=[dict(
                name=g["name"], key=g["key"], dir=g.get("dir"), state=g["state"], until=g["until"],
                low=g["low"], pids=[p["pid"] for p in g["procs"]], log=g["log"],
                other=bool(g.get("other"))) for g in gs]), indent=1))
            return 0
        print("hold: " + ("ON (new runs refused)" if st["hold"] else "off"))
        for g in gs:
            extra = ""
            if g["state"] == "paused" and g["until"]:
                extra = " until " + _fmt_time(g["until"])
            if g["procs"]:
                extra += f"  {len(g['procs'])} procs since {_fmt_time(g['started'])}"
                if g["low"]:
                    extra += "  low priority"
            print(f"{g['name']:<28} {g['state']}{extra}")
        return 0
    if a.cmd == "start":
        g = find(gs, a.run)
        print(f"started {g['name']} (launcher pid {start(g, a.force, a.low)})")
        return 0
    if a.all:
        targets = [g for g in gs if g["procs"]]
    elif a.run:
        targets = [find(gs, a.run)]
    else:
        raise SystemExit("name a run, or --all")
    for g in targets:
        if a.cmd == "pause":
            until = time.time() + parse_duration(a.dur) if a.dur else None
            n = pause(g, until)
            print(f"paused {g['name']} ({n} processes)" + (f" until {_fmt_time(until)}" if until else ""))
        elif a.cmd == "resume":
            print(f"resumed {g['name']} ({resume(g)} processes)")
        elif a.cmd == "stop":
            print(f"stopped {g['name']} ({stop(g)} processes killed)")
        elif a.cmd == "low":
            print(f"{g['name']}: {'normal' if a.off else 'idle'} priority on {set_low(g, not a.off)} processes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
