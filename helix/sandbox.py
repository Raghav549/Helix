"""Fail-closed, resource-bounded local Python execution on Linux.

Uses unprivileged user, mount, PID and network namespaces, a minimal chroot,
read-only system Python, dropped capabilities, seccomp, rlimits and a wall-clock
timeout. Never falls back to ordinary subprocess execution. This is defense in
depth, not a substitute for a dedicated VM against hostile kernel exploits.
"""
from __future__ import annotations

import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SandboxResult:
    ok: bool
    output: str
    error: str | None
    duration_s: float
    isolation: str = 'linux-namespaces+seccomp'


def run_python(code: str, *, timeout_s: float = 15, memory_mb: int = 256,
               max_output: int = 65536) -> SandboxResult:
    t0 = time.perf_counter()
    def result(ok, output='', error=None):
        return SandboxResult(ok, output, error, time.perf_counter() - t0)
    try:
        if not isinstance(code, str) or len(code.encode('utf-8')) > 128 * 1024:
            return result(False, error='code must be UTF-8 text under 128 KiB')
    except UnicodeEncodeError:
        return result(False, error='code must be valid UTF-8 text')
    if (not isinstance(max_output, int) or not 1 <= max_output <= 65536 or
            not 0.1 <= timeout_s <= 60 or not 64 <= memory_mb <= 2048):
        return result(False, error='invalid resource limits')
    unshare = shutil.which('unshare')
    if sys.platform != 'linux' or not unshare or not shutil.which('mount'):
        return result(False, error='isolation unavailable: Linux unshare and mount required')
    with tempfile.TemporaryDirectory(prefix='helix-sandbox-') as td:
        root = Path(td) / 'root'
        work = root / 'work'
        work.mkdir(parents=True)
        (root / 'tmp').mkdir()
        (work / 'main.py').write_text(code, encoding='utf-8')
        (work / 'runner.py').write_text(
            "namespace = {'__name__': '__main__', '__file__': '/work/main.py'}\n"
            "with open('/work/main.py', encoding='utf-8') as source:\n"
            "    compiled = compile(source.read(), '/work/main.py', 'exec')\n"
            "exec(compiled, namespace)\n"
            "if 'result' in namespace:\n"
            "    print('__HELIX_RESULT__', repr(namespace['result']))\n",
            encoding='utf-8')
        launcher = Path(__file__).with_name('_sandbox_runner.py').resolve()
        cmd = [unshare, '--user', '--map-root-user', '--mount', '--net', '--pid',
               '--fork', '--kill-child', sys.executable, '-I', str(launcher),
               str(root), str(memory_mb), str(max(1, math.ceil(timeout_s)))]
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            try:
                proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=stdout,
                                        stderr=stderr, start_new_session=True,
                                        env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'})
                try:
                    proc.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait()
                    return result(False, error='timeout')
            except (OSError, ValueError) as exc:
                return result(False, error=f'isolation unavailable: {exc}')
            stdout.seek(0); stderr.seek(0)
            raw_out, raw_err = stdout.read(max_output + 1), stderr.read(max_output + 1)
            out = raw_out[:max_output].decode('utf-8', 'replace')
            err = raw_err[:max_output].decode('utf-8', 'replace')
            if len(raw_out) > max_output:
                out += '\n[output truncated]'
            if len(raw_err) > max_output:
                err += '\n[error truncated]'
            if proc.returncode == 125 or 'HELIX isolation unavailable:' in err or err.startswith('unshare:'):
                return result(False, out.strip(), 'isolation unavailable: ' + err.strip())
            if proc.returncode != 0:
                return result(False, out.strip(), err.strip() or f'process exited {proc.returncode}')
            return result(True, out.strip())
