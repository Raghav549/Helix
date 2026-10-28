"""Internal Linux namespace launcher. Do not call directly with untrusted paths.

The parent creates the workspace. This process enters a new user/mount/network/
PID namespace before it is started, bind-mounts only the system Python runtime
read-only, chroots, drops capabilities, installs a seccomp denylist, and execs
Python. Isolation is deliberately fail-closed when any step is unavailable.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import errno
import os
import resource
import subprocess
import sys


def mount_readonly(source: str, dest: str) -> None:
    if os.path.isdir(source):
        os.makedirs(dest, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        open(dest, 'a').close()
    subprocess.run(['mount', '--bind', source, dest], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    subprocess.run(['mount', '-o', 'remount,bind,ro', dest], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def restrict_syscalls(lib) -> None:
    # libseccomp is loaded before chroot; the filter survives exec. Besides the
    # network namespace this denies socket calls, child processes, and attacks
    # on namespaces / tracing. It is defense in depth, not a VM boundary.
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                    ctypes.c_int, ctypes.c_uint]
    lib.seccomp_rule_add.restype = ctypes.c_int
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_load.restype = ctypes.c_int
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    ctx = lib.seccomp_init(0x7fff0000)  # SCMP_ACT_ALLOW
    if not ctx:
        raise RuntimeError('seccomp_init failed')
    denied = ('socket socketpair connect bind listen accept accept4 sendto sendmsg '
              'sendmmsg recvfrom recvmsg recvmmsg mount umount2 pivot_root '
              'chroot unshare setns ptrace process_vm_readv process_vm_writev '
              'bpf perf_event_open keyctl add_key request_key open_by_handle_at '
              'kexec_load kexec_file_load reboot fork vfork clone clone3').split()
    for name in denied:
        number = lib.seccomp_syscall_resolve_name(name.encode())
        if number >= 0 and lib.seccomp_rule_add(ctx, 0x00050000 | errno.EPERM,
                                                number, 0) != 0:
            raise RuntimeError(f'cannot restrict syscall {name}')
    if lib.seccomp_load(ctx) != 0:
        raise RuntimeError('seccomp_load failed')
    lib.seccomp_release(ctx)


def drop_caps() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    # Drop capability bounding set while CAP_SETPCAP is still effective.
    for cap in range(64):
        libc.prctl(24, cap, 0, 0, 0)  # PR_CAPBSET_DROP; unsupported caps ignored

    class Header(ctypes.Structure):
        _fields_ = [('version', ctypes.c_uint32), ('pid', ctypes.c_int)]
    class Data(ctypes.Structure):
        _fields_ = [('effective', ctypes.c_uint32), ('permitted', ctypes.c_uint32),
                    ('inheritable', ctypes.c_uint32)]
    header = Header(0x20080522, 0)  # Linux capability v3
    data = (Data * 2)()
    if libc.capset(ctypes.byref(header), ctypes.byref(data)) != 0:
        raise OSError(ctypes.get_errno(), 'capset failed')
    if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        raise OSError(ctypes.get_errno(), 'prctl(NO_NEW_PRIVS) failed')


def main() -> None:
    root, memory_mb, cpu_s = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    # Make mount propagation private; never affect the host's mount table.
    subprocess.run(['mount', '--make-rprivate', '/'], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    # Use the system interpreter, never the parent venv or user's home.
    binary = os.path.realpath('/usr/bin/python3')
    if not os.path.isfile(binary):
        raise RuntimeError('system Python not found at /usr/bin/python3')
    mount_readonly(binary, root + '/usr/bin/python3')
    version = f'python{sys.version_info.major}.{sys.version_info.minor}'
    stdlib = '/usr/lib/' + version
    if not os.path.isdir(stdlib):
        raise RuntimeError(f'system standard library not found: {stdlib}')
    mount_readonly(stdlib, root + '/usr/lib/' + version)
    for path in ('/lib', '/lib64'):
        if os.path.exists(path):
            mount_readonly(path, root + path)
    # libseccomp must be resolved before chroot. Configure only after mounts.
    # CDLL loads the actual library now; ctypes reuses it after chroot.
    sec = ctypes.util.find_library('seccomp')
    if not sec:
        raise RuntimeError('libseccomp unavailable')
    seccomp = ctypes.CDLL(sec, use_errno=True)
    os.chroot(root)
    os.chdir('/work')
    limit = memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    drop_caps()
    restrict_syscalls(seccomp)
    os.execve('/usr/bin/python3', ['/usr/bin/python3', '-I', '-B', '/work/runner.py'],
              {'PATH': '/usr/bin', 'HOME': '/work', 'LANG': 'C.UTF-8'})


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        print(f'HELIX isolation unavailable: {type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(125)
