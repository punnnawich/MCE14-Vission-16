"""
performance.py — System Performance Optimizer for MCE14 Vision-16
"""

import os
import sys
import cv2

_GPU_AVAILABLE: bool = False
_GPU_DEVICE_NAME = "None"
_timer_active   = False
_mmcss_handle   = None


# ════════════════════════════════════════
# CPU / Process Priority
# ════════════════════════════════════════

def set_high_priority():
    """Process priority → HIGH_PRIORITY_CLASS"""
    try:
        import ctypes; from ctypes import wintypes
        k32 = ctypes.WinDLL('kernel32', use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.SetPriorityClass.restype  = wintypes.BOOL
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00000080)  # HIGH
    except Exception:
        pass


def set_cpu_affinity(cores=None):
    """CPU affinity → all logical cores"""
    try:
        import ctypes; from ctypes import wintypes
        k32 = ctypes.WinDLL('kernel32', use_last_error=True)
        k32.GetCurrentProcess.restype       = wintypes.HANDLE
        k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
        k32.SetProcessAffinityMask.restype  = wintypes.BOOL
        total = os.cpu_count() or 12
        if cores is None:
            cores = list(range(total))
        mask = sum(1 << c for c in cores)
        k32.SetProcessAffinityMask(k32.GetCurrentProcess(), mask)
    except Exception:
        pass


def optimize_threading():
    """OpenCV + NumPy/BLAS thread count → all cores
    NOTE: env vars must also be set BEFORE numpy is first imported
    (see top of vision_pipeline.py for the pre-import block).
    """
    total = os.cpu_count() or 12
    cv2.setNumThreads(total)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = str(total)


def disable_process_power_throttling():
    """Disable Windows EcoQoS / Power Throttling for this process (Win 10 1709+).

    Without this, Windows 11 can silently throttle background Python processes
    even when the power plan is set to High Performance.
    """
    try:
        import ctypes; from ctypes import wintypes

        class PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
            _fields_ = [
                ("Version",     wintypes.DWORD),
                ("ControlMask", wintypes.DWORD),
                ("StateMask",   wintypes.DWORD),
            ]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetProcessInformation.restype  = wintypes.BOOL
        k32.SetProcessInformation.argtypes = [
            ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p, wintypes.DWORD
        ]

        state = PROCESS_POWER_THROTTLING_STATE()
        state.Version     = 1  # PROCESS_POWER_THROTTLING_CURRENT_VERSION
        state.ControlMask = 1  # PROCESS_POWER_THROTTLING_EXECUTION_SPEED
        state.StateMask   = 0  # 0 = disable throttling → full speed

        k32.SetProcessInformation(
            k32.GetCurrentProcess(),
            4,                       # ProcessPowerThrottling
            ctypes.byref(state),
            ctypes.sizeof(state)
        )
    except Exception:
        pass


# ════════════════════════════════════════
# MMCSS — Multimedia Class Scheduler
# ════════════════════════════════════════

def register_mmcss():
    """Register the main thread with MMCSS 'Capture' task.

    MMCSS gives the thread a guaranteed CPU time slice at each scheduler tick,
    which is how Windows media-capture apps achieve consistent 30/60 fps.
    Without it, other processes can steal CPU time mid-frame.
    """
    global _mmcss_handle
    try:
        import ctypes
        avrt = ctypes.WinDLL("avrt")
        avrt.AvSetMmThreadCharacteristicsW.restype  = ctypes.c_void_p
        avrt.AvSetMmThreadCharacteristicsW.argtypes = [
            ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_ulong)
        ]
        avrt.AvSetMmThreadPriority.restype  = ctypes.c_bool
        avrt.AvSetMmThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]

        task_index = ctypes.c_ulong(0)
        handle = avrt.AvSetMmThreadCharacteristicsW("Capture",
                                                    ctypes.byref(task_index))
        if handle:
            avrt.AvSetMmThreadPriority(handle, 2)  # AVRT_PRIORITY_CRITICAL
            _mmcss_handle = handle
    except Exception:
        pass


# ════════════════════════════════════════
# Power Plan
# ════════════════════════════════════════

def disable_power_throttling():
    """Power plan: try Ultimate Performance first, fallback to High Performance."""
    try:
        import subprocess
        # Ultimate Performance (GUID e9a42b02-...) is hidden by default on laptops;
        # duplicating the scheme unlocks it without modifying system policy.
        r = subprocess.run(
            ["powercfg", "/duplicatescheme",
             "e9a42b02-d5df-448d-aa00-03f14749eb61"],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0:
            guid = r.stdout.strip().split()[-1]
            subprocess.run(["powercfg", "/setactive", guid],
                           capture_output=True, timeout=5)
        else:
            subprocess.run(
                ["powercfg", "/setactive",
                 "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"],  # High Performance
                capture_output=True, timeout=5
            )
    except Exception:
        pass


# ════════════════════════════════════════
# I/O Priority (USB OAK-D throughput)
# ════════════════════════════════════════

def set_io_priority():
    """I/O priority → High to improve USB OAK-D frame-transfer throughput."""
    try:
        import ctypes
        ntdll = ctypes.WinDLL("ntdll")
        k32   = ctypes.WinDLL("kernel32")
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        handle = k32.GetCurrentProcess()
        io_priority = ctypes.c_ulong(3)  # IoPriorityHigh = 3
        # NtSetInformationProcess, ProcessIoPriority = 33
        ntdll.NtSetInformationProcess(
            handle, 33,
            ctypes.byref(io_priority), ctypes.sizeof(io_priority)
        )
    except Exception:
        pass


# ════════════════════════════════════════
# Windows Timer Resolution
# ════════════════════════════════════════

def set_timer_resolution():
    """Windows timer → 1ms (from 15.6ms default). Improves sleep/poll precision."""
    global _timer_active
    try:
        import ctypes
        winmm = ctypes.WinDLL('winmm')
        winmm.timeBeginPeriod.argtypes = [ctypes.c_uint]
        winmm.timeBeginPeriod.restype  = ctypes.c_uint
        if winmm.timeBeginPeriod(1) == 0:
            _timer_active = True
    except Exception:
        pass


def restore_timer_resolution():
    global _timer_active
    if not _timer_active:
        return
    try:
        import ctypes
        winmm = ctypes.WinDLL('winmm')
        winmm.timeEndPeriod.argtypes = [ctypes.c_uint]
        winmm.timeEndPeriod.restype  = ctypes.c_uint
        winmm.timeEndPeriod(1)
        _timer_active = False
    except Exception:
        pass


# ════════════════════════════════════════
# Console
# ════════════════════════════════════════

def disable_quickedit():
    """Disable Console QuickEdit — prevents terminal click from freezing the process."""
    if sys.platform != "win32":
        return
    try:
        import ctypes; from ctypes import wintypes
        k32 = ctypes.WinDLL('kernel32', use_last_error=True)
        k32.GetStdHandle.restype  = wintypes.HANDLE
        k32.GetStdHandle.argtypes = [wintypes.DWORD]
        handle = k32.GetStdHandle(ctypes.c_ulong(0xFFFFFFF6))  # STD_INPUT_HANDLE
        mode = wintypes.DWORD()
        k32.GetConsoleMode(handle, ctypes.byref(mode))
        k32.SetConsoleMode(handle, (mode.value & ~0x0040) | 0x0080)
    except Exception:
        pass


# ════════════════════════════════════════
# Garbage Collector
# ════════════════════════════════════════

def disable_gc():
    import gc
    gc.collect()
    gc.disable()


def enable_gc():
    import gc
    gc.enable()
    gc.collect()


# ════════════════════════════════════════
# UDP Socket Buffer
# ════════════════════════════════════════

def optimize_socket(sock, recv_buf=1024*1024, send_buf=1024*1024):
    import socket
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, recv_buf)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, send_buf)
    except Exception:
        pass


# ════════════════════════════════════════
# GPU (OpenCL / T-API)
# ════════════════════════════════════════

def init_gpu():
    global _GPU_AVAILABLE, _GPU_DEVICE_NAME
    if not cv2.ocl.haveOpenCL():
        return False
    cv2.ocl.setUseOpenCL(True)
    if cv2.ocl.useOpenCL():
        try:
            device = cv2.ocl.Device.getDefault()
            _GPU_DEVICE_NAME = device.name()
        except Exception:
            pass
        _GPU_AVAILABLE = True
        return True
    return False


def is_gpu_available():
    return _GPU_AVAILABLE


def gpu_device_name():
    return _GPU_DEVICE_NAME


def to_gpu(mat):
    if _GPU_AVAILABLE and mat is not None:
        return cv2.UMat(mat)
    return mat


def to_cpu(umat):
    if isinstance(umat, cv2.UMat):
        return umat.get()
    return umat


def gpu_cvt_color(frame, code):
    if _GPU_AVAILABLE:
        return cv2.cvtColor(cv2.UMat(frame), code)
    return cv2.cvtColor(frame, code)


def gpu_in_range(frame_hsv, lower, upper):
    if _GPU_AVAILABLE:
        if not isinstance(frame_hsv, cv2.UMat):
            frame_hsv = cv2.UMat(frame_hsv)
        return cv2.inRange(frame_hsv, lower, upper)
    return cv2.inRange(frame_hsv, lower, upper)


def gpu_morphology(mask, kernel, erode_iter=1, dilate_iter=2):
    if _GPU_AVAILABLE:
        if not isinstance(mask, cv2.UMat):
            mask = cv2.UMat(mask)
        mask = cv2.erode(mask, kernel, iterations=erode_iter)
        mask = cv2.dilate(mask, kernel, iterations=dilate_iter)
        return mask
    mask = cv2.erode(mask, kernel, iterations=erode_iter)
    mask = cv2.dilate(mask, kernel, iterations=dilate_iter)
    return mask


def gpu_bitwise_or(mask1, mask2):
    if _GPU_AVAILABLE:
        if not isinstance(mask1, cv2.UMat):
            mask1 = cv2.UMat(mask1)
        if not isinstance(mask2, cv2.UMat):
            mask2 = cv2.UMat(mask2)
        return cv2.bitwise_or(mask1, mask2)
    return cv2.bitwise_or(mask1, mask2)


def gpu_bitwise_and(mask1, mask2):
    if _GPU_AVAILABLE:
        if not isinstance(mask1, cv2.UMat):
            mask1 = cv2.UMat(mask1)
        if not isinstance(mask2, cv2.UMat):
            mask2 = cv2.UMat(mask2)
        return cv2.bitwise_and(mask1, mask2)
    return cv2.bitwise_and(mask1, mask2)


# ════════════════════════════════════════
# Master Initialization
# ════════════════════════════════════════

def init_performance():
    """Call once at startup. Order matters."""
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    disable_quickedit()
    set_high_priority()
    set_cpu_affinity()
    optimize_threading()
    disable_process_power_throttling()   # EcoQoS off
    disable_power_throttling()           # Ultimate / High Performance plan
    set_timer_resolution()               # 1ms timer
    register_mmcss()                     # Guaranteed frame-timing budget
    set_io_priority()                    # USB OAK-D throughput
    return init_gpu()


def cleanup_performance():
    restore_timer_resolution()
    enable_gc()


# ════════════════════════════════════════
# Competition Mode (called after SET ZERO when headless=true)
# ════════════════════════════════════════

def competition_mode():
    """Kill non-essential processes/services and escalate to REALTIME priority."""
    import subprocess

    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    kill_list = [
        "OneDrive.exe", "FileSyncHelper.exe",
        "Teams.exe", "ms-teams.exe", "Skype.exe", "SkypeApp.exe",
        "Slack.exe", "Discord.exe", "Zoom.exe",
        "msedge.exe", "chrome.exe", "firefox.exe", "brave.exe",
        "PhoneExperienceHost.exe", "Widgets.exe", "WidgetService.exe",
        "GameBar.exe", "GameBarPresenceWriter.exe",
        "XboxPcAppFT.exe", "XboxGameBarWidget.exe",
        "HxTsr.exe", "HxOutlook.exe", "YourPhone.exe",
        "AdobeIPCBroker.exe", "Adobe Desktop Service.exe",
        "CCXProcess.exe", "Creative Cloud.exe",
        "PerfWatson2.exe",
        "HPAudioSwitch.exe", "HPCommRecovery.exe",
    ]
    for proc in kill_list:
        try:
            subprocess.run(["taskkill", "/F", "/IM", proc],
                           capture_output=True, timeout=3)
        except Exception:
            pass

    for svc in ("WSearch", "SysMain", "DiagTrack",
                "WMPNetworkSvc", "MapsBroker", "lfsvc",
                "TabletInputService", "WerSvc"):
        try:
            subprocess.run(["net", "stop", svc],
                           capture_output=True, timeout=10)
        except Exception:
            pass

    # Disable transparency effects (reduces DWM GPU load)
    try:
        subprocess.run([
            "reg", "add",
            r"HKCU\Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
            "/v", "EnableTransparency", "/t", "REG_DWORD", "/d", "0", "/f"
        ], capture_output=True, timeout=5)
    except Exception:
        pass

    # Exclude project directory from Defender real-time scanning
    try:
        project_dir = os.path.dirname(os.path.abspath(__file__))
        subprocess.run(
            ["powershell", "-Command",
             f"Add-MpPreference -ExclusionPath '{project_dir}'"],
            capture_output=True, timeout=10
        )
    except Exception:
        pass

    # Escalate to REALTIME_PRIORITY_CLASS (requires Admin)
    try:
        import ctypes; from ctypes import wintypes
        k32 = ctypes.WinDLL('kernel32', use_last_error=True)
        k32.GetCurrentProcess.restype  = wintypes.HANDLE
        k32.SetPriorityClass.argtypes  = [wintypes.HANDLE, wintypes.DWORD]
        k32.SetPriorityClass.restype   = wintypes.BOOL
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00000100)  # REALTIME
    except Exception:
        pass
