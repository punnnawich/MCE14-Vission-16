"""
performance.py — System Performance Optimizer
ตั้งค่า CPU priority, GPU acceleration (OpenCL), และ thread affinity
เพื่อให้ vision pipeline ใช้ทรัพยากรเต็มประสิทธิภาพ

เรียก init_performance() ที่จุดเริ่มต้นของ vision_pipeline.py
"""

import os
import sys
import cv2
import numpy as np

# ════════════════════════════════════════
# CPU Optimization
# ════════════════════════════════════════

def set_high_priority():
    """
    ตั้ง process priority เป็น HIGH_PRIORITY_CLASS บน Windows
    ทำให้ OS จัดสรร CPU time ให้ vision pipeline มากกว่าโปรเซสอื่น
    """
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

        # Define proper argument/return types for Windows API
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.SetPriorityClass.restype = wintypes.BOOL

        handle = kernel32.GetCurrentProcess()
        # HIGH_PRIORITY_CLASS = 0x00000080
        # REALTIME_PRIORITY_CLASS = 0x00000100 (ไม่แนะนำ — อาจทำให้ OS ค้าง)
        result = kernel32.SetPriorityClass(handle, 0x00000080)
        if result:
            print("[Perf] ✅ Process priority → HIGH_PRIORITY_CLASS")
        else:
            err = ctypes.get_last_error()
            print(f"[Perf] ⚠️ Failed to set HIGH priority (error={err})")
        return bool(result)
    except Exception as e:
        print(f"[Perf] ⚠️ Priority setting skipped: {e}")
        return False


def set_cpu_affinity(cores=None):
    """
    ล็อก process ลง CPU cores เฉพาะ เพื่อลด context switching และ cache miss
    Default: ใช้ทุก core เพื่อให้ OS จัดสรรเอง (ไม่จำกัด)
    """
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

        # Define proper argument/return types
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
        kernel32.SetProcessAffinityMask.restype = wintypes.BOOL

        total_cores = os.cpu_count() or 12
        if cores is None:
            # ใช้ทุก core ที่มี — ให้ OS scheduler จัดสรรเต็มที่
            cores = list(range(total_cores))

        mask = 0
        for c in cores:
            mask |= (1 << c)

        handle = kernel32.GetCurrentProcess()
        result = kernel32.SetProcessAffinityMask(handle, mask)
        if result:
            print(f"[Perf] ✅ CPU affinity → cores {cores} (mask=0x{mask:X})")
        else:
            err = ctypes.get_last_error()
            print(f"[Perf] ⚠️ Failed to set CPU affinity (error={err})")
        return bool(result)
    except Exception as e:
        print(f"[Perf] ⚠️ CPU affinity skipped: {e}")
        return False


def optimize_threading():
    """
    ตั้งค่า thread count สำหรับ OpenCV, NumPy, OpenBLAS
    ให้ใช้ cores ที่มีอย่างเต็มที่
    """
    total_cores = os.cpu_count() or 12
    # OpenCV — ใช้ทุก core สำหรับ parallel operations (TBB/OpenMP)
    cv2.setNumThreads(total_cores)
    print(f"[Perf] ✅ OpenCV threads → {cv2.getNumThreads()}")

    # NumPy/OpenBLAS — ตั้งจำนวน threads
    for env_var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ[env_var] = str(total_cores)
    print(f"[Perf] ✅ NumPy/BLAS threads → {total_cores}")


# ════════════════════════════════════════
# GPU Optimization (OpenCL)
# ════════════════════════════════════════

_GPU_AVAILABLE = False
_GPU_DEVICE_NAME = "None"


def init_gpu():
    """
    เปิดใช้งาน GPU ผ่าน OpenCL (T-API / Transparent API ของ OpenCV)
    ทำให้ cv2.UMat operations ทำงานบน GPU อัตโนมัติ
    """
    global _GPU_AVAILABLE, _GPU_DEVICE_NAME

    if not cv2.ocl.haveOpenCL():
        print("[Perf] ⚠️ OpenCL not available — GPU acceleration disabled")
        return False

    cv2.ocl.setUseOpenCL(True)

    if cv2.ocl.useOpenCL():
        try:
            device = cv2.ocl.Device.getDefault()
            _GPU_DEVICE_NAME = device.name()
            _GPU_AVAILABLE = True
            print(f"[Perf] ✅ OpenCL GPU → {_GPU_DEVICE_NAME}")
            print(f"[Perf]    Type: {'GPU' if device.type() == cv2.ocl.Device_TYPE_GPU else 'Other'}")
            print(f"[Perf]    Compute Units: {device.maxComputeUnits()}")
            print(f"[Perf]    Global Memory: {device.globalMemSize() // (1024*1024)} MB")
            return True
        except Exception as e:
            print(f"[Perf] ⚠️ OpenCL device query failed: {e}")
            _GPU_AVAILABLE = True  # Still try to use it
            return True
    else:
        print("[Perf] ⚠️ OpenCL enable failed")
        return False


def is_gpu_available():
    """Check if GPU acceleration is active."""
    return _GPU_AVAILABLE


def gpu_device_name():
    """Get the name of the active GPU device."""
    return _GPU_DEVICE_NAME


# ════════════════════════════════════════
# GPU-Accelerated OpenCV Operations
# ════════════════════════════════════════
# UMat (Unified Memory) ทำให้ OpenCV ย้าย operations ไปทำงานบน GPU อัตโนมัติ
# ถ้า GPU ไม่พร้อม จะ fallback เป็น CPU ปกติโดยไม่ error

def to_gpu(mat):
    """Upload cv2.Mat → cv2.UMat (GPU memory) ถ้า GPU พร้อม"""
    if _GPU_AVAILABLE and mat is not None:
        return cv2.UMat(mat)
    return mat


def to_cpu(umat):
    """Download cv2.UMat → cv2.Mat (CPU memory)"""
    if isinstance(umat, cv2.UMat):
        return umat.get()
    return umat


def gpu_cvt_color(frame, code):
    """GPU-accelerated cv2.cvtColor"""
    if _GPU_AVAILABLE:
        return cv2.cvtColor(cv2.UMat(frame), code)
    return cv2.cvtColor(frame, code)


def gpu_in_range(frame_hsv, lower, upper):
    """GPU-accelerated cv2.inRange"""
    if _GPU_AVAILABLE:
        if not isinstance(frame_hsv, cv2.UMat):
            frame_hsv = cv2.UMat(frame_hsv)
        return cv2.inRange(frame_hsv, lower, upper)
    return cv2.inRange(frame_hsv, lower, upper)


def gpu_morphology(mask, kernel, erode_iter=1, dilate_iter=2):
    """GPU-accelerated erode + dilate"""
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
    """GPU-accelerated cv2.bitwise_or"""
    if _GPU_AVAILABLE:
        if not isinstance(mask1, cv2.UMat):
            mask1 = cv2.UMat(mask1)
        if not isinstance(mask2, cv2.UMat):
            mask2 = cv2.UMat(mask2)
        return cv2.bitwise_or(mask1, mask2)
    return cv2.bitwise_or(mask1, mask2)


def gpu_bitwise_and(mask1, mask2):
    """GPU-accelerated cv2.bitwise_and"""
    if _GPU_AVAILABLE:
        if not isinstance(mask1, cv2.UMat):
            mask1 = cv2.UMat(mask1)
        if not isinstance(mask2, cv2.UMat):
            mask2 = cv2.UMat(mask2)
        return cv2.bitwise_and(mask1, mask2)
    return cv2.bitwise_and(mask1, mask2)


# ════════════════════════════════════════
# Windows Power Management
# ════════════════════════════════════════

def disable_power_throttling():
    """
    ป้องกัน Windows จาก throttle CPU/GPU เพื่อประหยัดพลังงาน
    ตั้ง Power Plan เป็น High Performance ผ่าน subprocess
    """
    try:
        import subprocess
        # Get current power scheme
        result = subprocess.run(
            ["powercfg", "/getactivescheme"],
            capture_output=True, text=True, timeout=5
        )
        current = result.stdout.strip()
        print(f"[Perf] Current power plan: {current}")

        # Set High Performance power scheme GUID
        # 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c = High Performance
        subprocess.run(
            ["powercfg", "/setactive", "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"],
            capture_output=True, timeout=5
        )
        print("[Perf] ✅ Power plan → High Performance")
        return True
    except Exception as e:
        print(f"[Perf] ⚠️ Power plan change skipped: {e}")
        return False


# ════════════════════════════════════════
# Windows Timer Resolution
# ════════════════════════════════════════

_timer_active = False

def set_timer_resolution():
    """
    ตั้ง Windows timer resolution เป็น 1ms (จาก default 15.6ms)
    ส่งผลต่อ: time.sleep(), threading.Timer, UDP polling, scheduler accuracy
    ต้องเรียก restore_timer_resolution() ตอนปิดโปรแกรม
    """
    global _timer_active
    try:
        import ctypes
        winmm = ctypes.WinDLL('winmm')
        winmm.timeBeginPeriod.argtypes = [ctypes.c_uint]
        winmm.timeBeginPeriod.restype = ctypes.c_uint
        result = winmm.timeBeginPeriod(1)
        if result == 0:  # TIMERR_NOERROR
            _timer_active = True
            print("[Perf] ✅ Timer resolution → 1ms (was 15.6ms)")
            return True
        else:
            print(f"[Perf] ⚠️ Timer resolution failed (result={result})")
            return False
    except Exception as e:
        print(f"[Perf] ⚠️ Timer resolution skipped: {e}")
        return False


def restore_timer_resolution():
    """
    คืนค่า Windows timer resolution กลับเป็น default
    เรียกตอนปิดโปรแกรม (ใน finally block)
    """
    global _timer_active
    if not _timer_active:
        return
    try:
        import ctypes
        winmm = ctypes.WinDLL('winmm')
        winmm.timeEndPeriod.argtypes = [ctypes.c_uint]
        winmm.timeEndPeriod.restype = ctypes.c_uint
        winmm.timeEndPeriod(1)
        _timer_active = False
    except Exception:
        pass


# ════════════════════════════════════════
# Windows Console QuickEdit Mode
# ════════════════════════════════════════

def disable_quickedit():
    """
    ปิด Windows Console QuickEdit Mode
    
    ปัญหา: เมื่อ QuickEdit เปิด (default) การคลิกที่หน้าต่าง Terminal
    จะทำให้ Console เข้า "Selection Mode" ซึ่ง **หยุดทั้ง process**
    จนกว่าจะกด Enter/Escape — ทำให้ vision pipeline ค้าง (Not Responding)
    
    แก้: ปิด ENABLE_QUICK_EDIT_MODE (0x0040) ผ่าน Windows API
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

        # STD_INPUT_HANDLE = -10
        kernel32.GetStdHandle.restype = wintypes.HANDLE
        kernel32.GetStdHandle.argtypes = [wintypes.DWORD]
        handle = kernel32.GetStdHandle(ctypes.c_ulong(-10 & 0xFFFFFFFF))

        # Get current console mode
        mode = wintypes.DWORD()
        kernel32.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.GetConsoleMode.restype = wintypes.BOOL
        kernel32.GetConsoleMode(handle, ctypes.byref(mode))

        # Disable ENABLE_QUICK_EDIT_MODE (0x0040)
        # Enable ENABLE_EXTENDED_FLAGS (0x0080) — required for QuickEdit change to take effect
        ENABLE_QUICK_EDIT_MODE = 0x0040
        ENABLE_EXTENDED_FLAGS  = 0x0080
        new_mode = (mode.value & ~ENABLE_QUICK_EDIT_MODE) | ENABLE_EXTENDED_FLAGS

        kernel32.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.SetConsoleMode.restype = wintypes.BOOL
        result = kernel32.SetConsoleMode(handle, new_mode)

        if result:
            print("[Perf] ✅ Console QuickEdit → DISABLED (prevents click-freeze)")
        else:
            err = ctypes.get_last_error()
            print(f"[Perf] ⚠️ Failed to disable QuickEdit (error={err})")
        return bool(result)
    except Exception as e:
        print(f"[Perf] ⚠️ QuickEdit disable skipped: {e}")
        return False


# ════════════════════════════════════════
# Garbage Collector Control
# ════════════════════════════════════════

def disable_gc():
    """
    ปิด Garbage Collector ระหว่าง hot loop
    ป้องกัน random pause 1-5ms จาก GC sweep
    เรียก gc.collect() ก่อนปิดเพื่อ clean up ค้างอยู่
    """
    import gc
    gc.collect()   # Clean up ก่อน
    gc.disable()
    print("[Perf] ✅ Garbage Collector → DISABLED (hot loop mode)")


def enable_gc():
    """คืนค่า Garbage Collector กลับเป็นปกติ (เรียกตอนปิดโปรแกรม)"""
    import gc
    gc.enable()
    gc.collect()


# ════════════════════════════════════════
# UDP Socket Buffer
# ════════════════════════════════════════

def optimize_socket(sock, recv_buf=1024*1024, send_buf=1024*1024):
    """
    เพิ่ม UDP socket buffer เป็น 1MB (จาก default ~64KB)
    ลด packet drop ตอน CPU busy
    """
    import socket
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, recv_buf)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, send_buf)
        actual_recv = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        actual_send = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
        print(f"[Perf] ✅ UDP buffer → recv={actual_recv//1024}KB send={actual_send//1024}KB")
        return True
    except Exception as e:
        print(f"[Perf] ⚠️ UDP buffer optimization skipped: {e}")
        return False


# ════════════════════════════════════════
# Master Initialization
# ════════════════════════════════════════

def init_performance():
    """
    เรียกครั้งเดียวตอนเริ่มโปรแกรม — ตั้งค่าทุกอย่างให้พร้อม:
      1. Process priority → HIGH
      2. CPU affinity → all cores
      3. OpenCV/NumPy threading → max cores
      4. GPU (OpenCL) → enabled
      5. Power plan → High Performance
      6. Timer resolution → 1ms
      7. Console QuickEdit → DISABLED
    """
    # Force UTF-8 output for Windows console (prevent emoji encode errors)
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("\n" + "=" * 55)
    print("  ⚡ Performance Optimizer — MCE14 Vission-16")
    print("=" * 55)

    # CRITICAL: Disable QuickEdit FIRST to prevent terminal click-freeze
    disable_quickedit()
    set_high_priority()
    set_cpu_affinity()
    optimize_threading()
    gpu_ok = init_gpu()
    disable_power_throttling()
    set_timer_resolution()

    print("-" * 55)
    print(f"  CPU Cores: {os.cpu_count()}")
    print(f"  GPU: {_GPU_DEVICE_NAME}")
    print(f"  OpenCV: {cv2.__version__} (threads={cv2.getNumThreads()})")
    print(f"  NumPy: {np.__version__} (OpenBLAS)")
    print(f"  GPU Accel: {'ON' if gpu_ok else 'OFF'}")
    print(f"  Timer Res: {'1ms' if _timer_active else '15.6ms (default)'}")
    print("=" * 55 + "\n")

    return gpu_ok


def cleanup_performance():
    """เรียกตอนปิดโปรแกรม — คืนค่าทุกอย่างกลับ"""
    restore_timer_resolution()
    enable_gc()

