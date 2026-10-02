"""Host-side emulation of the Raspberry Pi 5: core pinning, shared-core load and a CPU slowdown throttle."""
from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass

import psutil

_THROTTLE_PERIOD_S = 0.03


@dataclass(frozen=True)
class DeviceProfile:
    name: str = "rpi5-8gb"
    cores: int = 4
    ram_gb: float = 8.0
    os_reserved_gb: float = 1.0  # STATUS: ESTIMATE
    background_busy_cores: int = 2  # STATUS: ESTIMATE (ASR and MT sharing the CPU)
    cpu_scale: float = 3.0  # STATUS: ESTIMATE (host-to-Pi slowdown; calibrate on the real board)

    @property
    def ram_budget_mb(self) -> float:
        return (self.ram_gb - self.os_reserved_gb) * 1024


def pick_cpus(count: int) -> list:
    """Choose logical CPUs on distinct physical cores where the host has hyper-threading."""
    logical = psutil.cpu_count(logical=True) or 1
    physical = psutil.cpu_count(logical=False) or logical
    count = max(1, min(count, logical))
    stride = max(1, logical // physical)
    cpus = [i * stride for i in range(count) if i * stride < logical]
    return cpus if len(cpus) == count else list(range(count))


def _burn(cpu: int) -> None:
    psutil.Process().cpu_affinity([cpu])
    while True:
        pass


def _throttle(pid: int, scale: float) -> None:
    """Suspend and resume the target so it only runs 1/scale of the time."""
    proc = psutil.Process(pid)
    run = _THROTTLE_PERIOD_S / scale
    try:
        while True:
            time.sleep(run)
            proc.suspend()
            time.sleep(_THROTTLE_PERIOD_S - run)
            proc.resume()
    except psutil.NoSuchProcess:
        pass


class EmulatedDevice:
    """Constrains one worker process to look like the target board; call detach() to undo."""

    def __init__(self, profile: DeviceProfile):
        self.profile = profile
        self._pid = None
        self._burners = []
        self._throttle = None

    def attach(self, pid: int) -> list:
        self._pid = pid
        cpus = pick_cpus(self.profile.cores)
        psutil.Process(pid).cpu_affinity(cpus)
        ctx = mp.get_context("spawn")
        busy = min(self.profile.background_busy_cores, max(0, len(cpus) - 1))
        for cpu in cpus[:busy]:
            burner = ctx.Process(target=_burn, args=(cpu,), daemon=True)
            burner.start()
            self._burners.append(burner)
        if self.profile.cpu_scale > 1.0:
            self._throttle = ctx.Process(
                target=_throttle, args=(pid, self.profile.cpu_scale), daemon=True
            )
            self._throttle.start()
        return cpus

    def detach(self) -> None:
        if self._throttle is not None:
            self._throttle.terminate()
            self._throttle.join(2)
            self._throttle = None
            try:
                psutil.Process(self._pid).resume()
            except psutil.NoSuchProcess:
                pass
        for burner in self._burners:
            burner.terminate()
            burner.join(2)
        self._burners = []
