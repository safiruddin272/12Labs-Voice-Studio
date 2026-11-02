"""Package repair must not guess hardware or replace another ROCm family."""
import re
import sys
from importlib import metadata

from .failures import VerificationFailure


def validate_repair_selection(recipe):
    """Stop ambiguous repairs before downloads; healthy reuse stays unchanged."""
    if sys.platform != "win32":
        return
    from ..gpu_inventory import detect_host_gpus
    from .recipe import detected_rdna2_arch

    amd = [gpu for gpu in detect_host_gpus() if gpu.vendor == "amd"]
    if not amd or any(not re.fullmatch(r"[0-9a-f]{4}", gpu.pci_device_id.lower().removeprefix("0x")) for gpu in amd):
        raise VerificationFailure("unsupported_gpu", "Reliable AMD GPU inventory is required before ROCm repair")
    architecture = detected_rdna2_arch()
    if architecture and architecture not in recipe["gpu_architectures"]:
        raise VerificationFailure("unsupported_gpu", "GPU architecture differs from the selected ROCm recipe")
    family = "10" if "+rocm10." in recipe["torch"]["packages"][0] else "7"
    library = "rocm-sdk-libraries" if family == "10" else "rocm-sdk-libraries-custom"
    for name in ("torch", "rocm", "rocm-sdk-core", library):
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
        match = re.search(r"\+rocm(\d+)\.", version) if name == "torch" else re.match(r"(\d+)\.", version)
        if match and match[1] != family:
            raise VerificationFailure(
                "unsupported_gpu", "Installed ROCm family differs from this recipe; use a fresh runtime instead of automatic repair"
            )
