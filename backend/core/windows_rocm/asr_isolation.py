"""Keep ROCm 10 Torch and CT2's incompatible OpenMP runtimes apart."""
import sys
import weakref
from importlib import metadata, util


def isolation_required():
    """The existing ROCm 7, CUDA and other-platform paths stay unchanged."""
    if sys.platform != "win32":
        return False
    try:
        return "+rocm10." in metadata.version("torch").lower()
    except metadata.PackageNotFoundError:
        return False


def worker_device(parent_device=None):
    """Resolve Windows ROCm 10 without importing Torch in the inference child."""
    if not isolation_required():
        return None
    from core.device_caps import requested_device_override
    from .ct2 import hip_ct2_available

    if parent_device == "cpu" or requested_device_override() == "cpu":
        return "cpu"  # CPU selection must not run GPU verification.
    return "cuda" if hip_ct2_available() else "cpu"


def availability_status():
    """CPU availability must not import CT2 into the Torch parent either."""
    if not isolation_required():
        return None
    if not util.find_spec("faster_whisper"):
        return False, "faster-whisper is not installed"
    from core.device_caps import requested_device_override
    from .ct2 import availability_status as hip_status

    if requested_device_override() != "cpu" and (status := hip_status()) is not None:
        return status
    return True, "Windows ROCm 10 CTranslate2 runs in a separate inference process"


def _proxy(owner):
    proxy = getattr(owner, "_rocm_asr_proxy", None)
    if proxy is None:
        from services.asr_backend import _local_model_source, _ctranslate2_cuda_ok
        from .asr_proxy import RocmASRProxy

        device = "cuda" if _ctranslate2_cuda_ok() else "cpu"
        proxy = RocmASRProxy(_local_model_source(owner._model_name), device)
        owner._rocm_asr_proxy = proxy
        owner._rocm_asr_finalizer = weakref.finalize(owner, proxy.close, True)
        owner.runs_out_of_process = True
    return proxy


def ensure_isolated(owner):
    if not isolation_required():
        return False
    _proxy(owner).ensure_loaded()
    return True


def transcribe_isolated(owner, audio_path, **options):
    if not isolation_required():
        return None
    proxy = _proxy(owner)
    result = proxy.transcribe(audio_path, **options)
    owner._device = proxy.execution_device
    owner._compute_type = proxy.execution_precision
    owner._fallback_reason = proxy.fallback_reason
    owner._fallback_stage = proxy.fallback_stage
    return result


def unload_isolated(owner):
    proxy = getattr(owner, "_rocm_asr_proxy", None)
    if proxy is not None:
        if not proxy.close():
            return True  # An active job still owns the sidecar.
        owner._rocm_asr_finalizer.detach()
        owner._rocm_asr_proxy = None
        owner._device = owner._compute_type = None
    return proxy is not None or isolation_required()


def evidence_loaded(owner):
    proxy = getattr(owner, "_rocm_asr_proxy", None)
    return proxy is not None and proxy.execution_evidence_loaded()
