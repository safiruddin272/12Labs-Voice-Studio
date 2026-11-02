"""Whisper requests must respect the loaded checkpoint's actual language support."""


def dependency_status():
    """Catalogue threads must not execute Transformers' heavy lazy imports."""
    from services.performance_readiness import package_available

    installed = package_available("transformers")
    return (True, "ready") if installed else (False, "transformers is not installed")


def english_only(pipe):
    """Recognize an English-only checkpoint from its loaded generation config."""
    model = getattr(pipe, "model", None)
    configuration = getattr(model, "generation_config", None)
    return getattr(configuration, "is_multilingual", None) is False


def generation_options(pipe, options):
    """English-only generation rejects even redundant English/task arguments."""
    if not english_only(pipe):
        return options
    if options.get("task", "transcribe") != "transcribe":
        raise ValueError("English-only Whisper checkpoints cannot translate; select a multilingual checkpoint")
    language = options.get("language")
    if language is not None and str(language).strip().lower() not in {"en", "english", "<|en|>"}:
        raise ValueError("English-only Whisper checkpoints only transcribe English; select a multilingual checkpoint")
    return {key: value for key, value in options.items() if key not in {"language", "task"}}
