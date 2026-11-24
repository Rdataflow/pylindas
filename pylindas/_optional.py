"""Helper for lazily importing optional dependencies (extras)."""
import importlib


def require_extra(module: str, extra: str):
    """Import ``module`` or raise ImportError pointing at ``pylindas[extra]``."""
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise ImportError(
            f"{module} is required for this function. "
            f"Install with: pip install 'pylindas[{extra}]'"
        ) from e
