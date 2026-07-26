"""Content-addressed local cache for parsed YAML config.

Parsing the Tensix config YAML (``tensix_instructions.yaml``,
``tensix_backend_cfg.yaml``) at device construction costs a couple of
seconds — a large fraction of wall-clock for short runs. The parsed
result is a plain dict/list tree, so we pickle it to a per-user cache
directory keyed by a hash of the source bytes.

The cache is local (never in the repo) and self-invalidating: the source
content hash is part of the cache filename, so editing a YAML file yields
a fresh key (cache miss → reparse → recache) and never returns stale
data. Caching is best-effort — any I/O failure falls back to parsing.
"""

import hashlib
import os
import pickle

import yaml

_CACHE_SUBDIR = os.path.join("tt-sim", "yaml")


def _cache_dir():
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    path = os.path.join(base, _CACHE_SUBDIR)
    os.makedirs(path, exist_ok=True)
    return path


def load_yaml_cached(traversable, name):
    """Parse ``traversable`` (an ``importlib.resources`` resource) as YAML,
    using a local pickle cache keyed by the source content hash.

    ``name`` is a short, stable label used in the cache filename.
    """
    raw = traversable.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()[:16]

    try:
        cache_dir = _cache_dir()
    except OSError:
        return yaml.safe_load(raw)

    cache_path = os.path.join(cache_dir, f"{name}.{digest}.pkl")
    try:
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    except (OSError, pickle.UnpicklingError, EOFError, ValueError):
        pass

    data = yaml.safe_load(raw)

    # Drop stale versions of this file so the cache doesn't grow unbounded
    # as the source YAML evolves.
    try:
        prefix = f"{name}."
        for existing in os.listdir(cache_dir):
            if existing.startswith(prefix) and existing.endswith(".pkl"):
                try:
                    os.remove(os.path.join(cache_dir, existing))
                except OSError:
                    pass
    except OSError:
        pass

    tmp_path = f"{cache_path}.tmp{os.getpid()}"
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, cache_path)
    except OSError:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return data
