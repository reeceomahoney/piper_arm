"""Content-addressed cache for precomputed advantage labels.

The cache is keyed by a signature over every input that influences the
computed advantages (dataset/value-network commit SHAs, scalar hyperparams,
schema version).  Cache files are named ``advantage_cache_<sig>.json`` and
mirrored in a HuggingFace Hub ``dataset``-type repo.
"""

import hashlib
import json
import logging
from pathlib import Path


def resolve_hub_sha(repo_id: str, repo_type: str, revision: str | None = None) -> str:
    """Return the commit SHA for a HF Hub repo, or a sentinel if unreachable.

    Sentinel values keep the signature deterministic even when the hub is not
    reachable, at the cost of treating "no-resolve" as a single cache bucket.
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        if repo_type == "dataset":
            info = api.dataset_info(repo_id, revision=revision)
        else:
            info = api.model_info(repo_id, revision=revision)
        if info.sha is not None:
            return info.sha
    except Exception as exc:  # noqa: BLE001
        logging.warning(
            f"Could not resolve {repo_type} SHA for {repo_id}@{revision}: {exc}. "
            "Using repo_id+revision as fallback; cache may be stale if the repo "
            "is mutated in place."
        )
    return f"noresolve:{repo_id}@{revision}"


def compute_signature(
    *,
    schema_version: int,
    dataset_repo_id: str,
    dataset_revision: str | None,
    episodes: list[int] | None,
    value_network_pretrained_path: str,
    c_fail: float,
    num_value_bins: int,
) -> str:
    """Build a deterministic 16-hex-char signature over all cache-affecting inputs."""
    dataset_sha = resolve_hub_sha(dataset_repo_id, "dataset", dataset_revision)
    vn_id = value_network_pretrained_path
    vn_sha = resolve_hub_sha(vn_id, "model")

    sig_dict = {
        "schema_version": schema_version,
        "dataset_repo_id": dataset_repo_id,
        "dataset_revision": dataset_revision,
        "dataset_sha": dataset_sha,
        "episodes": sorted(episodes) if episodes else None,
        "vn_id": vn_id,
        "vn_sha": vn_sha,
        "c_fail": c_fail,
        "num_value_bins": num_value_bins,
    }
    canonical = json.dumps(sig_dict, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def try_download(repo_id: str, signature: str) -> Path | None:
    """Try to fetch a signature-keyed advantage cache from HF Hub."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    filename = f"advantage_cache_{signature}.json"
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        logging.info(f"Downloaded advantage cache from hub: {repo_id}/{filename}")
        return Path(path)
    except (EntryNotFoundError, RepositoryNotFoundError):
        logging.info(
            f"No cached advantages on hub for signature {signature} in {repo_id}"
        )
        return None
    except Exception as exc:  # noqa: BLE001
        logging.warning(f"Failed to download cached advantages from hub: {exc}")
        return None


def upload(local_path: str | Path, repo_id: str, signature: str) -> None:
    """Upload an advantage cache file to HF Hub under a signature-keyed filename."""
    from huggingface_hub import HfApi

    filename = f"advantage_cache_{signature}.json"
    try:
        api = HfApi()
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="dataset",
        )
        logging.info(f"Uploaded advantage cache to hub: {repo_id}/{filename}")
    except Exception as exc:  # noqa: BLE001
        logging.warning(
            f"Failed to upload advantage cache to {repo_id}/{filename}: {exc}. "
            "Continuing with local-only cache."
        )


def save(
    path: str | Path,
    advantage_lookup: dict[int, float],
    episode_lookup: dict[int, int] | None = None,
    metadata: dict | None = None,
) -> None:
    """Save pre-computed advantages to a JSON file for reuse across runs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "advantages": {str(k): v for k, v in advantage_lookup.items()},
        "num_frames": len(advantage_lookup),
        "mean_advantage": sum(advantage_lookup.values())
        / max(1, len(advantage_lookup)),
    }
    if episode_lookup is not None:
        payload["episode_labels"] = {str(k): v for k, v in episode_lookup.items()}
    if metadata:
        payload["metadata"] = metadata
    with open(path, "w") as f:
        json.dump(payload, f)
    logging.info(f"Saved advantage cache ({len(advantage_lookup)} frames) to {path}")


def load(
    path: str | Path,
) -> tuple[dict[int, float], dict[int, int] | None]:
    """Load pre-computed advantages from a JSON cache file."""
    path = Path(path)
    with open(path) as f:
        payload = json.load(f)
    lookup = {int(k): float(v) for k, v in payload["advantages"].items()}
    episode_lookup = None
    if "episode_labels" in payload:
        episode_lookup = {int(k): int(v) for k, v in payload["episode_labels"].items()}
    logging.info(
        f"Loaded advantage cache from {path}: {len(lookup)} frames, "
        f"mean={sum(lookup.values()) / max(1, len(lookup)):.4f}"
    )
    return lookup, episode_lookup
