import clearml
import json
import os
from dataclasses import dataclass, asdict
from typing import Protocol

import equinox as eqx


@dataclass(frozen=True)
class CheckpointMetadata:
    global_step: int


class CheckpointManager(Protocol):
    def save(self, arr, metadata: CheckpointMetadata) -> None: ...
    def load(self, like) -> tuple | None: ...


class NullCheckpointManager:
    def save(self, arr, metadata: CheckpointMetadata) -> None:
        pass

    def load(self, like) -> tuple | None:
        return None


class FileCheckpointManager:
    def __init__(self, directory: str) -> None:
        self.directory = directory

    def save(self, arr, metadata: CheckpointMetadata) -> None:
        os.makedirs(self.directory, exist_ok=True)
        ckpt_path = os.path.join(self.directory, "checkpoint.eqx")
        meta_path = os.path.join(self.directory, "metadata.json")
        eqx.tree_serialise_leaves(ckpt_path + ".tmp", arr)
        with open(meta_path + ".tmp", "w") as f:
            json.dump(asdict(metadata), f)
        os.replace(ckpt_path + ".tmp", ckpt_path)
        os.replace(meta_path + ".tmp", meta_path)

    def load(self, like) -> tuple | None:
        ckpt_path = os.path.join(self.directory, "checkpoint.eqx")
        meta_path = os.path.join(self.directory, "metadata.json")
        if not os.path.exists(ckpt_path) or not os.path.exists(meta_path):
            return None
        arr = eqx.tree_deserialise_leaves(ckpt_path, like=like)
        with open(meta_path) as f:
            return arr, CheckpointMetadata(**json.load(f))


class ClearMLCheckpointManager:
    """Saves via OutputModel (uploaded to ClearML fileserver) + local cache.

    Supports two resume modes:
    - Automatic: requeued task loads its own latest output model
    - Manual: new task loads a specific model by ID (initial_model_id)
    """

    def __init__(self, task: clearml.Task, directory: str, initial_model_id: str | None = None) -> None:
        self.task = task
        self.file_mgr = FileCheckpointManager(directory)
        self.initial_model_id = initial_model_id
        self.output_model = clearml.OutputModel(task=task, framework="custom")

    def save(self, arr, metadata: CheckpointMetadata) -> None:
        self.file_mgr.save(arr, metadata)

        ckpt_path = os.path.join(self.file_mgr.directory, "checkpoint.eqx")
        meta_path = os.path.join(self.file_mgr.directory, "metadata.json")
        self.output_model.update_weights(
            weights_filename=ckpt_path,
            target_filename="checkpoint.eqx",
            iteration=metadata.global_step,
            auto_delete_file=False,
        )
        self.task.upload_artifact("checkpoint_metadata", meta_path)

    def load(self, like) -> tuple | None:
        # Priority 1: explicit model ID (new run starting from a prior checkpoint)
        if self.initial_model_id is not None:
            arr = self._download_arr(clearml.InputModel(model_id=self.initial_model_id), like)
            return (arr, CheckpointMetadata(global_step=0)) if arr is not None else None

        # Priority 2: local cache (fast path, same node after requeue)
        local = self.file_mgr.load(like)
        if local is not None:
            return local

        # Priority 3: current task's output models (cross-node requeue)
        models = self.task.models.get("output", [])
        if not models:
            return None
        arr = self._download_arr(models[-1], like)
        if arr is None:
            return None
        meta = self._download_metadata()
        return (arr, meta) if meta is not None else None

    def _download_arr(self, model, like):
        local = model.get_local_copy()
        if not local:
            return None
        return eqx.tree_deserialise_leaves(local, like=like)

    def _download_metadata(self) -> CheckpointMetadata | None:
        meta_artifact = self.task.artifacts.get("checkpoint_metadata")
        if meta_artifact is None:
            return None
        local_meta = meta_artifact.get_local_copy()
        if not local_meta:
            return None
        with open(local_meta) as f:
            return CheckpointMetadata(**json.load(f))
