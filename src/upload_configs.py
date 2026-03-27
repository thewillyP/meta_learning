"""Utility for uploading GodConfig instances to ClearML Model Registry.

Not meant to be run directly — call upload_config() from configs.py instead.
"""

import json
import tempfile
from pathlib import Path

import clearml
from clearml import OutputModel, Task

from main_clearml import make_converter

PROJECT = "oho"


def upload_config(name: str, config):
    converter = make_converter()

    task = Task.init(
        project_name=PROJECT,
        task_name=f"upload_{name}",
        task_type=clearml.TaskTypes.data_processing,
        output_uri=True,
    )

    data = converter.unstructure(config)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix=f"{name}_", delete=False
    ) as f:
        json.dump(data, f, indent=2, default=lambda o: sorted(o, key=str) if isinstance(o, (set, frozenset)) else o)
        tmp_path = f.name

    output_model = OutputModel(task=task, name=name, tags=["config"])
    output_model.update_weights(weights_filename=tmp_path)
    output_model.publish()

    Path(tmp_path).unlink(missing_ok=True)

    print(f"Uploaded {name} -> model_id={output_model.id}")
    task.close()
    return output_model.id
