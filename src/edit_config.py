import argparse
import copy
import json
import tempfile
from pathlib import Path

import clearml
from clearml import InputModel, OutputModel, Task

PROJECT = "oho"


def parse_value(raw: str):
    if raw.startswith("[") or raw.startswith("{") or raw.startswith('"'):
        return json.loads(raw)
    if raw == "true":
        return True
    if raw == "false":
        return False
    if raw == "null":
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def apply_edit(cfg: dict, dotted_path: str, value) -> None:
    keys = dotted_path.split("/")
    cursor = cfg
    for k in keys[:-1]:
        cursor = cursor[k]
    cursor[keys[-1]] = value


def main():
    parser = argparse.ArgumentParser(
        description="Fetch a config from ClearML, apply field edits, upload as a new model."
    )
    parser.add_argument("--base-id", required=True, help="ClearML model ID of the base config to edit")
    parser.add_argument("--name", required=True, help="Name for the new config")
    parser.add_argument(
        "--set",
        dest="edits",
        action="append",
        default=[],
        metavar="DOTTED/PATH=VALUE",
        help="Edit one field, e.g. 'hyperparameters/meta2_sgd1_lr/value=1e-3'. May be repeated.",
    )
    args = parser.parse_args()

    edits: dict = {}
    for spec in args.edits:
        if "=" not in spec:
            raise SystemExit(f"--set expects DOTTED/PATH=VALUE, got: {spec}")
        path, raw = spec.split("=", 1)
        edits[path] = parse_value(raw)

    src = InputModel(model_id=args.base_id)
    local = src.get_local_copy()
    with open(local) as f:
        cfg = json.load(f)

    for path, value in edits.items():
        apply_edit(cfg, path, value)
        print(f"  set {path} = {value!r}")

    task = Task.init(
        project_name=PROJECT,
        task_name=f"upload_{args.name}",
        task_type=clearml.TaskTypes.data_processing,
        output_uri=True,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", prefix=f"{args.name}_", delete=False) as f:
        json.dump(cfg, f, indent=2, default=lambda o: sorted(o, key=str) if isinstance(o, (set, frozenset)) else o)
        tmp_path = f.name

    output_model = OutputModel(task=task, name=args.name, tags=["config"])
    output_model.update_weights(weights_filename=tmp_path)
    output_model.publish()

    Path(tmp_path).unlink(missing_ok=True)

    print(f"Uploaded {args.name} -> model_id={output_model.id}")
    task.close()


if __name__ == "__main__":
    main()
