import dataclasses
import re
import numpy as np

from meta_learn_lib.config import *
from meta_learn_lib.constants import *
from meta_learn_lib.lib_types import *
from meta_learn_lib.logger import Logger


def get_pixel_mean_std(task: Task) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    """Return per-channel (mean, std) used at dataset load time, so we can invert it
    before display. None means no unnormalization needed (raw / binarize / non-image)."""
    match task:
        case MNISTTaskFamily(pixel_transform="normalize"):
            return MNIST_MEAN, MNIST_STD
        case FashionMNISTTaskFamily(pixel_transform="normalize"):
            return FASHION_MNIST_MEAN, FASHION_MNIST_STD
        case CIFAR10TaskFamily(_, _, _):
            return CIFAR10_MEAN, CIFAR10_STD
        case CIFAR100TaskFamily(_, _, _):
            return CIFAR100_MEAN, CIFAR100_STD
        case _:
            return None


def validate_sample_generators(config: GodConfig) -> list[str]:
    errors: list[str] = []
    for i, sg in enumerate(config.sample_generators):
        all_graph_nodes = set(sg.transition_graph.keys()) | set(sg.readout_graph.keys())
        merged_nodes = set(config.nodes) | set(sg.source_nodes)

        for node_name in all_graph_nodes:
            if node_name in sg.source_nodes:
                node = sg.source_nodes[node_name]
                if not isinstance(node, (UnlabeledSource, LabeledSource)):
                    errors.append(
                        f"Sample generator {i}: source_nodes['{node_name}'] must be UnlabeledSource or LabeledSource"
                    )
            elif node_name not in merged_nodes:
                errors.append(f"Sample generator {i}: node '{node_name}' not found in config.nodes or source_nodes")

        for node_name in sg.source_nodes:
            if node_name in config.nodes:
                errors.append(
                    f"Sample generator {i}: source_nodes['{node_name}'] conflicts with existing node in config.nodes"
                )

        for node_name, deps in sg.readout_graph.items():
            for dep in deps:
                if dep not in all_graph_nodes:
                    errors.append(
                        f"Sample generator {i}: readout_graph['{node_name}'] depends on '{dep}' not in the sample graph"
                    )

        for node_name, deps in sg.transition_graph.items():
            for dep in deps:
                if dep not in all_graph_nodes:
                    errors.append(
                        f"Sample generator {i}: transition_graph['{node_name}'] depends on '{dep}' not in the sample graph"
                    )

    return errors


def make_sample_config(config: GodConfig, sg: SampleGeneratorConfig) -> GodConfig:
    identity_grad = GradientConfig(
        method=IdentityLearnerConfig(bptt_config=BPTTConfig(None)),
        add_clip=None,
        scale=1.0,
    )
    no_track = TrackLogs(
        gradient=False,
        hessian_contains_nans=False,
        largest_eigenvalue=False,
        influence_tensor_norm=False,
        immediate_influence_tensor=False,
        largest_jac_eigenvalue=False,
        jacobian=False,
    )

    new_levels = []
    for level in config.levels:
        match sg.input:
            case GaussianSampleInput():
                dataset_source: Task = GaussianNoiseTaskFamily(
                    shape=sg.input_shape,
                    n=sg.num_samples * len(config.levels),
                )
            case DataSampleInput():
                dataset_source = level.dataset_source
            case GridSampleInput(min_value, max_value, n_per_axis):
                dataset_source = GridTaskFamily(
                    dim=sg.input_shape[0],
                    min_value=min_value,
                    max_value=max_value,
                    n_per_axis=n_per_axis,
                )

        new_levels.append(
            dataclasses.replace(
                level,
                objective_fn=NoopObjective(),
                dataset_source=dataset_source,
                dataset=DatasetConfig(
                    num_examples_in_minibatch=sg.num_samples,
                    num_examples_total=sg.num_samples,
                    is_test=False,
                    augment=False,
                ),
                nested=StepConfig(
                    num_steps=1,
                    batch=level.nested.batch,
                    reset_t=None,
                    track_influence_in=level.nested.track_influence_in,
                ),
                learner=LearnConfig(
                    model_learner=identity_grad,
                    optimizer_learner=identity_grad,
                    optimizer={},
                ),
                track_logs=no_track,
                collect_predictions=True,
            )
        )

    return dataclasses.replace(
        config,
        transition_graph=sg.transition_graph,
        readout_graph=sg.readout_graph,
        nodes=config.nodes | sg.source_nodes,
        levels=new_levels,
        sample_generators=[],
        epochs=1,
    )


LEVEL_RE = re.compile(r"level(\d+)/prediction$")


def display_ready(ns: NamedStat, task: Task) -> NamedStat:
    """Invert the dataset-time normalization (if any) and clip to [0, 1] so the
    NamedStat is ready for image display."""
    data = np.asarray(ns.data)
    mean_std = get_pixel_mean_std(task)
    if mean_std is not None:
        mean, std = mean_std
        mean_arr = np.asarray(mean, dtype=data.dtype).reshape(-1, 1, 1)
        std_arr = np.asarray(std, dtype=data.dtype).reshape(-1, 1, 1)
        data = data * std_arr + mean_arr
    data = np.clip(data, 0.0, 1.0)
    return NamedStat(data, ns.axes)


def report_samples(sg: SampleGeneratorConfig, stats: STAT, logger: Logger, config: GodConfig) -> None:
    prediction_stats = {k: v for k, v in stats.items() if k.endswith("/prediction")}
    if not prediction_stats:
        return
    match sg.reporter:
        case ImageReporter(title):
            display_stats = {}
            for k, ns in prediction_stats.items():
                m = LEVEL_RE.search(k)
                level = int(m.group(1))
                display_stats[k] = display_ready(ns, config.levels[level].dataset_source)
            logger.log_image_stats(display_stats, title)
        case PlotReporter(title):
            logger.log_plot_stats(prediction_stats, title)
        case UMAPReporter(title):
            label_stats = {k: v for k, v in stats.items() if k.endswith("/label")}
            logger.log_umap_stats(prediction_stats | label_stats, title)
        case GridReporter(title, rows, cols):
            display_stats = {}
            for k, ns in prediction_stats.items():
                m = LEVEL_RE.search(k)
                level = int(m.group(1))
                display_stats[k] = display_ready(ns, config.levels[level].dataset_source)
            logger.log_grid_stats(display_stats, title, rows, cols)
