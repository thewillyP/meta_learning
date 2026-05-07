import dataclasses

from meta_learn_lib.config import *
from meta_learn_lib.lib_types import STAT
from meta_learn_lib.logger import Logger


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
    identity_learner = LearnConfig(
        model_learner=identity_grad,
        optimizer_learner=identity_grad,
        optimizer={},
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
                dataset_source: Task = GaussianNoiseTaskFamily(shape=sg.input_shape, n=sg.num_samples)
            case DataSampleInput():
                dataset_source = level.dataset_source

        new_levels.append(
            dataclasses.replace(
                level,
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
                learner=identity_learner,
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


def report_samples(sg: SampleGeneratorConfig, stats: STAT, logger: Logger) -> None:
    prediction_stats = {k: v for k, v in stats.items() if k.endswith("/prediction")}
    if not prediction_stats:
        return
    match sg.reporter:
        case ImageReporter(title):
            logger.log_image_stats(prediction_stats, title)
        case PlotReporter(title):
            logger.log_plot_stats(prediction_stats, title)
