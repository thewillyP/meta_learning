import dataclasses

from meta_learn_lib.config import (
    DataSampleInput,
    GaussianSampleInput,
    GaussianNoiseTaskFamily,
    GodConfig,
    GradientConfig,
    IdentityLearnerConfig,
    ImageReporter,
    LearnConfig,
    SampleGeneratorConfig,
    TrackLogs,
)
from meta_learn_lib.logger import ThreadedScalarLogger
from meta_learn_lib.lib_types import STAT


def validate_sample_generators(config: GodConfig) -> list[str]:
    errors: list[str] = []
    for i, sg in enumerate(config.sample_generators):
        all_graph_nodes = set(sg.transition_graph.keys()) | set(sg.readout_graph.keys())

        for node_name in all_graph_nodes:
            if node_name not in config.nodes:
                errors.append(f"Sample generator {i}: node '{node_name}' not found in config.nodes")

        for node_name, deps in sg.readout_graph.items():
            for dep in deps:
                if dep not in all_graph_nodes:
                    errors.append(
                        f"Sample generator {i}: readout_graph['{node_name}'] depends on '{dep}' which is not in the sample graph"
                    )

        for node_name, deps in sg.transition_graph.items():
            for dep in deps:
                if dep not in all_graph_nodes:
                    errors.append(
                        f"Sample generator {i}: transition_graph['{node_name}'] depends on '{dep}' which is not in the sample graph"
                    )

    return errors


def make_sample_config(config: GodConfig, sg: SampleGeneratorConfig) -> GodConfig:
    identity_grad = GradientConfig(
        method=IdentityLearnerConfig(),
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
    identity_learner = LearnConfig(
        model_learner=identity_grad,
        optimizer_learner=identity_grad,
        optimizer={},
    )

    match sg.input:
        case GaussianSampleInput():
            dataset_source = GaussianNoiseTaskFamily(
                shape=sg.input_shape,
                n=sg.num_samples * len(config.levels),
            )
        case DataSampleInput():
            dataset_source = config.levels[0].dataset_source

    new_levels = []
    for level in config.levels:
        new_levels.append(
            dataclasses.replace(
                level,
                dataset_source=dataset_source,
                dataset=dataclasses.replace(
                    level.dataset,
                    num_examples_in_minibatch=sg.num_samples,
                    num_examples_total=sg.num_samples,
                    is_test=False,
                    augment=False,
                ),
                nested=dataclasses.replace(level.nested, num_steps=1, reset_t=None),
                learner=identity_learner,
                track_logs=no_track,
                collect_predictions=True,
            )
        )

    return dataclasses.replace(
        config,
        transition_graph=sg.transition_graph,
        readout_graph=sg.readout_graph,
        levels=new_levels,
        sample_generators=[],
        epochs=1,
    )


def report_samples(
    sg: SampleGeneratorConfig,
    stats: STAT,
    sample_logger: ThreadedScalarLogger,
) -> None:
    match sg.reporter:
        case ImageReporter(title):
            sample_logger.log_image(stats, title, 3)
