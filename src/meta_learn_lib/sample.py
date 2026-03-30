from typing import Callable
import jax
import numpy as np

from meta_learn_lib.config import (
    GaussianSampleInput,
    GodConfig,
    ImageReporter,
    SampleGeneratorConfig,
    UnlabeledSource,
    LabeledSource,
)
from meta_learn_lib.env import Outputs
from meta_learn_lib.inference import ReadoutFn, TransitionFn, create_raw_inference
from meta_learn_lib.interface import GodInterface
from meta_learn_lib.logger import Logger
from meta_learn_lib.lib_types import PRNG


type SampleRunner[ENV] = Callable[[ENV, Logger, PRNG, int], None]


def validate_sample_generators(config: GodConfig) -> list[str]:
    errors: list[str] = []
    for i, sg in enumerate(config.sample_generators):
        all_graph_nodes = set(sg.transition_graph.keys()) | set(sg.readout_graph.keys())

        for node_name in all_graph_nodes:
            if node_name in sg.source_nodes:
                node = sg.source_nodes[node_name]
                if not isinstance(node, (UnlabeledSource, LabeledSource)):
                    errors.append(
                        f"Sample generator {i}: source_nodes['{node_name}'] must be UnlabeledSource or LabeledSource"
                    )
            elif node_name not in config.nodes:
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
                        f"Sample generator {i}: readout_graph['{node_name}'] depends on '{dep}' which is not in the sample graph"
                    )

        for node_name, deps in sg.transition_graph.items():
            for dep in deps:
                if dep not in all_graph_nodes:
                    errors.append(
                        f"Sample generator {i}: transition_graph['{node_name}'] depends on '{dep}' which is not in the sample graph"
                    )

    return errors


def build_sample_runner[ENV](
    config: GodConfig,
    meta_interfaces: list[dict[str, GodInterface[ENV]]],
    iterations_per_epoch: int,
) -> SampleRunner[ENV]:
    meta_interface = meta_interfaces[0]

    sample_fns: list[tuple[SampleGeneratorConfig, Callable[[ENV, jax.Array], jax.Array]]] = []
    for sg in config.sample_generators:
        merged_nodes = config.nodes | sg.source_nodes
        transition_fn, readout_fn = create_raw_inference(
            sg.transition_graph,
            sg.readout_graph,
            merged_nodes,
            meta_interface,
        )

        def make_sample_fn(t: TransitionFn[ENV], r: ReadoutFn[ENV]) -> Callable[[ENV, jax.Array], jax.Array]:
            def sample_fn(env: ENV, z: jax.Array) -> jax.Array:
                env = t(env, (z, z))
                outputs: Outputs = r(env, (z, z))
                return outputs.prediction

            return sample_fn

        sample_fns.append((sg, make_sample_fn(transition_fn, readout_fn)))

    def generate_input(sg: SampleGeneratorConfig, prng: PRNG) -> jax.Array:
        match sg.input:
            case GaussianSampleInput():
                return jax.random.normal(prng, (sg.num_samples, *sg.input_shape))

    def report_image(logger: Logger, title: str, samples: np.ndarray, iteration: int) -> None:
        for i, sample in enumerate(samples):
            if sample.ndim == 3:
                sample = np.transpose(sample, (1, 2, 0))
            logger.log_image(title, f"sample_{i}", iteration, sample)

    def run(env: ENV, logger: Logger, prng: PRNG, iteration: int) -> None:
        for sg, sample_fn in sample_fns:
            interval = iterations_per_epoch * sg.every_n_epochs
            if interval <= 0 or iteration % interval != 0:
                continue

            sample_prng, prng = jax.random.split(prng)
            z: jax.Array = generate_input(sg, PRNG(sample_prng))
            samples = np.asarray(jax.vmap(lambda z_i: sample_fn(env, z_i))(z))

            match sg.reporter:
                case ImageReporter(title):
                    report_image(logger, title, samples, iteration)

    return run
