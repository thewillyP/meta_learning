import clearml
import h5py
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Any, Protocol
import matplotlib.pyplot as plt
from collections import defaultdict
import threading
import queue
from dataclasses import dataclass
import jax.numpy as jnp


class Logger(Protocol):
    def get_context(self): ...

    def close_context(self, context): ...

    def log_scalar(self, context, title: str, series: str, value: float, iteration: int, max_count: int): ...


class HDF5Logger:
    def __init__(self, log_dir: str, task_id: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.task_id = task_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"metrics_{task_id}_{timestamp}.h5"
        self.metric_indices = {}

        # Initialize HDF5 file with task metadata
        with h5py.File(self.log_file, "w") as f:
            f.attrs["task_id"] = task_id
            f.attrs["created"] = datetime.now().isoformat()

    def get_context(self) -> h5py.File:
        """Get an open HDF5 file context for logging"""
        return h5py.File(self.log_file, "a")

    def close_context(self, f: h5py.File):
        """Close the HDF5 file context"""
        f.close()

    def log_scalar(self, f: h5py.File, title: str, series: str, value: float, iteration: int, max_count: int):
        """Log a scalar metric to an open HDF5 file"""
        if series not in f:
            dataset = f.create_dataset(series, shape=(max_count,), dtype=np.float64, fillvalue=np.nan)
            dataset.attrs["title"] = title
            f.create_dataset(f"{series}_iterations", shape=(max_count,), dtype=np.int32, fillvalue=-1)
            self.metric_indices[series] = 0

        idx = self.metric_indices[series]
        if idx < len(f[series]):
            f[series][idx] = value
            f[f"{series}_iterations"][idx] = iteration
            self.metric_indices[series] += 1


class ClearMLLogger:
    def __init__(self, task: clearml.Task):
        self.task = task

    def get_context(self):
        """No context needed for ClearML"""
        return None

    def close_context(self, context):
        """No context to close for ClearML"""
        pass

    def log_scalar(self, context, title: str, series: str, value: float, iteration: int, max_count: int):
        """Log a scalar metric to ClearML"""
        self.task.get_logger().report_scalar(title=title, series=series, value=value, iteration=iteration)


class PrintLogger:
    def __init__(self):
        pass

    def get_context(self):
        """No context needed for PrintLogger"""
        return None

    def close_context(self, context):
        """No context to close for PrintLogger"""
        pass

    def log_scalar(self, context, title: str, series: str, value: float, iteration: int, max_count: int):
        """Print the scalar metric to console"""
        print(f"[{title}] {series} @ {iteration}/{max_count}: {value}")


class MatplotlibLogger:
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.data = defaultdict(lambda: {"iterations": [], "values": [], "title": None})

    def get_context(self):
        return None

    def close_context(self, context):
        pass

    def log_scalar(self, context, title: str, series: str, value: float, iteration: int, max_count: int):
        self.data[series]["iterations"].append(iteration)
        self.data[series]["values"].append(value)
        self.data[series]["title"] = title

    def generate_figures(self):
        for series, data in self.data.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(data["iterations"], data["values"], linewidth=1)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Value")
            ax.set_title(f"{data['title']} - {series}")
            ax.grid(True, alpha=0.3)

            save_path = self.save_dir / f"{series}.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)


class MultiLogger:
    def __init__(self, loggers: list[Logger]):
        self.loggers = loggers

    def get_context(self):
        return [logger.get_context() for logger in self.loggers]

    def close_context(self, contexts):
        for logger, context in zip(self.loggers, contexts):
            logger.close_context(context)

    def log_scalar(self, contexts, title: str, series: str, value: float, iteration: int, max_count: int):
        for logger, context in zip(self.loggers, contexts):
            logger.log_scalar(context, title, series, value, iteration, max_count)


@dataclass
class LogJob:
    """Represents a single log_scalar job to be processed"""

    title: str
    series: str
    value: float
    iteration: int
    max_count: int


class ThreadedLogger:
    """Wrapper logger that processes log_scalar calls asynchronously in a background thread"""

    def __init__(self, logger: Logger):
        self.logger = logger
        self.job_queue = queue.Queue()  # Unbounded queue
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.context = None
        self.worker_thread.start()

    def _worker(self):
        """Background worker that processes jobs from the queue"""
        # Get context once for the worker thread
        self.context = self.logger.get_context()

        while not self.stop_event.is_set():
            try:
                # Get job with timeout to allow periodic checking of stop_event
                job = self.job_queue.get(timeout=0.001)

                # Process the job
                self.logger.log_scalar(self.context, job.title, job.series, job.value, job.iteration, job.max_count)

                # Mark job as done
                self.job_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in ThreadedLogger worker: {e}")

    def get_context(self):
        """Return None as context is managed internally by the worker thread"""
        return None

    def close_context(self, context):
        """No external context to close"""
        pass

    def log_scalar(self, context, title: str, series: str, value: float, iteration: int, max_count: int):
        """Queue a log_scalar job to be processed asynchronously"""
        job = LogJob(title, series, value, iteration, max_count)
        self.job_queue.put(job)

    def flush(self):
        """Wait for all queued jobs to be processed"""
        self.job_queue.join()

    def shutdown(self):
        """Gracefully shutdown the worker thread"""
        # First flush all pending jobs
        self.flush()

        # Signal worker to stop
        self.stop_event.set()

        # Wait for worker thread to finish
        self.worker_thread.join(timeout=5.0)

        # Close the logger context if it exists
        if self.context is not None:
            self.logger.close_context(self.context)

    def __del__(self):
        """Ensure cleanup on deletion"""
        if hasattr(self, "stop_event") and not self.stop_event.is_set():
            self.shutdown()


@dataclass
class IterationData:
    """Bundle all data needed for logging an iteration"""

    k: int
    tr_loss: Any
    tr_acc: Any
    lrs1: Any
    lrs2: Any
    wds1: Any
    wds2: Any
    tr_grs: Any
    vl_loss: Any
    vl_acc: Any
    meta_grs: Any
    hT: Any
    aT: Any  # Remove aT_buffer from here!
    te_loss: Any
    te_acc: Any
    config: Any
    total_tr_vb: int


class SequentialLoggingThread:
    """Processes logging iterations sequentially in a background thread"""

    def __init__(self, logger, window_size=100):
        self.logger = logger
        self.queue = queue.Queue()
        self.window_size = window_size
        self.aT_buffer = []
        self.worker = threading.Thread(target=self._process_iterations, daemon=False)
        self.worker.start()

    def _process_iterations(self):
        """Process iterations in order"""
        while True:
            data = self.queue.get()  # Block indefinitely
            if data is None:  # Shutdown signal
                break

            self._log_iteration(data)
            self.queue.task_done()

    def _log_iteration(self, data: IterationData):
        """Your nested loop logic"""
        context = self.logger.get_context()

        def corr_per_example_dot(a_prev, a_curr, eps=1e-8):
            dot = jnp.sum(a_prev * a_curr, axis=1)
            norm_prev = jnp.linalg.norm(a_prev, axis=1)
            norm_curr = jnp.linalg.norm(a_curr, axis=1)
            corr_per_ex = dot / (norm_prev * norm_curr + eps)
            return jnp.mean(corr_per_ex)

        for i in range(data.tr_loss.shape[0]):
            for j in range(data.tr_loss.shape[1]):
                iteration = data.k * data.tr_loss.shape[0] * data.tr_loss.shape[1] + i * data.tr_loss.shape[1] + j + 1
                if iteration % data.config.checkpoint_every_n_minibatches == 0:
                    self.logger.log_scalar(
                        context,
                        "train/loss",
                        "train_loss",
                        data.tr_loss[i, j],
                        iteration,
                        data.total_tr_vb * data.config.num_base_epochs,
                    )
                    self.logger.log_scalar(
                        context,
                        "train/accuracy",
                        "train_accuracy",
                        data.tr_acc[i, j],
                        iteration,
                        data.total_tr_vb * data.config.num_base_epochs,
                    )
                    self.logger.log_scalar(
                        context,
                        "train/recurrent_learning_rate",
                        "train_recurrent_learning_rate",
                        data.lrs1[i, j][0],
                        iteration,
                        data.total_tr_vb * data.config.num_base_epochs,
                    )
                    self.logger.log_scalar(
                        context,
                        "train/readout_learning_rate",
                        "train_readout_learning_rate",
                        data.lrs2[i, j][0],
                        iteration,
                        data.total_tr_vb * data.config.num_base_epochs,
                    )
                    self.logger.log_scalar(
                        context,
                        "train/recurrent_weight_decay",
                        "train_recurrent_weight_decay",
                        data.wds1[i, j][0],
                        iteration,
                        data.total_tr_vb * data.config.num_base_epochs,
                    )
                    self.logger.log_scalar(
                        context,
                        "train/readout_weight_decay",
                        "train_readout_weight_decay",
                        data.wds2[i, j][0],
                        iteration,
                        data.total_tr_vb * data.config.num_base_epochs,
                    )
                    self.logger.log_scalar(
                        context,
                        "train/gradient_norm",
                        "train_gradient_norm",
                        jnp.linalg.norm(data.tr_grs[i, j]),
                        iteration,
                        data.total_tr_vb * data.config.num_base_epochs,
                    )
                    self.logger.log_scalar(
                        context,
                        "validation/loss",
                        "validation_loss",
                        data.vl_loss[i, j],
                        iteration,
                        data.total_tr_vb * data.config.num_base_epochs,
                    )
                    self.logger.log_scalar(
                        context,
                        "validation/accuracy",
                        "validation_accuracy",
                        data.vl_acc[i, j],
                        iteration,
                        data.total_tr_vb * data.config.num_base_epochs,
                    )
                    self.logger.log_scalar(
                        context,
                        "meta/gradient_norm",
                        "meta_gradient_norm",
                        jnp.linalg.norm(data.meta_grs[i, j]),
                        iteration,
                        data.total_tr_vb * data.config.num_base_epochs,
                    )
                    self.logger.log_scalar(
                        context,
                        "train/final_rnn_activation_norm",
                        "train_final_rnn_activation_norm",
                        data.hT[i, j],
                        iteration,
                        data.total_tr_vb * data.config.num_base_epochs,
                    )

                    aT_curr = data.aT[i, j]

                    # Use self.aT_buffer, not data.aT_buffer!
                    self.aT_buffer.append(aT_curr)
                    if len(self.aT_buffer) > self.window_size:
                        self.aT_buffer.pop(0)

                    if len(self.aT_buffer) > 1:
                        corrs = [corr_per_example_dot(prev, aT_curr) for prev in self.aT_buffer[:-1]]
                        at_corr = jnp.mean(jnp.array(corrs))
                        self.logger.log_scalar(
                            context,
                            "train/aT_correlation_window",
                            "train_aT_correlation_window",
                            at_corr,
                            iteration,
                            data.total_tr_vb * data.config.num_base_epochs,
                        )

            self.logger.log_scalar(
                context,
                "test/loss",
                "test_loss",
                data.te_loss[i],
                iteration,
                data.total_tr_vb * data.config.num_base_epochs,
            )
            self.logger.log_scalar(
                context,
                "test/accuracy",
                "test_accuracy",
                data.te_acc[i],
                iteration,
                data.total_tr_vb * data.config.num_base_epochs,
            )

        self.logger.close_context(context)

    def add_iteration(self, data: IterationData):
        """Queue an iteration for logging"""
        self.queue.put(data)

    def shutdown(self):
        """Wait for all iterations and shutdown"""
        self.queue.put(None)  # Signal shutdown
        self.worker.join()
