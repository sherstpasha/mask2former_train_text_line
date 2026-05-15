__all__ = [
    "TrainConfig",
    "TrainingResult",
    "find_resume_checkpoint",
    "run_training",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module 'utils' has no attribute {name!r}")
    from .training import TrainConfig, TrainingResult, find_resume_checkpoint, run_training

    exports = {
        "TrainConfig": TrainConfig,
        "TrainingResult": TrainingResult,
        "find_resume_checkpoint": find_resume_checkpoint,
        "run_training": run_training,
    }
    return exports[name]
