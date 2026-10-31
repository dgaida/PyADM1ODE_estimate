"""The amortised GRU observer (PINN B): model, datasets, training, online driver and
the dataset adapter shared with the filters.
"""

from ..observation_torch import TorchObservationModel
from .adapter import (
    FeatureSpec,
    PinnData,
    SplitRecord,
    parse_noise_spec,
)
from .datasets import (
    MeasurementDataset,
    ObserverDataset,
    generate_observer_dataset,
)
from .model import Adm1Observer
from .online import OnlineEstimate, SlidingWindowObserver
from .train import (
    PretrainResult,
    SelfSupPretrainResult,
    finetune_observer,
    observer_predict,
    per_state_nrmse,
    pretrain_observer,
    pretrain_observer_selfsup,
    pretrain_observer_sim2real,
)

__all__ = [
    "Adm1Observer",
    "FeatureSpec",
    "MeasurementDataset",
    "ObserverDataset",
    "OnlineEstimate",
    "PinnData",
    "PretrainResult",
    "SelfSupPretrainResult",
    "SlidingWindowObserver",
    "SplitRecord",
    "TorchObservationModel",
    "finetune_observer",
    "generate_observer_dataset",
    "observer_predict",
    "parse_noise_spec",
    "per_state_nrmse",
    "pretrain_observer",
    "pretrain_observer_selfsup",
    "pretrain_observer_sim2real",
]
