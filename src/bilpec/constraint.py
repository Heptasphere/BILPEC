"""Ensemble Method designed for tabarena."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING

from numpy._typing._array_like import NDArray

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


@dataclass(kw_only=True)
class Constraint(ABC):
    # Sort of threshold, possibilities depend on constraint type
    threshold_type: str
    # Threshold that needs to be cleared
    threshold: float | int
    # Whether the constraint is to be above or below the threshold.
    below_threshold: bool = False
    # Whether it should be possible to relax the constraint
    relaxation: bool = False
    # Penalty to apply through relaxation. A type of a float.
    penalty: str | float = 0.0


@dataclass(kw_only=True)
class MeasureConstraint(Constraint):
    # Name of the measure
    measure_name: str
    # The actual measures per model
    measures: NDArray
    # Whether the bound is a total, a mean or otherwise
    threshold_type: str = "total"
    # Threshold types:
    # 'total'
    # 'mean'


@dataclass(kw_only=True)
class LabelConstraint(Constraint):
    # ID of the label
    label_id: int
    # Maximum number of samples used for constraint.
    # All of them by default.
    sample_size: int = 10**15
    # Threshold types:
    # 'error_rate'
    # 'mean_probability'


@dataclass(kw_only=True)
class SampleSetConstraint(Constraint):
    # Method saying whether a sample is counted or not
    parser: Callable[NDArray, bool]
    # Threshold types:
    # 'accuracy'
    # 'mean_probability'
