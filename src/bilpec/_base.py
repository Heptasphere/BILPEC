"""Ensemble Method designed for tabarena."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from numpy._typing._array_like import NDArray

if TYPE_CHECKING:
    from autogluon.core.metrics import Scorer
    from numpy.typing import NDArray


class AbstractEnsemble(ABC):
    """Abstract Class for Ensemble Methods."""

    def __init__(
        self,
        ensemble_size: int,
        problem_type: str,
        metric: Scorer,
    ) -> None:
        """Base arguments needed for an ensemble method to function.

        Args:
            ensemble_size (int): Number of base learner in ensemble.
            problem_type (str): Type of problem, among binary, multiclass and regression.
            metric (Scorer): _description_
        """
        self.ensemble_size: int = ensemble_size
        # Used by tab_arena, not the type of the actual task.
        self.problem_type: str = problem_type
        self.metric: Scorer = metric
        self.weights_: NDArray[float] | None = None
        # Used by our custom ensemble methods, the type of the actual task.
        match self.metric.name:
            case "roc_auc":
                self.actual_problem_type = "binary"
            case "log_loss":
                self.actual_problem_type = "multiclass"
            case _:
                raise NotImplementedError(
                    f"We haven't implemented ensemble methods for the metric {self.metric.name}."
                )

    @abstractmethod
    def fit(self, predictions: NDArray[float], labels: NDArray) -> None:
        """Train the ensemble model.

        Args:
            predictions (NDArray): Predicted class probabilities.
                                    Shape (n_models, n_samples, n_classes) for multiclass.
                                    Shape (n_models, n_samples) for binary. Considers the prediction of class 1.
            labels (NDArray): True labels. Shape (n_samples,)
        """
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, predictions: NDArray[float]) -> NDArray[float]:
        """Predict class probabilities using the ensemble model.

        Args:
            predictions (NDArray):  Predicted class probabilities
                                    Shape (n_models, n_samples, n_classes) for multiclass.
                                    Shape (n_models, n_samples) for binary. Considers the prediction of class 1.

        Returns:
            NDArray: Predicted class probabilities.
                    Shape (n_models, n_samples, n_classes) for multiclass.
                    Shape (n_models, n_samples) for binary. Gives the prediction of class 1.
        """
        raise NotImplementedError

    def predict(self, predictions: NDArray[int]) -> NDArray[int]:
        """Predict a class. Not used by our metrics.

        Args:
            predictions (NDArray): Array of predicted class probabilities
                                    of shape (n_models, n_samples, n_classes) or (n_models, n_samples).

        Returns:
            NDArray: Predicted classes. Shape (n_samples,).
        """
        raise NotImplementedError(
            f"predict method not implemented for {type(self).__class__}."
        )
