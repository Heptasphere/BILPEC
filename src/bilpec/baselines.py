"""Code for baselines used."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, override

import numpy as np
from autogluon.core.metrics import log_loss
from scipy.stats import binom

from bilpec._base import AbstractEnsemble
from bilpec.constraint import (
    Constraint,
    MeasureConstraint,
)

if TYPE_CHECKING:
    from autogluon.core.metrics import Scorer
    from numpy.typing import NDArray


class SoftVoting(AbstractEnsemble):
    """Ensemble Method that implements Soft Voting."""

    @override
    def __init__(
        self,
        ensemble_size: int,
        problem_type: str,
        metric: Scorer,
    ) -> None:
        super().__init__(ensemble_size, problem_type, metric)

    @override
    def fit(
        self, predictions: NDArray[float], labels: NDArray[int], **fit_params
    ) -> None:
        self.weights_: NDArray[float] = np.ones(predictions.shape[0])

    @override
    def predict_proba(
        self, predictions: NDArray[float], **predict_params
    ) -> NDArray[float]:
        return predictions.mean(axis=0)


class HardVoting(AbstractEnsemble):
    """Ensemble Method that implements Hard Voting.
    Should be used with NoPreprocessingTaskEvaluator.
    """

    @override
    def __init__(
        self,
        ensemble_size: int,
        problem_type: str,
        metric: Scorer,
    ) -> None:
        super().__init__(ensemble_size, problem_type, metric)

    @override
    def fit(
        self, predictions: NDArray[float], labels: NDArray[int], **fit_params
    ) -> None:
        self.weights_: NDArray[float] = np.ones(predictions.shape[0])

    # def predict(self, predictions: NDArray) -> NDArray:
    #     if self.actual_problem_type == "binary":
    #         # TODO: check if boolean array is what we want
    #         return (predictions > 0.5).mean(axis=0) > 0.5
    #     elif self.actual_problem_type == "multiclass":
    #         return (predictions > 0.5).mean(axis=0).argmax(axis=1)
    #     else:
    #         raise NotImplementedError(f"get_prediction not implemented for problems of type {self.problem_type}.")

    # TODO
    @override
    def predict_proba(
        self, predictions: NDArray[float], **predict_params
    ) -> NDArray[float]:
        if self.actual_problem_type not in ["binary", "multiclass"]:
            raise NotImplementedError(
                f"Not implemented problem type {self.actual_problem_type} for {self.__class__.__name__}."
            )
        if predictions.ndim == 2:
            return ((predictions > 0.5) + 0.5 * (predictions == 0.5)).mean(axis=0)
        # For multiclass and not preprocessed binary.
        return self._multiclass_predict_proba(predictions)

    def _multiclass_predict_proba(self, predictions: NDArray[float]) -> NDArray[float]:
        # We create a matrix full of zeros with ones at the location
        # of each "best" class for a learner.
        chosen_class_probas: NDArray[float] = np.max(predictions, axis=2, keepdims=True)
        binary_matrix: NDArray[int] = predictions == chosen_class_probas
        # In cases where multiple classes are equally the most probable,
        # we divide their weight by the number of equally probable classes.
        binary_matrix_row_sum: NDArray[int] = binary_matrix.sum(axis=2, keepdims=True)
        binary_matrix: NDArray[int] = binary_matrix / binary_matrix_row_sum
        return binary_matrix.mean(axis=0)


class SingleBest(AbstractEnsemble):
    """Ensemble Method that implements Single Best."""

    @override
    def __init__(
        self,
        ensemble_size: int,
        problem_type: str,
        metric: Scorer,
    ) -> None:
        super().__init__(
            ensemble_size=ensemble_size, problem_type=problem_type, metric=metric
        )

    @override
    def fit(
        self, predictions: NDArray[float], labels: NDArray[int], **fit_params
    ) -> None:
        # Meant to be used with validation data.
        # Every base learner has a weight of 0 except the single best one.
        self.weights_: NDArray[float] = np.zeros(predictions.shape[0])
        if predictions.ndim == 2:
            single_metrics: list[float] = [
                self.metric(labels.astype(np.bool_), predictions[ind, :])
                for ind in range(predictions.shape[0])
            ]
        elif predictions.ndim == 3:
            single_metrics: list[float] = [
                self.metric(labels, predictions[ind, :])
                for ind in range(predictions.shape[0])
            ]
        else:
            raise ValueError(
                f"predictions neither has 2 or 3 dimensions but {predictions.ndim}"
            )
        if self.metric.greater_is_better:
            self.best_index: int = np.argmax(single_metrics)
        else:
            self.best_index: int = np.argmin(single_metrics)
        self.weights_[self.best_index] = 1

    # TODO: Take into account multiclass case
    @override
    def predict_proba(
        self, predictions: NDArray[float], **predict_params
    ) -> NDArray[float]:
        return predictions[np.where(self.weights_ == 1)][0]


@dataclass(frozen=True)
class SherlockResult:
    """Result of the SHErLoCk stochastic search."""

    selected: NDArray[np.int_]
    score: float
    prediction: NDArray
    resource_usage: dict[str, float]
    steps: int
    stop_reason: str
    history: tuple[float, ...]


class SHERLOCK(AbstractEnsemble):
    """
    This code is based on the matlab implementation of SHErLoCk found at https://codeocean.com/capsule/3944336/tree/v2

    It is under the following license:

    MIT License

    Copyright (c) 2019 Attila Tiba, Andras Hajdu, Henrietta Toman, Gyorgy Terdik

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """

    def __init__(
        self,
        ensemble_size: int,
        problem_type: str,
        metric: Scorer = log_loss,
        constraints: list[dict | Constraint] | None = None,
        *,
        max_steps: int = 1000,
        stop_score: float | None = None,
        random_state: int | None = None,
        min_size: int = 1,
        odd_only: bool = True,
    ) -> None:
        super().__init__(
            ensemble_size=ensemble_size,
            problem_type=problem_type,
            metric=metric,
        )

        # TODO: This is specifically for our experiments.
        # Remove or change this afterwards.
        if constraints is None:
            self.constraints = [
                MeasureConstraint(
                    threshold_type="total",
                    threshold=20,
                    below_threshold=True,
                    measure_name="cardinality",
                    measures=np.ones(80),
                )
            ]
        else:
            self.constraints: list[dict | Constraint] = list(constraints)

        self.max_steps = max_steps
        self.stop_score = stop_score
        self.random_state = random_state
        self.min_size = min_size
        self.odd_only = odd_only

        # Populated by _parse_constraints() during fit().
        self.measure_constraints: list[MeasureConstraint] = []

        self.costs_: NDArray[np.float64] | None = None
        self.budgets_: NDArray[np.float64] | None = None
        self.constraint_names_: list[str] = []

        # Populated by fit().
        self.selected_indices_: NDArray[np.int_] | None = None
        self.result_: SherlockResult | None = None
        self.weights_: NDArray[np.float64] | None = None
        self.n_models_in_: int | None = None

    @override
    def fit(
        self,
        predictions: NDArray[np.float64],
        labels: NDArray,
        **fit_params,
    ) -> SHERLOCK:
        del fit_params

        predictions = np.asarray(predictions, dtype=float)
        labels = np.asarray(labels)

        self.n_models_in_ = predictions.shape[0]

        (
            self.costs_,
            self.budgets_,
            self.constraint_names_,
        ) = self._parse_constraints(
            constraints=self.constraints,
            n_models=self.n_models_in_,
        )

        self.result_ = self._search(
            predictions=predictions,
            labels=labels,
        )

        self.selected_indices_ = self.result_.selected.copy()

        # Binary selection mask: 1 if selected, 0 otherwise.
        self.weights_ = np.zeros(
            self.n_models_in_,
            dtype=np.int8,
        )
        self.weights_[self.selected_indices_] = 1

        return self

    @override
    def predict_proba(
        self,
        predictions: NDArray[np.float64],
        **predict_params,
    ) -> NDArray[np.float64]:
        """Return vote proportions from the selected learners."""

        predictions = np.asarray(
            predictions,
            dtype=float,
        )

        selected_mask = self.weights_.astype(bool)

        if not np.any(selected_mask):
            raise RuntimeError("The fitted ensemble contains no selected learners.")

        selected_predictions = predictions[selected_mask]

        return self._vote_proportions(selected_predictions)

    def predict(
        self,
        predictions: NDArray[np.float64],
    ) -> NDArray[np.int_]:
        """Return class predictions from the selected ensemble."""
        probabilities = self.predict_proba(predictions)

        if self.problem_type == "binary":
            return (probabilities >= 0.5).astype(int)

        return np.argmax(probabilities, axis=1).astype(int)

    def _search(
        self,
        predictions: NDArray[np.float64],
        labels: NDArray,
    ) -> SherlockResult:
        """Run the budget-driven stochastic SHErLoCk search.

        A candidate ensemble grows until no remaining learner fits the
        remaining resource budgets. There is no fixed ensemble cardinality.
        """
        if self.costs_ is None:
            raise RuntimeError("Constraints have not been initialized.")

        if self.budgets_ is None:
            raise RuntimeError("Budgets have not been initialized.")

        if self.budgets_.size == 0:
            raise RuntimeError("SHErLoCk requires at least one resource budget.")

        n_models = predictions.shape[0]

        individual_scores = self._individual_correctness(
            predictions=predictions,
            labels=labels,
        )

        rng = np.random.default_rng(self.random_state)

        best_indices = np.empty(
            0,
            dtype=np.int64,
        )
        best_accuracy = -np.inf
        best_prediction: NDArray[np.int_] | None = None
        best_usage = np.zeros_like(
            self.budgets_,
            dtype=float,
        )

        history: list[float] = []
        stop_reason = "max_steps"
        steps_completed = 0

        # MAXSTEP controls how many stochastic constructions are attempted.
        for step in range(1, self.max_steps + 1):
            steps_completed = step

            chosen: list[int] = []

            usage = np.zeros_like(
                self.budgets_,
                dtype=float,
            )

            available = np.ones(
                n_models,
                dtype=bool,
            )

            # No ensemble-size condition. The budget determines when this
            # candidate construction ends.
            while True:
                remaining = np.maximum(
                    self.budgets_ - usage,
                    0.0,
                )

                # A learner is feasible if it respects every remaining budget.
                feasible = available & np.all(
                    self.costs_ <= remaining[None, :] + 1e-12,
                    axis=1,
                )

                candidates = np.flatnonzero(feasible)

                # Original budget-driven stopping condition for this restart.
                if candidates.size == 0:
                    break

                candidate_costs = self.costs_[candidates]

                # For every candidate and resource:
                #
                # floor(remaining resource / candidate cost)
                capacities = np.floor(remaining[None, :] / candidate_costs)

                # Multiple-resource extension: the tightest resource determines
                # the hypothetical replica count.
                replica_counts = np.min(
                    capacities,
                    axis=1,
                ).astype(np.int64)

                replica_counts = np.maximum(
                    replica_counts,
                    1,
                )

                # Equation (5) uses odd replica counts.
                effective_replica_counts = replica_counts - (replica_counts % 2 == 0)

                effective_replica_counts = np.maximum(
                    effective_replica_counts,
                    1,
                )

                # Vectorized equation (5).
                utilities = binom.sf(
                    effective_replica_counts // 2,
                    effective_replica_counts,
                    individual_scores[candidates],
                )

                utilities = np.nan_to_num(
                    utilities,
                    nan=0.0,
                    posinf=1.0,
                    neginf=0.0,
                )

                utility_sum = float(utilities.sum())

                # Equation (6).
                if utility_sum > 0.0:
                    selection_probabilities = utilities / utility_sum
                else:
                    selection_probabilities = None

                selected = int(
                    rng.choice(
                        candidates,
                        p=selection_probabilities,
                    )
                )

                chosen.append(selected)
                usage += self.costs_[selected]
                available[selected] = False

                current_size = len(chosen)

                # The original algorithm evaluates odd-sized intermediate
                # ensembles.
                should_evaluate = current_size >= self.min_size and (
                    not self.odd_only or current_size % 2 == 1
                )

                if not should_evaluate:
                    continue

                chosen_indices = np.asarray(
                    chosen,
                    dtype=np.int64,
                )

                selected_predictions = predictions[chosen_indices]

                vote_proportions = self._vote_proportions(selected_predictions)

                predicted_labels = self._vote_proportions_to_prediction(
                    vote_proportions
                )
                current_accuracy = self._accuracy(
                    predicted_labels=predicted_labels,
                    labels=labels,
                )

                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    best_indices = chosen_indices.copy()
                    best_prediction = predicted_labels.copy()
                    best_usage = usage.copy()

                # Optional approximation of STOP. MAXSTEP still applies when
                # stop_score is absent or never reached.
                if self.stop_score is not None and current_accuracy >= self.stop_score:
                    stop_reason = "stop_score"
                    history.append(float(best_accuracy))

                    return SherlockResult(
                        selected=best_indices,
                        score=float(best_accuracy),
                        prediction=best_prediction,
                        resource_usage={
                            name: float(value)
                            for name, value in zip(
                                self.constraint_names_,
                                best_usage,
                            )
                        },
                        steps=steps_completed,
                        stop_reason=stop_reason,
                        history=tuple(history),
                    )

            # Store one value after each completed stochastic construction.
            if np.isfinite(best_accuracy):
                history.append(float(best_accuracy))

        if best_indices.size == 0 or best_prediction is None:
            warnings.warn(
                "SHERLOCK: No feasible odd-sized ensemble satisfies the given constraints. Defaulting to Hard Voting."
            )
            best_indices = np.arange(n_models)

        return SherlockResult(
            selected=best_indices,
            score=float(best_accuracy),
            prediction=best_prediction,
            resource_usage={
                name: float(value)
                for name, value in zip(
                    self.constraint_names_,
                    best_usage,
                )
            },
            steps=steps_completed,
            stop_reason=stop_reason,
            history=tuple(history),
        )

    def _vote_proportions(
        self,
        predictions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return hard-vote proportions from selected learners."""
        predictions = np.asarray(
            predictions,
            dtype=float,
        )

        if predictions.shape[0] == 0:
            raise ValueError("Cannot aggregate an empty ensemble.")

        if self.problem_type == "binary":
            if predictions.ndim != 2:
                raise ValueError(
                    "Binary predictions must have shape (n_selected_models, n_samples)."
                )

            learner_votes = self._binary_votes(predictions)

            # Proportion of votes for class 1.
            return learner_votes.mean(
                axis=0,
                dtype=float,
            )

        if self.problem_type == "multiclass":
            if predictions.ndim != 3:
                raise ValueError(
                    "Multiclass predictions must have shape "
                    "(n_selected_models, n_samples, n_classes)."
                )

            n_classes = predictions.shape[2]

            learner_classes = np.argmax(
                predictions,
                axis=2,
            )

            one_hot_votes = np.eye(
                n_classes,
                dtype=np.float32,
            )[learner_classes]

            return one_hot_votes.mean(
                axis=0,
                dtype=float,
            )

        raise NotImplementedError(f"Unsupported problem type {self.problem_type!r}.")

    def _vote_proportions_to_prediction(
        self,
        vote_proportions: NDArray[np.float64],
    ) -> NDArray[np.int_]:
        if self.problem_type == "binary":
            # Ensemble tie convention: class 1 wins at exactly 0.5.
            return (vote_proportions >= 0.5).astype(np.int8)

        return np.argmax(
            vote_proportions,
            axis=1,
        ).astype(np.int64)

    def _individual_correctness(
        self,
        predictions: NDArray[np.float64],
        labels: NDArray,
    ) -> NDArray[np.float64]:
        """Calculate hard-decision accuracy for each learner."""
        if self.problem_type == "binary":
            learner_votes = self._binary_votes(predictions)

            # Must match _vote_proportions_to_prediction().
            predicted_labels = (learner_votes >= 0.5).astype(np.int8)

        elif self.problem_type == "multiclass":
            predicted_labels = np.argmax(
                predictions,
                axis=2,
            )

        else:
            raise NotImplementedError(
                f"Unsupported problem type {self.problem_type!r}."
            )

        return np.mean(
            predicted_labels == labels[None, :],
            axis=1,
        )

    def _accuracy(
        self,
        predicted_labels: NDArray[np.int_],
        labels: NDArray,
    ) -> float:
        """Return empirical classification accuracy."""
        return float(np.mean(predicted_labels == labels))

    def _binary_votes(
        self,
        predictions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Convert class-1 probabilities into binary votes.

        Returns:
            1.0 when P(class 1) > 0.5,
            0.0 when P(class 1) < 0.5,
            0.5 when P(class 1) == 0.5.
        """
        predictions = np.asarray(
            predictions,
            dtype=float,
        )

        return (predictions > 0.5).astype(float) + 0.5 * np.isclose(
            predictions,
            0.5,
            rtol=0.0,
            atol=1e-12,
        )

    def _parse_constraints(
        self,
        constraints: list[dict | Constraint] | None,
        n_models: int,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        list[str],
    ]:
        """Convert Constraint objects into SHErLoCk cost matrices.

        The current stochastic knapsack search supports constraints of the form:

            sum(measures[selected_models]) <= threshold

        Consequently, directly supported MeasureConstraint instances require:

            threshold_type == "total"
            below_threshold is True
            relaxation is False

        Args:
            constraints:
                Constraint instances or dictionaries from which constraint
                instances can be constructed.

            n_models:
                Number of learners in the prediction tensor.

        Returns:
            costs:
                Shape ``(n_models, n_constraints)``.

            budgets:
                Shape ``(n_constraints,)``.

            names:
                Constraint names in the same order as the cost columns.
        """
        self.measure_constraints = []

        if constraints is None:
            constraints = []

        for constraint_index, constraint_input in enumerate(constraints):
            if isinstance(constraint_input, MeasureConstraint):
                measure_constraint = constraint_input

            elif isinstance(constraint_input, Constraint):
                raise NotImplementedError(
                    "SHERLOCK currently supports MeasureConstraint only. "
                    f"Received {type(constraint_input).__name__} at "
                    f"constraints[{constraint_index}]."
                )

            elif isinstance(constraint_input, dict):
                constraint_dict = dict(constraint_input)

                if "measure_name" in constraint_dict:
                    try:
                        measure_constraint = MeasureConstraint(**constraint_dict)
                    except TypeError as error:
                        raise ValueError(
                            "Could not construct MeasureConstraint from "
                            f"constraints[{constraint_index}]: {error}"
                        ) from error
                else:
                    raise ValueError(
                        f"constraints[{constraint_index}] does not contain "
                        "'measure_name', so its constraint type cannot be "
                        "determined."
                    )

            else:
                raise TypeError(
                    f"constraints[{constraint_index}] must be a dictionary "
                    "or Constraint instance, but received "
                    f"{type(constraint_input).__name__}."
                )

            self.measure_constraints.append(measure_constraint)

        if not self.measure_constraints:
            # No resource constraint. This allows every model to be selected
            # until ensemble_size is reached.
            costs = np.empty((n_models, 0), dtype=float)
            budgets = np.empty(0, dtype=float)
            names: list[str] = []

            return costs, budgets, names

        cost_columns: list[NDArray[np.float64]] = []
        budgets_list: list[float] = []
        names: list[str] = []
        seen_names: set[str] = set()

        for constraint in self.measure_constraints:
            if constraint.threshold_type != "total":
                raise NotImplementedError(
                    f"MeasureConstraint {constraint.measure_name!r} uses "
                    f"threshold_type={constraint.threshold_type!r}. "
                    "The current SHErLoCk implementation supports only "
                    "'total' constraints. A 'mean' constraint requires "
                    "cardinality-dependent feasibility logic."
                )

            if not constraint.below_threshold:
                raise NotImplementedError(
                    f"MeasureConstraint {constraint.measure_name!r} is an "
                    "above-threshold constraint. SHErLoCk's current knapsack "
                    "search supports upper bounds of the form "
                    "sum(costs) <= threshold. Set below_threshold=True for "
                    "resource-budget constraints."
                )

            if constraint.relaxation:
                raise NotImplementedError(
                    f"MeasureConstraint {constraint.measure_name!r} enables "
                    "relaxation. The current search treats constraints as "
                    "hard feasibility conditions and does not apply penalty "
                    "terms."
                )

            measures = np.asarray(
                constraint.measures,
                dtype=float,
            ).reshape(-1)

            if measures.size != n_models:
                raise ValueError(
                    f"MeasureConstraint {constraint.measure_name!r} contains "
                    f"{measures.size} measures, but predictions contain "
                    f"{n_models} learners."
                )

            if np.any(~np.isfinite(measures)):
                invalid_indices = np.flatnonzero(~np.isfinite(measures)).tolist()

                raise ValueError(
                    f"MeasureConstraint {constraint.measure_name!r} contains "
                    "non-finite values at learner indices "
                    f"{invalid_indices}."
                )

            if np.any(measures < 0):
                invalid_indices = np.flatnonzero(measures < 0).tolist()

                raise ValueError(
                    f"MeasureConstraint {constraint.measure_name!r} contains "
                    "negative costs at learner indices "
                    f"{invalid_indices}. Additive knapsack resource costs "
                    "must be non-negative."
                )

            threshold = float(constraint.threshold)

            if not np.isfinite(threshold):
                raise ValueError(
                    f"Threshold for {constraint.measure_name!r} must be finite."
                )

            if threshold < 0:
                raise ValueError(
                    f"Threshold for {constraint.measure_name!r} must be non-negative."
                )

            name = constraint.measure_name

            if name in seen_names:
                raise ValueError(
                    f"Duplicate MeasureConstraint name {name!r}. "
                    "Each constraint needs a unique measure_name."
                )

            seen_names.add(name)
            names.append(name)
            cost_columns.append(measures)
            budgets_list.append(threshold)

        costs = np.column_stack(cost_columns)
        budgets = np.asarray(
            budgets_list,
            dtype=float,
        )

        expected_shape = (
            n_models,
            len(self.measure_constraints),
        )

        if costs.shape != expected_shape:
            raise RuntimeError(
                f"Internal constraint shape error: costs have shape "
                f"{costs.shape}, expected {expected_shape}."
            )

        return costs, budgets, names
