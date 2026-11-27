"""Ensemble Method designed for tabarena."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, override

import numpy as np
import pulp
from autogluon.core.metrics import log_loss
from numpy._typing._array_like import NDArray

from bilpec._base import AbstractEnsemble
from bilpec.constraint import (
    Constraint,
    LabelConstraint,
    MeasureConstraint,
    SampleSetConstraint,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from autogluon.core.metrics import Scorer
    from numpy.typing import NDArray


class AbstractBILPEC(AbstractEnsemble, ABC):
    @override
    def __init__(
        self,
        ensemble_size: int,
        problem_type: str,
        metric: Scorer,
        solver: str = "CBC",
        solver_time_limit: float = 20,
        solver_relative_gap: float = 0.0,
        subset_size: int = 1000,
        subset_choice: str = "random",
        constraints: None | list = None,
        seed: int = 42,
    ) -> None:
        super().__init__(ensemble_size, problem_type, metric)
        self.solver: str = solver
        self.solver_time_limit: float = solver_time_limit
        self.solver_relative_gap: float = solver_relative_gap
        self.subset_size: int = subset_size
        self.subset_choice: str = subset_choice
        self.subset_indices: NDArray = np.array([])
        self.seed: int = 42

        # We constrain our problem to have at least one nonzero weight.
        minimum_ensemble_size_constraint = MeasureConstraint(
            measure_name="ensemble_size",
            measures=np.ones(ensemble_size),
            threshold=1,
            threshold_type="total",
            below_threshold=False,
        )

        self.measure_constraints: list[MeasureConstraint] = [
            minimum_ensemble_size_constraint
        ]
        self.label_constraints: list[LabelConstraint] = []
        self.sample_constraints: list[SampleSetConstraint] = []

        self._parse_constraints(constraints)
        self.problem_stats: pulp.LpSolveStats | None = None

    def _parse_constraints(self, constraints) -> None:
        if constraints is None:
            constraints = []

        for constraint in constraints:
            if isinstance(constraint, MeasureConstraint):
                self.measure_constraints.append(constraint)
            elif isinstance(constraint, LabelConstraint):
                self.label_constraints.append(constraint)
            elif isinstance(constraint, SampleSetConstraint):
                self.sample_constraints.append(constraint)
            elif isinstance(constraint, Constraint):
                raise TypeError(f"Type of constraint not handled: {type(constraint)}")
            elif "measure_name" in constraint:
                measure_constraint: MeasureConstraint = MeasureConstraint(**constraint)
                self.measure_constraints.append(measure_constraint)
            elif "label_id" in constraint:
                label_constraint: LabelConstraint = LabelConstraint(**constraint)
                self.label_constraints.append(label_constraint)
            elif "parser" in constraint:
                sample_constraint: SampleSetConstraint = SampleSetConstraint(
                    **constraint
                )
                self.sample_constraints.append(sample_constraint)
            else:
                warnings.warn(
                    f"The following constraint couldn't be parsed: {constraint}"
                )

    def _get_true_class_predictions(self, predictions, labels) -> NDArray:

        if predictions.ndim == 2:
            return np.where(
                labels[np.newaxis, :] == 1,
                predictions,
                1.0 - predictions,
            )
        elif predictions.ndim == 3:
            return predictions[:, np.arange(len(labels)), labels]
        raise ValueError(
            f"Predictions of shape {predictions.shape} not taken into account."
        )

    def _get_solver(self, solver_kwargs):
        match self.solver:
            case "SCIP":
                return pulp.SCIP_PY(**solver_kwargs)
            case "CBC":
                return pulp.COIN_CMD(**solver_kwargs)
            case "GUROBI":
                # GUROBI requires installation and a license
                return pulp.GUROBI(**solver_kwargs)
        raise ValueError(
            f"The available solvers are SCIP, CBC and GUROBI, not {self.solver}"
        )

    @override
    def fit(
        self, predictions: NDArray[float], labels: NDArray[int], **fit_params
    ) -> None:

        # Crashes randomly if garbage collection is enabled.

        problem, weight_vars = self._create_problem(
            predictions,
            labels=labels,
        )

        # Add constraints linked to measures
        problem = self._add_measure_constraints(problem, weight_vars)

        solver_kwargs = {
            "msg": False,
            "timeLimit": self.solver_time_limit,
            "gapRel": self.solver_relative_gap,
        }

        solver = self._get_solver(solver_kwargs=solver_kwargs)

        self.problem_stats = problem.solve(solver=solver)

        self.weights_ = np.array([var.value() for _, var in weight_vars.items()])

        # Fallback: in case of failure, use all base learners.
        if not self.problem_stats.has_solution or np.sum(np.abs(self.weights_)) == 0:
            warnings.warn(
                f"No solutions were found that respected all constraints for problem with sample_size: {len(labels)}. Defaulting to Soft Voting."
            )
            self.weights_ = np.ones(self.ensemble_size)

        # We delete the problem after getting its result.
        del problem
        del solver

    def _get_filtered_sample_indices(
        self,
        predictions: NDArray,
        labels: NDArray,
    ) -> NDArray:
        """Return non-unanimous samples filtered samples. Keeps info on filtered indices.

        Returns
        -------
        kept_indices
            Indices where base learners do not all predict the same class.
        """
        if predictions.ndim == 3:
            predicted_classes = predictions.argmax(axis=2)
        elif predictions.ndim == 2:
            predicted_classes = predictions > 0.5
            predictions = np.stack([1 - predictions, predictions], axis=2)
        else:
            raise ValueError(
                f"Expected 2D or 3D predictions, got shape {predictions.shape}."
            )

        # For each sample, do all models predict the same class?
        unanimous = np.all(
            predicted_classes == predicted_classes[0],
            axis=0,
        )

        # The predicted class is well-defined for unanimous samples.
        # TODO: Check if this doesn't fail for binary.
        correct = predicted_classes[0] == labels

        self.filtered_label_info = {
            label: {
                "correct": int(np.sum(unanimous & correct & (labels == label))),
                "incorrect": int(np.sum(unanimous & ~correct & (labels == label))),
                "mean_probability": (
                    float(
                        predictions[:, unanimous & (labels == label), int(label)].mean()
                    )
                    if np.any(unanimous & (labels == label))
                    else 0.0
                ),
            }
            for label in np.unique(labels)
        }
        self.n_total_samples = len(labels)

        kept_indices = np.flatnonzero(~unanimous)

        return kept_indices

    def _get_least_correct_sample_indices(
        self,
        predictions: NDArray,
        labels: NDArray,
    ) -> NDArray:
        """Get predictions that are the least correctly predicted while removing predicitons that are always or never predicted correctly.

        Parameters
        ----------
        predictions : NDArray
            Results of models on samples.
            Either (n_models, n_samples, n_classes) for multiclass results or (n_models, n_samples) for binary results.
        labels : NDArray
            True labels for problem.

        Returns
        -------
        NDArray[int]
            Array of all indices that respect the given conditions.
        """

        filtered_indices = set(self._get_filtered_sample_indices(predictions, labels))

        true_predictions = self._get_true_class_predictions(
            predictions=predictions,
            labels=labels,
        )

        true_predictions_sum = true_predictions.sum(axis=0)

        return np.array(
            [ind for ind in np.argsort(true_predictions_sum) if ind in filtered_indices]
        )[: self.subset_size]

    def _get_subset_indices(
        self,
        predictions: NDArray,
        labels: NDArray,
    ) -> NDArray:

        if len(predictions) == 0:
            raise ValueError(f"There are no predictions: {predictions}")

        filtered_indices = self._get_filtered_sample_indices(predictions, labels)

        # If there are no index left, return the index [0]
        if len(filtered_indices) == 0:
            self.subset_size = 1
            return np.array([0])

        if self.subset_size >= len(filtered_indices):
            self.subset_size = len(filtered_indices)
            return filtered_indices

        # Choose samples among the filtered ones.
        match self.subset_choice:
            # Random samples
            case "random":
                return np.random.default_rng(self.seed).choice(
                    filtered_indices,
                    size=self.subset_size,
                    replace=False,
                )
            # Samples which are the least correctly predicted
            # Currently disabled for our experiments.
            # case "least_correct":
            #     return self._get_least_correct_sample_indices(predictions, labels)
            case _:
                raise NotImplementedError(
                    f"Choosing subset indices using {self.subset_choice} hasn't been implemented."
                )

    @abstractmethod
    def _create_problem(
        self, predictions: NDArray, labels: NDArray
    ) -> tuple[pulp.LpProblem, dict[int, pulp.LpVariable]]:
        raise NotImplementedError

    def _add_sample_constraints(
        self, problem, weight_vars, predictions, labels
    ) -> None:
        pass

    def _add_error_rate_label_constraint(
        self,
        problem: pulp.lpProblem,
        constraint: LabelConstraint,
        sample_failure_vars: None | dict[int, pulp.LpVariable],
        predictions: NDArray,
        labels: NDArray,
        label_indices: NDArray,
        label: int,
    ) -> None:
        raise NotImplementedError(
            "Label constraints of type error rate are only implemented for BILPEC."
        )

    def _add_mean_probability_label_constraint(
        self,
        problem: pulp.lpProblem,
        constraint: LabelConstraint,
        weight_vars: None | dict[int, pulp.LpVariable],
        true_predictions: NDArray,
        label_indices: NDArray,
        label: int,
    ) -> None:
        raise NotImplementedError(
            "Label constraints of type mean probability are only implemented for BILPEC."
        )

    def _add_label_constraints(
        self,
        problem: pulp.LpProblem,
        weight_vars: dict[int, pulp.LpVariable],
        predictions: NDArray,
        labels: NDArray,
        sample_failure_vars: dict[int, pulp.LpVariable] | None = None,
    ) -> None:

        predictions = predictions[:, self.subset_indices]
        labels = labels[self.subset_indices]

        for constraint in self.label_constraints:
            label_indices = np.where(labels == constraint.label_id)[0]

            match constraint.threshold_type:
                case "error_rate":
                    # This only works in BILPEC
                    self._add_error_rate_label_constraint(
                        problem=problem,
                        constraint=constraint,
                        sample_failure_vars=sample_failure_vars,
                        weight_vars=weight_vars,
                        predictions=predictions,
                        labels=labels,
                        label_indices=label_indices,
                        label=constraint.label_id,
                    )
                case "probability":
                    true_predictions = self._get_true_class_predictions(
                        predictions, labels
                    )
                    self._add_mean_probability_label_constraint(
                        problem=problem,
                        constraint=constraint,
                        weight_vars=weight_vars,
                        true_predictions=true_predictions,
                        label_indices=label_indices,
                        label=constraint.label_id,
                    )
                case _:
                    raise ValueError(
                        f"Threshold of type {constraint.threshold_type} isn't handled yet."
                    )

    def _add_measure_constraints(
        self, problem: pulp.LpProblem, weight_vars: dict[int, pulp.LpVariable]
    ) -> pulp.LpProblem:
        """Add measure constraints to problem.

        Parameters
        ----------
        problem : pulp.LpProblem
            Linear programming problem.
        weights : dict[int, pulp.LpVariable]
            Weight of each base learner.

        Returns
        -------
        pulp.LpProblem
            Linear programming problem with constraints.
        """

        n_models = len(weight_vars)

        for constraint in self.measure_constraints:
            match constraint.threshold_type:
                case "total":
                    vector = [
                        constraint.measures[i] * weight_vars[i] for i in range(n_models)
                    ]
                    final_threshold = constraint.threshold
                case "mean":
                    vector = [
                        (constraint.measures[i] - constraint.threshold) * weight_vars[i]
                        for i in range(n_models)
                    ]
                    final_threshold = 0
                case _:
                    raise ValueError(
                        f"Threshold of type {constraint.threshold_type} isn't handled yet."
                    )

            if constraint.below_threshold:
                new_constraint = pulp.lpSum(vector=vector) <= final_threshold
            else:
                new_constraint = pulp.lpSum(vector=vector) >= final_threshold

            constraint_name = f"{constraint.measure_name}_{constraint.threshold_type}_{'below' if constraint.below_threshold else 'above'}_{constraint.threshold}"

            problem.addConstraint(constraint=new_constraint, name=constraint_name)

        return problem

    def predict_proba(self, predictions):
        return self.weight_pred_probas(predictions, weights=self.weights_)

    @staticmethod
    def weight_pred_probas(pred_probas, weights):
        # preds_norm = [
        #     pred * weight for pred, weight in zip(pred_probas, weights) if weight != 0
        # ]
        # preds_ensemble = np.mean(preds_norm, axis=0)
        preds_ensemble = np.average(pred_probas, axis=0, weights=weights)
        return preds_ensemble

    def is_feasible(self) -> bool:
        """Check if problem is feasible after fitting to the problem.

        Returns
        -------
        bool
            True if feasible. False if infeasible.

        Raises
        ------
        RuntimeError
            Called if fit isn't called before this.
        ValueError
            Called if no solution was found, but in part due to limits defined by user.
        """
        if self.problem_stats is None:
            raise RuntimeError("fit should be called before is_feasible.")

        if self.problem_stats.status == pulp.LpSolveStatus.Infeasible:
            return False
        elif self.problem_stats.has_solution:
            return True
        else:
            raise ValueError(
                f"No solution was found but it may be due to bad parameters.\nStats: {self.problem_stats!s}"
            )


class BILPEC(AbstractBILPEC):
    def __init__(
        self,
        ensemble_size: int,
        problem_type: str,
        metric: Scorer = log_loss,
        fixed_ensemble_size: int = 20,
        solver: str = "CBC",
        solver_time_limit: float = 20,
        solver_relative_gap: float = 0.0,
        subset_choice: str = "random",
        constraints: None | list = None,
    ) -> None:
        super().__init__(
            ensemble_size=ensemble_size,
            problem_type=problem_type,
            metric=metric,
            solver=solver,
            solver_time_limit=solver_time_limit,
            solver_relative_gap=solver_relative_gap,
            subset_size=10**15,  # Infinite
            subset_choice=subset_choice,
            constraints=constraints,
        )
        self.ensemble_size: int = ensemble_size
        self.fixed_ensemble_size = fixed_ensemble_size

        # We constrain our problem to have a fixed size
        self._update_size_constraints()

    def _update_size_constraints(self):

        self.measure_constraints = [
            el for el in self.measure_constraints if el.measure_name != "ensemble_size"
        ]
        measure_constraint_min = MeasureConstraint(
            measure_name="ensemble_size",
            measures=np.ones(self.ensemble_size),
            threshold=self.fixed_ensemble_size,
            threshold_type="total",
            below_threshold=False,
        )
        measure_constraint_max = MeasureConstraint(
            measure_name="ensemble_size",
            measures=np.ones(self.ensemble_size),
            threshold=self.fixed_ensemble_size,
            threshold_type="total",
            below_threshold=True,
        )
        self.measure_constraints.append(measure_constraint_min)
        self.measure_constraints.append(measure_constraint_max)

    @override
    def _create_problem(self, predictions: NDArray, labels: NDArray):
        if predictions.ndim == 3:
            (n_models, n_samples, _) = predictions.shape
        else:
            (n_models, n_samples) = predictions.shape

        self.subset_size = min(n_samples, self.subset_size)

        self.subset_indices = self._get_subset_indices(predictions, labels)

        problem: pulp.LpProblem = pulp.LpProblem(
            name="EnsembleWeights", sense=pulp.LpMaximize
        )

        weight_vars = problem.add_variable_dicts(
            name="base_learner_weights",
            indices=range(n_models),
            cat=pulp.LpBinary,
        )

        self._set_objective(
            problem=problem,
            weight_vars=weight_vars,
            predictions=predictions,
            labels=labels,
        )

        self._add_label_constraints(
            problem=problem,
            weight_vars=weight_vars,
            predictions=predictions,
            labels=labels,
        )

        return problem, weight_vars

    def _set_objective(
        self,
        problem: pulp.lpProblem,
        weight_vars,
        predictions: NDArray,
        labels: NDArray,
    ) -> None:
        n_models = predictions.shape[0]

        true_predictions = self._get_true_class_predictions(
            predictions=predictions,
            labels=labels,
        )

        # Truth prediction for soft voting: svt
        # We just average over all
        soft_voting_true_predictions = true_predictions.mean(axis=0)

        # For each i, we divide by svt and then sum over i
        objective_coefficients = (true_predictions / soft_voting_true_predictions).sum(
            axis=1
        )

        problem.setObjective(
            pulp.lpDot(
                objective_coefficients,
                [weight_vars[i] for i in range(n_models)],
            )
        )

    @override
    def _add_error_rate_label_constraint(
        self,
        problem: pulp.lpProblem,
        constraint: LabelConstraint,
        sample_failure_vars: None | dict[int, pulp.LpVariable],
        weight_vars: dict[int, pulp.LpVariable],
        predictions: NDArray,
        labels: NDArray,
        label_indices: NDArray,
        label: int,
    ) -> None:

        if sample_failure_vars is not None:
            raise ValueError(
                "In BILPEC, sample failure variables are supposed to be created, not given. Given variables:",
                sample_failure_vars,
            )

        n_samples = min(len(label_indices), constraint.sample_size)
        if n_samples == 0:
            warnings.warn(
                f"Can't create label constraint as there are {n_samples} labels of class {label}."
            )
            return

        sample_failure_vars: dict[int, pulp.LpVariable] = problem.add_variable_dicts(
            name="sample_failure",
            indices=range(n_samples),
            cat=pulp.LpBinary,
        )

        n_models = predictions.shape[0]

        true_predictions = self._get_true_class_predictions(
            predictions=predictions,
            labels=labels,
        )

        epsilon = 10 ** (-6)

        # Add right prediction constraint for each variable
        if predictions.ndim == 2:
            n_classes = 2
            for ind, j in enumerate(label_indices[:n_samples]):
                # We create the bare minimum constraint with 1 / n_classes as a threshold.
                problem.addConstraint(
                    constraint=pulp.lpSum(
                        vector=[
                            (true_predictions[i][j] - 0.5) * weight_vars[i]
                            for i in range(n_models)
                        ]
                    )
                    >= epsilon - n_models * sample_failure_vars[ind],
                    name=f"RightlyPredictingSample{ind}",
                )
        elif predictions.ndim == 3:
            n_classes = predictions.shape[2]
            for ind, j in enumerate(label_indices[:n_samples]):
                for k in range(n_classes):
                    if k == int(label):
                        continue
                    problem.addConstraint(
                        constraint=pulp.lpSum(
                            vector=[
                                (true_predictions[i][j] - predictions[i][j][k])
                                * weight_vars[i]
                                for i in range(n_models)
                            ]
                        )
                        >= epsilon - 2 * n_models * sample_failure_vars[ind],
                        name=f"RightlyEstimatingClass{k}ForSample{ind}",
                    )
        else:
            raise ValueError(
                f"Predictions has {predictions.ndim} dimensions.\n{predictions}"
            )

        vector = list(sample_failure_vars.values())

        constraint_name: str = f"Label_{constraint.label_id}_error_rate_{'below' if constraint.below_threshold else 'above'}_{constraint.threshold}"

        if constraint.below_threshold:
            problem.addConstraint(
                constraint=pulp.lpSum(vector) / len(vector) <= constraint.threshold,
                name=constraint_name,
            )
        else:
            problem.addConstraint(
                constraint=pulp.lpSum(vector) / len(vector) >= constraint.threshold,
                name=constraint_name,
            )


class BILPECStar(BILPEC):
    """Recursive BILPEC that reduces S every time it fails conditions until S = 1."""

    @override
    def fit(
        self, predictions: NDArray[float], labels: NDArray[int], **fit_params
    ) -> None:

        # Crashes randomly if garbage collection is enabled.

        problem, weight_vars = self._create_problem(
            predictions,
            labels=labels,
        )

        # Add constraints linked to measures
        problem = self._add_measure_constraints(problem, weight_vars)

        solver_kwargs = {
            "msg": False,
            "timeLimit": self.solver_time_limit,
            "gapRel": self.solver_relative_gap,
        }

        solver = self._get_solver(solver_kwargs=solver_kwargs)

        self.problem_stats = problem.solve(solver=solver)

        found_weights = np.array([var.value() for _, var in weight_vars.items()])

        # Fallback: in case of failure, use all base learners.
        if not self.problem_stats.has_solution or np.sum(np.abs(self.weights_)) == 0:
            warnings.warn(
                f"No solutions were found that respected all constraints for problem with sample_size: {len(labels)}. Defaulting to Soft Voting."
            )
            self.weights_ = np.ones(self.ensemble_size)

        if self.problem_stats.status != pulp.LpSolveStatus.Optimal:
            warnings.warn(
                f"Not the optimal function for problem with sample_size: {len(labels)}:\n{self.problem_stats!s}"
            )

        no_solution: bool = (not self.problem_stats.has_solution) or np.sum(
            np.abs(found_weights)
        ) == 0

        if no_solution:
            if self.fixed_ensemble_size == 1:
                warnings.warn(
                    "No single base learner respects the constraints. Failing.\nCurrent status: {self.problem_stats!s}"
                )
                self.weights_ = np.ones(self.ensemble_size)
                return

            # Fallback: in case of failure, use all base learners.
            warnings.warn(
                f"No solutions were found that respected all constraints for problem with fixed ensemble size: {self.fixed_ensemble_size}.\nCurrent status: {self.problem_stats!s}\nTrying with {self.fixed_ensemble_size - 1}."
            )
            self.fixed_ensemble_size -= 1
            self._update_size_constraints()
            return self.fit(predictions=predictions, labels=labels, **fit_params)

        self.weights_ = found_weights

        # We delete the problem after getting its result.
        del problem
        del solver
