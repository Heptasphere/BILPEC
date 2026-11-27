# BILPEC

Git for the code in the article "Constrained Ensemble Pruning via Binary Integer Linear Programming".

This contains the BILPEC/BILPEC\* implementation and baselines implementations.

This requires uv, to use the python environment:

```{bash}
uv venv
uv sync
source .venv/bin/activate
```

Then, use the [example](example.ipynb) to see how to use the models.

By default, only the [CBC](https://github.com/coin-or/Cbc) solver is installed. To use Gurobi 13, the solver used for article experiments, you need to install the [Gurobi solver](https://www.gurobi.com/product/download-center) and its python package, `gurobipy`, separately.
