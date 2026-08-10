# Azul: Stained Glass stochastic strategy model

This repository contains the reproducible simulation and policy-search code for
the simplified finite-horizon model of *Azul: Stained Glass of Sintra*.

## Model

- Eight ordered windows; each window has two sides.
- Each side requires three successful fills.
- Horizon: 45 actions.
- A fill action is available with probability 0.8, 0.6, or 0.4 when the
  current side has respectively 0, 1, or 2 fills.
- The random action mask is observed before the action is chosen.
- Moving the worker back to the left consumes one complete action.

## Files

- `azul_policy_search.py`: simulator, first-order and hierarchical
  second-order policy features, and validated policy parameters.
- `azul_second_order_search.py`: interaction screening and second-order joint
  search.
- `azul_reset_rollout.py`: paired-rollout training and validation of the
  learned reset policy.
- `azul_stochastic_model.tex`: Chinese modeling note and experiment log.

## Reproduce the searches

```bash
python azul_second_order_search.py screen --episodes 16000 --seed 20260830
```

```bash
python azul_reset_rollout.py \
  --collection-episodes 8000 \
  --states 1000 \
  --rollouts 192 \
  --selection-episodes 30000 \
  --final-episodes 100000 \
  --final-thresholds 0.1,0.2,0.35 \
  --seed 20260842
```

The recommended score/reset trade-off uses the learned reset threshold 0.2.
On the untouched 100,000-trajectory test set it reduced mean total resets from
11.2830 to 10.6812 while increasing mean score from 45.2860 to 45.4450.

## Referencing the code from Overleaf

An Overleaf build cannot reliably fetch a remote Python file during LaTeX
compilation. Use a permanent repository commit link for reproducibility. If the
source code should also appear in the PDF, place the corresponding `.py` file
inside the Overleaf project and include it with `listings`:

```tex
\usepackage{listings}

\lstinputlisting[
  language=Python,
  caption={Stochastic policy simulator},
  label={lst:policy-simulator}
]{azul_policy_search.py}
```

The paper can link to the exact repository commit using `\href{...}{source
code}`. A commit permalink is preferable to a branch URL because it keeps the
reported experiment reproducible.

