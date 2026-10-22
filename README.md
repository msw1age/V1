# V1: early prototype of a surround-suppression model

Early code from my Ph.D. project on surround suppression in primary visual cortex (V1), with Lai-Sang Young at NYU Courant. This repository holds the first prototype: a small conductance-based leaky integrate-and-fire (LIF) network with three cell types, written in Numba, and a neural-network surrogate used to search its parameter space.

The current model is much larger (136,000 neurons, 36 million synapses, short-term plasticity) and runs on the GPU; it is not in this repository. A manuscript is in preparation, and that code will be released with the paper.

## The prototype

- 400 neurons: 300 excitatory, 66 PV and 34 SOM inhibitory cells, with type-specific synaptic strengths and randomized connectivity.
- Conductance-based LIF dynamics with Poisson external drive, integrated at 0.1 ms steps and compiled with Numba.
- Readouts: firing rate and spike-count variability for each cell type.

## Files

| File | Purpose |
|---|---|
| `LIF_sim.py` | The simulator, plus functions that generate training data for the surrogate (directed or random parameter sampling). |
| `LIF Model Numba 3 neuron types.ipynb` | Runs the simulation and makes plots; the easiest place to experiment. |
| `Parameter Selection.ipynb` | Finds parameter sets with the desired network behavior, then perturbs them with noise to build a training set of 30,000+ parameter sets. |
| `LIF DNN Three Neuron Types.ipynb` | Trains a neural network to predict simulation outputs from input parameters. Run `Parameter Selection.ipynb` first. |

This is research code; the notebooks are kept as they were when the prototype was in use.
