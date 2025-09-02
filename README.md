The code I have developed for my PhD project involving modeling V1 surround suppression. 

Parameter selection.ipynb generates a large set of sets of input parameters by finding a few sets of parameters within given ranges that have desired properties in the output model, and then adding random noise to create 30,000+ sets of input parameters.

LIF Model Numba 3 neuron types.ipynb contains the code to run the simulation and generate plots. It is the easiest way to tweak the simulation for experimentation without affecting other notebooks.

LIF DNN Three Neuron Types.ipynb contains the code to train a neural network to predict the output of the simulation given a set of input parameters. It requires Parameter selection.ipynb to be run first to generate the training data.

LIF_sim.py contains the actual simulation code, along with code to generate training data for the neural network, either in a directed fashion or a random fashion.

Please note that this project is a WIP and the code is currently being developed by me alone, so there may be formatting issues or extraneous code blocks which will be fixed in later iterations.
