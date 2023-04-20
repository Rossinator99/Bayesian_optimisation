# Bayesian_optimisation
Code that finds optimised hyperparameters using Bayesian optimisation for RNN and CNN that predicts the Hurst exponents for fbm tracks
Code has been tested with python 3.10 and 3.9, and tensorflow 2.12.0

Make sure you pip install scikit-optimize (for the bayesian optimization code) and pip install stochastic

model_finder_RNN.py finds the best model. You will want to save the output file and if you are using a linux machine the command is to run the code will be model_finder_RNN.py > output_model_finder.txt

This save the output as a .txt file. Then you can use model_finder_output_analyser.py to find the 10 best models and their parameters within the file.

Then insert these parameters into makel_maker_RNN.py and build your model.

model_finder_conv.py finds best convolution model. This is slightly different because it will save the best model file as a .h5 file so you don't have to rebuild it. When opening this model you will have to load it differently using a custom scope object but that is explained within the code.

I recommend using google collab to run convolution models as GPUs on there make it much quicker process.
