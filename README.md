Repository for the results presented in "Considering Control Flow Constructs for Predicting Business Process Outcomes with Deep Learning"

# Content
The repo contains three folders:
* code: contains the python machine learning code to train and evaluate the models
* data: contains both raw and transformed data of the used dataset
* transform: little .NET 4.5 application that was used to transform the data in the ./data/ folder

# Install
To run the python code you need the following:
* Python 2
* Keras
* Backend of your choice (we used CNTK)
* GPU: cuda installation and cuda compatible backend

and the following python packages:
* unicodecsv
* distance
* jellyfish

If you want to use docker, there are ready-to-use images in [chemsorly/keras-cntk](https://hub.docker.com/r/chemsorly/keras-cntk/)

# Run
* Clone repository
* Navigate to ./code/
* run "python aio_numeric_s2s_noplanned.py 1 100 0.1 20 1"

Parameters are as follows:
* Running number (unused int)
* Neurons per layer (int)
* Dropout (double 0-1)
* Patience (int)
* Optimization algorithm (int 1-7)

# Credits
Credits go to [verenich](https://github.com/verenich) whose [work](https://github.com/verenich/ProcessSequencePrediction) was used as base for this project.

# Reference
(tba)
