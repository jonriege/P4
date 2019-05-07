**Requirements**

Running the code requires python 3.6 with the following packages installed:
* tensorflow
* matplotlib
* pygame
* numpy

**How to run**

Type the following into the command line:

`python run.py [agent] --train=False --epochs=400 --tick_length=50`

The agent parameter is required, while --train, --epochs and --tick_length
are optional. The default values are shown above.

The agent parameter accepts 4 values:
* `low_g_table` - Low granularity tabular Q-function
* `high_g_table` - High granularity tabular Q-function
* `simple_nn` - Neural network Q-function
* `replay_nn` - Neural network Q-function with experience replay

If `--train=True` the code will start training a new agent of the given type. 
If `--train=False` a pre-trained agent is used. For the high granularity tabular
Q-function convergence can take thousands of epochs, so running the 
pretrained agent is recommended.
