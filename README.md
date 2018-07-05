# ![Image](https://www.knime.com/files/knime_logo_github_40x40_4layers.png) KNIME® Deep Learning - TensorFlow Integration

This repository contains the plugins for the KNIME TensorFlow Integration which contains a set of KNIME nodes for executing TensorFlow models in KNIME.

## Overview

The extension contains the following nodes:

* The TensorFlow Network Reader node for reading [TensorFlow SavedModels](https://www.tensorflow.org/programmers_guide/saved_model#save_and_restore_models).
* The TensorFlow Network Writer node for writing [TensorFlow SavedModels](https://www.tensorflow.org/programmers_guide/saved_model#save_and_restore_models).
* The Keras to TensorFlow Network Converter node for converting Keras models to TensorFlow.

Additionally, the DL Python nodes provided in [KNIME Deep Learning](https://www.knime.com/deeplearning) can be used to create, edit, execute and train models with user-supplied Python scripts.

![Workflow Screenshot](https://files.knime.com/sites/default/files/KNIME-TF-Screenshot.png)

## TensorFlow Python Bindings

The KNIME TensorFlow Integration can be used with the _DL Python Network_ nodes which allow you to create, train and modify a TensorFlow model using the powerful TensorFlow Python API.

The KNIME TensorFlow Integration provides a `TFModel` object to Python whenever a model is loaded into Python and requires you to set a `TFModel` object whenever a model should be returned to KNIME for further usage.

### Required Python Packages

* `tensorflow` or `tensorflow-gpu` (version: 1.8.0)

Note that newer or older versions can cause problems because the TensorFlow for Java version used is 1.8.0.

### Example

__Create a TensorFlow model:__

```
import tensorflow as tf
from TFModel import TFModel

# Create a graph
graph = tf.Graph()

# Set the graph as default -> Create every tensor in this graph
with graph.as_default():

    # Create an input tensor
    x = tf.placeholder(tf.float32, shape=(None,4))

    # define the graph...

    # Create an output tensor
    y = tf.nn.softmax(last_layer)

# Create the output network
output_network = TFModel(inputs={ 'input': x }, outputs={ 'output': y }, graph=graph)
```

__Use/Train/Edit a TensorFlow model:__

```
import tensorflow as tf
from TFModel import TFModel

# Use the session from the TFModel
with input_network.session as sess:

    # Get the input tensor
    x = input_network.inputs['input']

    # Get the output tensor
    y = input_network.outputs['output']

    # Use/Train/Edit the model...

    # Create the output network
    # NOTE: The whole session is passed to the model (to save the variables)
    #       This needs to be called before the session is closed
    output_network = TFModel(inputs={ 'input': x }, outputs={ 'output': y }, session=sess)
```


## Example Workflows

You can download the example workflows from the KNIME public example server (See [here how to connect...](https://www.knime.org/example-workflows)) or from the [KNIME node guide](https://www.knime.com/nodeguide/analytics/deep-learning).

## Development Notes

You can find instructions on how to work with our code or develop extensions for
KNIME Analytics Platform in the _knime-sdk-setup_ repository
on [BitBucket](https://bitbucket.org/KNIME/knime-sdk-setup)
or [GitHub](http://github.com/knime/knime-sdk-setup).

## Join the Community!

* [KNIME Forum](https://tech.knime.org/forum)
