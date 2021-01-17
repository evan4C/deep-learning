# Understanding how to train for an appropriate number of epochs as we'll explore below is a useful skill.
# To prevent over fitting, the best solution is to use more complete training data.
# The dataset should cover the full range of inputs that the model is expected to handle.
# Additional data may only be useful if it covers new and interesting cases.

# A model trained on more complete data will naturally generalize better.
# When that is no longer possible, the next best solution is to use techniques like regularization.
# These place constraints on the quantity and type of information your model can store.
# If a network can only afford to memorize a small number of patterns,
# the optimization process will force it to focus on the most prominent patterns,
# which have a better chance of generalizing well.

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

from matplotlib import pyplot as plt
import numpy as np
import pathlib
import shutil
import tempfile

logdir = pathlib.Path(tempfile.mkdtemp())/"tensorboard_logs"
shutil.rmtree(logdir, ignore_errors=True)

