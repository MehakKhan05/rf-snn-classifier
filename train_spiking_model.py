from urllib.request import urlretrieve

import nengo 
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt

import nengo_dl

# Load the processed RF data
data = np.load("data/processed_rf_data.npz")
X, Y = data["X"], data["Y"]


Y = tf.keras.utils.to_categorical(Y, num_classes=4)

# Reshape X to be suitable for repeated input over time (N, 1, D)
X = X[:, None, :]

# Define constants
n_steps = 30       # simulation time steps per sample
n_input = X.shape[-1]
n_hidden1 = 512
n_hidden2 = 128
n_classes = 4

# Define the Nengo model
with nengo.Network(seed=0) as net:
    # Set a global neuron default
    nengo_dl.configure_settings(trainable=True)

    # Input node (present constant vector for all time steps)
    inp = nengo.Node(np.zeros(n_input))

    # First LIF layer
    x = nengo.Ensemble(n_neurons=n_hidden1, dimensions=1, neuron_type=nengo.LIF())
    conn0 = nengo.Connection(inp, x.neurons, transform=np.random.randn(n_hidden1, n_input), synapse=None)

    # Second LIF layer
    y = nengo.Ensemble(n_neurons=n_hidden2, dimensions=1, neuron_type=nengo.LIF())
    conn1 = nengo.Connection(x.neurons, y.neurons, transform=np.random.randn(n_hidden2, n_hidden1), synapse=None)

    # Output layer (dense)
    out = nengo.Node(size_in=n_classes)
    conn2 = nengo.Connection(y.neurons, out, transform=np.random.randn(n_classes, n_hidden2), synapse=None)

    # Probes
    out_p = nengo.Probe(out, synapse=0.01)

# Wrap with NengoDL simulator
minibatch_size = 200
with nengo_dl.Simulator(net, minibatch_size=minibatch_size, unroll_simulation=5) as sim:
    # Repeat input across time steps
    train_data = {inp: np.tile(X, (1, n_steps, 1))}
    train_targets = {out_p: np.tile(Y[:, None, :], (1, n_steps, 1))}


    # Compile and train
    sim.compile(loss={out_p: tf.losses.CategoricalCrossentropy(from_logits=True)},
                optimizer=tf.optimizers.Adam(1e-3),
                metrics=["accuracy"])
    sim.fit(train_data, train_targets, epochs=5)

    # Save model
    sim.save_params("./saved_model")

    # Evaluate
    test_output = sim.predict(train_data)[out_p]
    predictions = np.mean(test_output, axis=1)
    predicted_classes = np.argmax(predictions, axis=-1)

# Output a few prediction results
import pandas as pd

df = pd.DataFrame({
    "True Label": np.argmax(Y, axis=-1),
    "Predicted Label": predicted_classes
})
# tools.display_dataframe_to_user(name="Prediction Results", dataframe=df.head(30))
