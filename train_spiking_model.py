import numpy as np
import tensorflow as tf
import nengo
import nengo_dl
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# --- Load and preprocess data ---
data = np.load("data/processed_rf_data.npz")
X, Y = data["X"], data["Y"]
Y = tf.keras.utils.to_categorical(Y, num_classes=4)

# Increase simulation duration to improve spike integration
n_steps = 90  # extended time window
X = np.tile(X[:, None, :], (1, n_steps, 1))                # Shape: (batch, n_steps, features)
Y = np.tile(Y[:, None, :], (1, n_steps, 1))                # Shape: (batch, n_steps, classes)

# --- Define model parameters ---
n_input = X.shape[-1]
n_hidden1 = 512
n_hidden2 = 128
n_classes = 4
minibatch_size = 200

# --- Construct Nengo model with training-optimized LIF setup ---
with nengo.Network(seed=0) as net:
    neuron_type = nengo.LIF(amplitude=0.01)
    net.config[nengo.Ensemble].neuron_type = neuron_type
    net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
    net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
    net.config[nengo.Connection].synapse = None
    nengo_dl.configure_settings(stateful=False)

    # Input node with Gaussian noise
    inp = nengo.Node(lambda t: np.zeros(n_input) + np.random.normal(0, 0.01, n_input))

    # First hidden layer
    x = nengo_dl.Layer(neuron_type)(inp)

    # Second hidden layer
    y = nengo_dl.Layer(neuron_type)(x)

    # Optional third hidden layer
    z = nengo_dl.Layer(neuron_type)(y)

    # Dense output layer
    out = nengo_dl.Layer(tf.keras.layers.Dense(units=n_classes))(z)

    # Probes
    out_p = nengo.Probe(out)
    out_p_filt = nengo.Probe(out, synapse=0.01)

# --- Train using surrogate gradient approach ---
with nengo_dl.Simulator(net, minibatch_size=minibatch_size, unroll_simulation=5) as sim:

    # Custom temporal weighted cross-entropy loss
    def weighted_cce(true, pred):
        # true, pred shape: (batch, timesteps, classes)
        time_weights = tf.linspace(0.1, 1.0, num=tf.shape(pred)[1])  # shape: (timesteps,)
        time_weights = tf.reshape(time_weights, (1, -1, 1))          # shape: (1, timesteps, 1)

        cce = tf.keras.losses.categorical_crossentropy(true, pred, from_logits=True)  # shape: (batch, timesteps)
        cce = tf.expand_dims(cce, axis=-1)                                            # shape: (batch, timesteps, 1)

        weighted_loss = cce * time_weights                                            # shape: (batch, timesteps, 1)
        return tf.reduce_mean(weighted_loss)


    sim.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss={out_p: weighted_cce},
        metrics=["accuracy"]
    )

    sim.fit({inp: X}, {out_p: Y}, epochs=30)

    # Switch to true spiking LIF for inference
    sim.freeze_params(net)
    net.config[nengo.Ensemble].neuron_type = nengo.LIF()
    test_output = sim.predict({inp: X})[out_p_filt]

# --- Evaluate ---
predictions = np.mean(test_output, axis=1)
predicted_classes = np.argmax(predictions, axis=-1)
true_classes = np.argmax(Y[:, 0, :], axis=-1)

cm = confusion_matrix(true_classes, predicted_classes)
acc = accuracy_score(true_classes, predicted_classes)

fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3])
disp.plot(ax=ax, cmap="Blues", values_format='d')
plt.title(f"Spiking SNN Confusion Matrix (Acc = {acc:.2f})")
plt.tight_layout()
plt.show()