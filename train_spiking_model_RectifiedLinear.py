import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import nengo
import nengo_dl

# === Load Data ===
data = np.load("data/processed_rf_data.npz")
X, Y = data["X"], data["Y"]
Y = tf.keras.utils.to_categorical(Y, num_classes=4)

# === Reshape for temporal input ===
n_steps = 30
X_reshaped = np.tile(X[:, None, :], (1, n_steps, 1))
Y_reshaped = np.tile(Y[:, None, :], (1, n_steps, 1))

# === Model Params ===
n_input = X.shape[-1]
n_hidden1 = 512
n_hidden2 = 128
n_classes = 4
minibatch_size = 200

# === Build ANN with ReLU ===
with nengo.Network(seed=0) as ann_net:
    nengo_dl.configure_settings(trainable=True)

    inp = nengo.Node(np.zeros(n_input))

    # Use tf.keras layers with NengoDL
    x = nengo_dl.Layer(tf.keras.layers.Dense(units=n_hidden1))(inp)
    x = nengo_dl.Layer(tf.keras.layers.ReLU())(x)

    y = nengo_dl.Layer(tf.keras.layers.Dense(units=n_hidden2))(x)
    y = nengo_dl.Layer(tf.keras.layers.ReLU())(y)

    out = nengo_dl.Layer(tf.keras.layers.Dense(units=n_classes))(y)

    out_p = nengo.Probe(out)

# === Train ANN ===
with nengo_dl.Simulator(ann_net, minibatch_size=minibatch_size, unroll_simulation=5) as sim:
    sim.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={out_p: tf.keras.losses.CategoricalCrossentropy(from_logits=True)},
        metrics=["accuracy"]
    )

    sim.fit(
        {inp: X_reshaped},
        {out_p: Y_reshaped},
        epochs=30
    )

    sim.save_params("saved_ann_params")

# === Convert to Spiking Model (LIF neurons) ===
with nengo.Network(seed=0) as snn_net:
    nengo_dl.configure_settings(trainable=True)

    inp = nengo.Node(np.zeros(n_input))

    x = nengo_dl.Layer(nengo.LIF())(inp, size_in=n_input, size_out=n_hidden1)
    y = nengo_dl.Layer(nengo.LIF())(x, size_in=n_hidden1, size_out=n_hidden2)
    out = nengo_dl.Layer(tf.keras.layers.Dense(units=n_classes))(y)

    out_p = nengo.Probe(out, synapse=0.01)

# === Load weights and evaluate SNN ===
with nengo_dl.Simulator(snn_net, minibatch_size=minibatch_size, unroll_simulation=5) as sim:
    sim.load_params("saved_ann_params")

    output = sim.predict({inp: X_reshaped})[out_p]
    predictions = np.mean(output, axis=1)

    predicted_classes = np.argmax(predictions, axis=-1)
    true_classes = np.argmax(Y, axis=-1)

    acc = accuracy_score(true_classes, predicted_classes)
    cm = confusion_matrix(true_classes, predicted_classes)

# === Plot Confusion Matrix ===
fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3])
disp.plot(ax=ax, cmap="Blues", values_format='d')
plt.title(f"Spiking SNN Confusion Matrix (Acc = {acc:.2f})")
plt.tight_layout()
plt.show()
