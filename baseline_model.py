import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# --- Load Data ---
data = np.load("data/processed_rf_data.npz")
X, Y = data["X"], data["Y"]

# One-hot encode labels
Y = tf.keras.utils.to_categorical(Y, num_classes=4)

# Split dataset
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# --- Build DNN Model ---
model = models.Sequential([
    layers.Input(shape=(X.shape[1],)),        # 256-length input
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(4, activation='softmax')     # 4 classes
])

# --- Compile ---
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- Train ---
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=20,
    batch_size=64,
    verbose=1
)

# --- Evaluate ---
test_loss, test_acc = model.evaluate(X_val, Y_val)
print(f"\n🧪 DNN Validation Accuracy: {test_acc * 100:.2f}%")
