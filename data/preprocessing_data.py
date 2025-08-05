import pickle
import numpy as np

# Load original dataset
with open("data/RML2016.10a_dict.pkl", "rb") as f:
    data = pickle.load(f, encoding="latin-1")

# Set target modulations and SNRs
target_mods = ['BPSK', 'QPSK', 'AM-DSB', 'QAM16']
target_snrs = [-6, 2, 12]
mod_to_idx = {mod: idx for idx, mod in enumerate(target_mods)}

# Collect normalized, flattened samples
x = []
y = []

for mod in target_mods:
    for snr in target_snrs:
        key = (mod, snr)
        if key in data:
            samples = data[key]
            for sample in samples:
                vec = sample.flatten()
                vec = (vec - np.mean(vec)) / np.std(vec)
                x.append(vec)
                y.append(mod_to_idx[mod])

# Convert to numpy arrays
X = np.array(x)
Y = np.array(y)

# Shuffle the dataset
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

# Save for later loading in training script
np.savez("data/processed_rf_data.npz", X=X, Y=Y)
print(f"Saved processed dataset: X shape = {X.shape}, Y shape = {Y.shape}")
