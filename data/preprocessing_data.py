import pickle
import numpy as np

with open("data/RML2016.10a_dict.pkl", "rb") as f:
    data = pickle.load(f, encoding="latin-1") # required due to Python2 encoding

    # Exploring Available Keys
    # The dataset is a dictionary: each key is {modulation_type, snr), and each value is a set of signal smaples}
    keys = list(data.keys())
    print(keys[:10])  # Display first 10 keys

    # Choose classes and SNRs 
    target_mods = ['BPSK', 'QPSK', 'AM-DSB', 'QAM16']
    target_snrs = [-6,2,12]


# First I am manually checking the first few keys and samples
# What modulations exist?
# What shape is each sample (hint: it should be 2 channels * 128 time steps)?

x = []
y = []
mod_to_idx = {mod: idx for idx, mod in enumerate(target_mods)}  # Create a mapping from modulation to index

# Flattening the 2d matrix into a 1d vector
for target_mod in target_mods:
    for target_snr in target_snrs:
        key = (target_mod, target_snr)
        if key in data:
            samples = data[key]
            for sample in samples: 
                vector_1d = sample.flatten()
                # To normalize the vector: 
                # Substract the mean and divide by the standard deviation
                vector_1d = (vector_1d - np.mean(vector_1d)) / np.std(vector_1d)
                x.append(vector_1d)
                y.append(mod_to_idx[target_mod])
                print(vector_1d)

X = np.array(x)
Y = np.array(y)
np.savez("data/processed_rf_data.npz", X=X, Y=Y)

# 
