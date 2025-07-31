# # Generate BPSK, QAM, OFDM using NumPy/Scipy
        
# #     - Add AWGN noise at multiple SNRs
        
# #     - Normalize and shape as input tensors
# import numpy as np
# from numpy import random
# import scipy.signal as signal
# import matplotlib.pyplot as plt


# # Generate bit stream
# def generate_bit_stream(): 
#     x= random.randint(100, size=(1000)) 
#     return x

# # BPSK modulation

# def bpsk_modulation(bit_stream):
#     bpsk_signal = 2 * bit_stream - 1  # Map 0 to -1 and 1 to +1
#     return bpsk_signal

# # Convert BPSK signal to time domain 
# def bpsk_to_time_domain(): 
