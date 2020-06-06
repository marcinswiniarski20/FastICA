import numpy as np
from scipy.io import wavfile

import matplotlib.pyplot as plt
import glob
 
def center(x):
    mean = x.mean(axis=1, keepdims=True)
    x_centered = x - mean
    return x_centered

def compute_covariance(x):
    x_centered = center(x)
    cov = np.dot(x_centered, x_centered.T)/(x.shape[1] - 1)
    return cov
 
def whiten(x):
    # Calculate the covariance matrix
    covariance = compute_covariance(x)
 
    # Eigendecomposition
    eigen_values, eigen_vectors = np.linalg.eigh(covariance)
    # Compute whitening matrix for Principal Component Analysis (PCA)
    W_PCA = np.dot(np.diag(eigen_values**(-1/2)), eigen_vectors.T)
    X_whiten = np.dot(W_PCA, x)
    
    return X_whiten 


def estimate_contrast_functions(u):
    g = 1.0 / 4.0 * u**4
    g_derivative = u**3

    return g, g_derivative

def fastIca(signals, alpha = 1, threshold=1e-12, nb_iter=1000):
    n = len(signals)
    # Initialize random weights
    W = np.random.rand(n, n)
    for k, signal in enumerate(signals):
            w = W[k, :]
            w = w / np.sqrt((w ** 2).sum())
            i = 0
            while True:
                g, g_prim = estimate_contrast_functions(np.dot(w.T, signals))
                # Compute new weights
                wNew = np.mean(signals * g, axis=1) - np.mean(g_prim) * w
 
                # Decorrelate weights
                wNew = wNew - np.dot(np.dot(wNew, W[:k].T), W[:k])
                wNew = wNew / np.sqrt((wNew ** 2).sum())
                
                # Compute convergence condition
                manhattan_dist = np.abs(np.abs((wNew * w).sum()) - 1)
                w_dot_product = np.dot(wNew.T, w)
                
                # Update weights
                w = wNew
                i += 1
                
                # Check if algorithm converges
                if manhattan_dist <= threshold or i > nb_iter:
                    print(f"Extracting signal {k} finished for iter: {i} with dot(wNew, w) product: {w_dot_product}")
                    break
            W[k, :] = w.T
    return W
 
 
def plot_signals(signals, title="Source signals"):
    n = signals.shape[1]
    fig, axs = plt.subplots(n, 1)
    for i, signal in enumerate(signals.T):
        nb_samples = len(signal)
        t = np.linspace(0, 30, nb_samples)
        axs[i].plot(t, signal)
 
    for ax in axs.flat:
        ax.set(xlabel='Time [s]', ylabel='Amplitude')
 
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
 
    fig.suptitle(title)
    plt.show()
 
def convert_to_32bit(signal):
    m = np.max(np.abs(signalA))
    return (signal/m).astype(np.float32)
 
sample_rate, signalA = wavfile.read(f"./data/mono/Beethoven_30s.wav")
sample_rateB, signalB = wavfile.read(f"./data/mono/Chopin_30s.wav")
sample_rateB, signalC = wavfile.read(f"./data/mono/Vivaldi_30s.wav")

S = np.array([signalA, signalB, signalC]).T
plot_signals(S, title="Source signals")

# Mixing matrix
A = np.array([[0.5, 0.6, 0.2],
              [0.3, 0.5, 0.4],
              [0.5, 0.8, 0.1]])
 
# Mixed signal matrix
X = np.dot(S, A).T
plot_signals(X.T, title="Mixed signals")
 
# Center signals
Xc = center(X)
 
# Whiten mixed signals
Xw = whiten(Xc)
 
# Check if covariance of whitened matrix equals identity matrix
print("Covariance matrix of whitened data")
print(np.round(compute_covariance(Xw)))

W = fastIca(Xw,  alpha=1)
 
#Un-mix signals using
unMixed = Xw.T.dot(W.T)
plot_signals(unMixed, title="Recovered Signals")

wavfile.write("./data/mono/example_with_III_signals/initial_signalA.wav", sample_rate, convert_to_32bit(signalA))
wavfile.write("./data/mono/example_with_III_signals/initial_signalB.wav", sample_rate, convert_to_32bit(signalB))
wavfile.write("./data/mono/example_with_III_signals/initial_signalC.wav", sample_rate, convert_to_32bit(signalC))
wavfile.write("./data/mono/example_with_III_signals/mixed_signalA.wav", sample_rate,  convert_to_32bit(X[0, :]))
wavfile.write("./data/mono/example_with_III_signals/mixed_signalB.wav", sample_rate,  convert_to_32bit(X[1, :]))
wavfile.write("./data/mono/example_with_III_signals/mixed_signalC.wav", sample_rate,  convert_to_32bit(X[2, :]))
wavfile.write("./data/mono/example_with_III_signals/unMixedA.wav", sample_rate, unMixed[:, 0].astype(np.float32))
wavfile.write("./data/mono/example_with_III_signals/unMixedB.wav", sample_rate, unMixed[:, 1].astype(np.float32))
wavfile.write("./data/mono/example_with_III_signals/unMixedC.wav", sample_rate, unMixed[:, 2].astype(np.float32))

print("FastICA has been successfully applied to mixed signals.")
print("Results have been writen to path './data/mono/example_with_III_signals/'.")


