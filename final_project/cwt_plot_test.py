from utils import *
import matplotlib.pyplot as plt
X_train, y_train, X_test, y_test = init_data(subject=None)
X_train, y_train, X_test, y_test, X_valid, y_valid = preprocess_data(X_train, y_train, X_test, y_test, verbose=True)

print(X_train.shape)
scales = np.arange(1, 50)

X_train_cwt = cwt_transform(X_train[0:10, :, :], scales, ricker, verbose=True)

cwt_coeffs = np.abs(X_train_cwt[0, 0, :, :])
plt.figure(figsize=(10, 4))
plt.imshow(np.abs(cwt_coeffs), extent=[0, 30, scales[-1], scales[0]], cmap='jet', aspect='auto')
plt.colorbar(label='Magnitude')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('CWT of Trial 1, Electrode 1 with the Ricker wavelet')
plt.show()