import os
import struct
import numpy as np

def read_idx(path: str) -> np.ndarray:                  # reading algorithm generation

    with open(path, 'rb') as f:                             
        magic = struct.unpack('>I', f.read(4))[0]           
        nd = magic & 0xFF
        dtype_code = (magic >> 8) & 0xFF
        dtype_map = {
            0x08: np.uint8, 0x09: np.int8, 0x0B: np.int16,
            0x0C: np.int32, 0x0D: np.float32, 0x0E: np.float64
        }
        dims = [struct.unpack('>I', f.read(4))[0] for _ in range(nd)]
        data = np.frombuffer(f.read(), dtype=dtype_map[dtype_code])
        return data.reshape(dims)


def load_mnist(data_dir: str):                          # use read_idx to load the dataset
    
    files = {
        'train_images': 'train-images-idx3-ubyte',
        'train_labels': 'train-labels-idx1-ubyte',
        'test_images':  't10k-images-idx3-ubyte',
        'test_labels':  't10k-labels-idx1-ubyte',
    }
    X_train = read_idx(os.path.join(data_dir, files['train_images'])).astype(np.float32)
    y_train = read_idx(os.path.join(data_dir, files['train_labels'])).astype(np.int64)
    X_test  = read_idx(os.path.join(data_dir, files['test_images'])).astype(np.float32)
    y_test  = read_idx(os.path.join(data_dir, files['test_labels'])).astype(np.int64)
    return X_train, y_train, X_test, y_test


def preprocess(X: np.ndarray) -> np.ndarray:                # normalise

    return (X.reshape(X.shape[0], -1) / 255.0).astype(np.float32)


def stratified_split(X: np.ndarray, y: np.ndarray, val_ratio: float = 0.1, seed: int = 42):      # splitting trsining into train and val

    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_ratio))
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])
    
    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

   