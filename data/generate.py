import numpy as np
import pandas as pd

def generate_dataset(n_samples: int = 1500, noise: float = 0.3, random_state: int = 42):
    np.random.seed(random_state)
    n_class = n_samples // 2
    
    e = np.linspace(0, 4 * np.pi, n_class)
    #class0
    x0 = e * np.cos(e)
    y0 = e * np.sin(e)
    z0 = e

    #class1
    x1 = e * np.cos(e + np.pi)
    y1 = e * np.sin(e + np.pi)
    z1 = e

    X0 = np.vstack((x0, y0, z0)).T
    X1 = np.vstack((x1, y1, z1)).T
    y0 = np.zeros(n_class)
    y1 = np.ones(n_class)

    X = np.vstack((X0, X1))
    y = np.concatenate((y0, y1))
    df = pd.DataFrame(X, columns = ['x', 'y', 'z'])
    df['label'] = y.astype(int)
    df = df.sample(random_state=random_state, frac = 1).reset_index(drop = True)

    return df

if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv('data/dataset.csv', index = False)