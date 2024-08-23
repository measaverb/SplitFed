import jax.numpy as jnp
import numpy as np
from PIL import Image


class SkinData(object):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        X = np.array(Image.open(self.df["path"][index]).resize((64, 64)))
        y = jnp.array(int(self.df["target"][index]))

        if self.transform:
            X = self.transform(X)

        return X, y

    def get_batch(self, indices):
        X_batch = []
        y_batch = []
        for idx in indices:
            X, y = self.__getitem__(idx)
            X_batch.append(X)
            y_batch.append(y)
        return jnp.stack(X_batch), jnp.stack(y_batch)

