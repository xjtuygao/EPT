# This code is adapted from https://github.com/rtqichen/ffjord.
# It can generate 2D toy datasets.

import numpy as np
import sklearn
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle


# Dataset iterator
def toy_data_gen(data, rng=None, batch_size=2000):
    if rng is None:
        rng = np.random.RandomState()

    if data == "gaussian":
        data = np.random.normal(0, 1, size=(batch_size, 2))
        data = data.astype("float32")
        return data

    if data == "swissroll":
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        return data

    elif data == "circles":
        data = sklearn.datasets.make_circles(n_samples=batch_size, factor=.5, noise=0.08)[0]
        data = data.astype("float32")
        data *= 3
        return data

    elif data == "small_circle":
        theta = np.random.uniform(0, 1, size=batch_size)
        radius = np.random.normal(0, 0.1, size=batch_size) + 1.2
        data = np.array([radius * np.cos(2 * np.pi * theta), radius * np.sin(2 * np.pi * theta)]).T
        data = data.astype("float32")
        return data

    elif data == "large_circle":
        theta = np.random.uniform(0, 1, size=batch_size)
        radius = np.random.normal(0, 0.1, size=batch_size) + 3
        data = np.array([radius * np.cos(2 * np.pi * theta), radius * np.sin(2 * np.pi * theta)]).T
        data = data.astype("float32")
        return data

    elif data == "large_4gaussians":
        scale = 3
        theta = np.random.randint(0, 4, size=batch_size)/4
        data = np.array([scale * np.cos(2 * np.pi * theta - np.pi/4), scale * np.sin(2 * np.pi * theta - np.pi/4)]).T \
                  + np.random.randn(batch_size, 2) * 0.3
        data = np.array(data, dtype="float32")
        return data

    elif data == "small_4gaussians":
        scale = 1.5
        theta = np.random.randint(0, 4, size=batch_size)/4
        data = np.array([scale * np.cos(2 * np.pi * theta - np.pi/4), scale * np.sin(2 * np.pi * theta - np.pi/4)]).T \
                  + np.random.randn(batch_size, 2) * 0.3
        data = np.array(data, dtype="float32")
        return data

    elif data == "rings":
        n_samples4 = n_samples3 = n_samples2 = batch_size // 4
        n_samples1 = batch_size - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, we set endpoint=False
        linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
        linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
        linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
        linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

        circ4_x = np.cos(linspace4)
        circ4_y = np.sin(linspace4)
        circ3_x = np.cos(linspace4) * 0.75
        circ3_y = np.sin(linspace3) * 0.75
        circ2_x = np.cos(linspace2) * 0.5
        circ2_y = np.sin(linspace2) * 0.5
        circ1_x = np.cos(linspace1) * 0.25
        circ1_y = np.sin(linspace1) * 0.25

        X = np.vstack([
            np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
            np.hstack([circ4_y, circ3_y, circ2_y, circ1_y])
        ]).T * 3.0
        X = util_shuffle(X, random_state=rng)

        # Add noise
        X = X + rng.normal(scale=0.08, size=X.shape)

        return X.astype("float32")

    elif data == "moons":
        data = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.1)[0]
        data = data.astype("float32")
        data = data * 2 + np.array([-1, -0.2])
        return data

    elif data == "8gaussians":
        scale = 3
        theta = np.random.randint(0, 8, size=batch_size)/8
        data = np.array([scale * np.cos(2 * np.pi * theta), scale * np.sin(2 * np.pi * theta)]).T \
                  + np.random.randn(batch_size, 2) * 0.3
        data = np.array(data, dtype="float32")
        return data

    elif data == "pinwheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = batch_size // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = rng.randn(num_classes*num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        return 2 * rng.permutation(np.einsum("ti,tij->tj", features, rotations))

    elif data == "2spirals":
        n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * 0.1
        return x

    elif data == "spiral1":
        n = np.sqrt(np.random.rand(batch_size, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(batch_size, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(batch_size, 1) * 0.5
        x = np.hstack((d1x, d1y)) / 3
        x += np.random.randn(*x.shape) * 0.1
        return x

    elif data == "spiral2":
        n = np.sqrt(np.random.rand(batch_size, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(batch_size, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(batch_size, 1) * 0.5
        x = np.hstack((-d1x, -d1y)) / 3
        x += np.random.randn(*x.shape) * 0.1
        return x

    elif data == "checkerboard":
        x1 = np.random.rand(batch_size) * 4 - 2
        x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        return np.concatenate([x1[:, None], x2[:, None]], 1) * 2

    elif data == "5squares":
        idx = np.random.randint(0, 5, batch_size)
        idx_zo = 1 - idx // 4
        x1 = (np.random.rand(batch_size) - 1/2) + idx_zo * (np.random.randint(0, 2, batch_size) * 4 - 2) 
        x2 = (np.random.rand(batch_size) - 1/2) + idx_zo * (np.random.randint(0, 2, batch_size) * 4 - 2)
        return  np.concatenate([x1[:, None], x2[:, None]], 1)

    elif data == "4squares":
        idx = np.random.randint(0, 2, batch_size)
        x1 = (np.random.rand(batch_size) - 1/2) + idx * (np.random.randint(0, 2, batch_size) * 4 - 2) 
        x2 = (np.random.rand(batch_size) - 1/2) + (1 - idx) * (np.random.randint(0, 2, batch_size) * 4 - 2)
        return np.concatenate([x1[:, None], x2[:, None]], 1)

    elif data == "line":
        x = rng.rand(batch_size) * 5 - 2.5
        y = x
        return np.stack((x, y), 1)
    elif data == "cos":
        x = rng.rand(batch_size) * 5 - 2.5
        y = np.sin(x) * 2.5
        return np.stack((x, y), 1)
    else:
        raise ValueError('Wrong dataset name: "%s"' % data)
