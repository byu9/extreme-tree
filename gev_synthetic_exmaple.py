#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from extreme_tree import ExtremeTree

np.random.seed(seed=123)

# Create synthetic time series
n_blocks_per_section = 100
n_samples_per_block = 100
n_samples_per_section = n_blocks_per_section * n_samples_per_block

section1_x = np.linspace(0, 1, n_samples_per_section).reshape(n_samples_per_block, n_blocks_per_section)
section2_x = np.linspace(1, 2, n_samples_per_section).reshape(n_samples_per_block, n_blocks_per_section)
section3_x = np.linspace(2, 3, n_samples_per_section).reshape(n_samples_per_block, n_blocks_per_section)

section1_z = norm.rvs(loc=section1_x, scale=0.1).reshape(n_samples_per_block, n_blocks_per_section)
section2_z = norm.rvs(loc=-section2_x, scale=0.3).reshape(n_samples_per_block, n_blocks_per_section)
section3_z = norm.rvs(loc=1 / section3_x, scale=0.2).reshape(n_samples_per_block, n_blocks_per_section)

# Take the maximum of each block
section1_indices = section1_z.argmax(axis=-1, keepdims=True)
section2_indices = section2_z.argmax(axis=-1, keepdims=True)
section3_indices = section3_z.argmax(axis=-1, keepdims=True)

section1_feature = np.take_along_axis(section1_x, section1_indices, axis=-1)
section2_feature = np.take_along_axis(section2_x, section2_indices, axis=-1)
section3_feature = np.take_along_axis(section3_x, section3_indices, axis=-1)

section1_target = np.take_along_axis(section1_z, section1_indices, axis=-1)
section2_target = np.take_along_axis(section2_z, section2_indices, axis=-1)
section3_target = np.take_along_axis(section3_z, section3_indices, axis=-1)

feature = np.vstack([section1_feature, section2_feature, section3_feature])
target = np.vstack([section1_target, section2_target, section3_target])

# Fit extreme value prediction model
model = ExtremeTree(dist='GenExtreme', max_split=20)
model.fit(feature, target, feature_names=['x'])
predict = model.predict(feature)


# Visualize fitment
def plot_heatmap(dist, vlo, vhi, hlo, hhi, resolution=1000):
    vertical = np.linspace(vlo, vhi, resolution).reshape(-1, 1)
    density = dist.pdf(vertical)
    plt.imshow(density, extent=(hlo, hhi, vlo, vhi), aspect='auto', origin='lower', cmap='hot')
    plt.colorbar(label='Probability Density')


plt.figure()
plt.scatter(section1_x.ravel(), section1_z.ravel(), marker='.', label='Section 1', s=1, alpha=0.4)
plt.scatter(section2_x.ravel(), section2_z.ravel(), marker='.', label='Section 2', s=1, alpha=0.4)
plt.scatter(section3_x.ravel(), section3_z.ravel(), marker='.', label='Section 3', s=1, alpha=0.4)
plt.scatter(feature, target, marker='+', color='white', label='Block Maxima', linewidths=1, s=50)
plot_heatmap(predict, hlo=feature.min(), hhi=feature.max(),
             vlo=min(target.min(), predict.ppf(0.1).min()),
             vhi=max(target.max(), predict.ppf(0.9).max()))

plt.show()
