import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import idx2numpy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

images = idx2numpy.convert_from_file('train-images.idx3-ubyte')
labels = idx2numpy.convert_from_file('train-labels.idx1-ubyte')

sample_images = images[:1000]
sample_labels = labels[:1000]

# Hiển thị một vài hình ảnh mẫu
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.ravel()


plt.tight_layout()
plt.show()
