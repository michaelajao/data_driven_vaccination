from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# load MNIST data
train_dataset = datasets.MNIST( root = 'data', train = True, transform = transforms.ToTensor(), download = True)
test_dataset = datasets.MNIST( root = 'data', train = False, transform = transforms.ToTensor())

len(train_dataset), len(test_dataset)

from torch.utils.data.dataset import random_split
train_dataset, val_dataset = random_split(train_dataset, [50000, 10000])

train_loader = DataLoader(dataset = train_dataset, batch_size = 64, shuffle = True)
val_loader = DataLoader(dataset = val_dataset, batch_size = 64, shuffle = False)
test_loader = DataLoader(dataset = test_dataset, batch_size = 64, shuffle = False)

from collections import Counter

train_counter = Counter()
for images, labels in train_loader:
    train_counter.update(labels.tolist())
    
print(f"train label distribution: {sorted(train_counter.items())}")

val_counter = Counter()
for images, labels in val_loader:
    val_counter.update(labels.tolist())
    
print(f"val label distribution: {sorted(val_counter.items())}")

test_counter = Counter()
for images, labels in test_loader:
    test_counter.update(labels.tolist())
    
print(f"test label distribution: {sorted(test_counter.items())}")

# majority rule classifier
majority_class = train_counter.most_common(1)[0][0]

baseline_accuracy = train_counter[majority_class] / sum(train_counter.values())
print(f"Baseline accuracy: {baseline_accuracy:.2f} {baseline_accuracy * 100:.2f}%")


# visualise a few images
import matplotlib.pyplot as plt
import numpy as np

# convert images to numpy for visualisation
for images, labels in train_loader:
    break

fig, axes = plt.subplots(nrows = 1, ncols = 10, figsize = (20, 2))

for i, (image, label) in enumerate(zip(images, labels)):
    if i >= 10:
        break
    axes[i].imshow(image[0], cmap = 'gray')
    axes[i].set_title(label.item())
    axes[i].axis('off')
    
plt.show()

