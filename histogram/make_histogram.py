import re
import numpy as np
import matplotlib.pyplot as plt


f = open('cvlog.log')

accuracies = []

for line in f:
    if line[33:].startswith('This Accuracy:'):
        this_accuracy = float(line[48:])
        if (this_accuracy > 0.1):
            accuracies = accuracies + [this_accuracy]

accuracies = np.array(accuracies)

plt.hist(100*accuracies, bins=8, range=(93.3, 100))
plt.title("Fold Accuracies During Cross Validation")
plt.xlabel("Accuracy (%)")
plt.ylabel("Incidence")
plt.show()

print np.std(accuracies)
print np.mean(accuracies)
print np.median(accuracies)
