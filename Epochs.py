from Perceptron import *
from Perceptron_applicability import *

ppn = Perceptron(step_size=0.1, num_iter=40)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_list)+1), ppn.errors_list, marker="o")
plt.xlabel("Epochs")
plt.ylabel("Number of updates")
plt.show()