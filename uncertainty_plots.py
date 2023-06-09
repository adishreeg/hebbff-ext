import numpy as np
import matplotlib.pyplot as plt

from data import generate_recog_data
from uncertainty_clf import Certainty

R = 1
filename = lambda R: f'publish/mechanism/HebbNet[25,25,1]_R={R}.pkl'
module = Certainty(filename(R))

data = generate_recog_data(T=5000, R=R, d=25, P=0.5, multiRep=False)
module.train(data.tensors)
print(f'Parameters: {module.parameters}')
print(f'Priors: {module.priors}')

test = generate_recog_data(T=5000, R=R, d=25, P=0.5, multiRep=False)
c, a2, out = module.evaluate(test.tensors)

truth = np.round(test[:][1].flatten().tolist())
preds = np.round(out.flatten().tolist())

plt.clf()
fig, ax = plt.subplots(2, sharex=True)
ax[0].set_title('Certainty rating')
ax[0].hist(c[truth == preds], alpha=0.8, color='tab:blue')
ax[0].set_ylabel('correct', c='tab:blue')
ax[1].hist(c[truth != preds], alpha=0.8, color='tab:red')
ax[1].set_ylabel('incorrect', c='tab:red')
plt.savefig('uncertainty/hist.png')

plt.clf()
plt.scatter(out[preds == truth], c[truth == preds], alpha=0.5, color='tab:blue', label='Correct')
plt.scatter(out[preds != truth], c[truth != preds], alpha=0.5, color='tab:red', label='Incorrect')
plt.xlabel('Non-binarized output')
plt.ylabel('Certainty rating')
plt.legend()
plt.savefig('uncertainty/scatter.png')