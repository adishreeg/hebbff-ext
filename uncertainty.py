import numpy as np
import matplotlib.pyplot as plt
import torch
from net_utils import load_from_file
from data import generate_recog_data

fig, ax = plt.subplots(3, 1, sharex=True)
data = generate_recog_data(T=5000, R=1, d=25, P=0.5, multiRep=False)
y = data[:][1]
a2s = []
for i,Rtrain in enumerate([1,7,14]):
    fname = 'publish/mechanism/HebbNet[25,25,1]_R={}.pkl'.format(Rtrain)
    net = load_from_file(fname)

    with torch.no_grad():
        results = net.evaluate_debug(data.tensors)
    a2, y_pred = results['a2'], np.round(results['out'])
    a2s.append(a2)
    
    ax[i].hist(a2s[i][y == 0], alpha=0.5)
    ax[i].plot([0, 0], [0, ax[i].get_ylim()[1]], color='r')
    ax[i].set_ylabel('y==0')
    ax[i].set_title(f'R={Rtrain}', pad=-10)
    ax2 = ax[i].twinx()
    ax2.hist(a2s[i][y == 1], alpha=0.5, color='tab:orange')
    ax2.set_ylabel('y==1')

ax[2].set_xlabel('a2')

plt.savefig('uncertainty/y.png')
plt.show()