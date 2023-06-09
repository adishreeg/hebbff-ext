import torch
import matplotlib.pyplot as plt

from net_utils import load_from_file
from data import generate_recog_data
import plotting

fig, ax = plt.subplots(3,1)
data1 = generate_recog_data(T=5000, R=1, d=25, P=0.5, multiRep=False)
hs = []
for i,Rtrain in enumerate([1,7,14]):
    fname = 'publish/mechanism/HebbNet[25,25,1]_R={}.pkl'.format(Rtrain)
    net = load_from_file(fname)

    res = plotting.get_evaluation_result(fname, data1, R=1)
    hs.append(torch.flatten(res['h']))
    ax[i].hist(hs[i], alpha=1)
    ax[i].set_ylabel(f'R={Rtrain}')
ax[2].set_xlabel('weight of unit')

plt.savefig('replication/4a_ext.png')
plt.show()
