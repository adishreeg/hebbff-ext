import numpy as np
import matplotlib.pyplot as plt

from net_utils import load_from_file
from data import generate_recog_data_batch
import plotting

# Encoded information by Landauer model 
bits = lambda p, n: -np.log2(1 - np.power(2 * p - 1, 1 / n))

fig = plt.figure(figsize=(8, 6))
ax = [fig.add_subplot(121), fig.add_subplot(122)]
Rs, accs, tpr, fpr, inf = [], [], [], [], []
i = 0
Rtrain = 6
for force in ['Anti', 'Hebb']:
    c = 'tab:red' if force == 'Anti' else 'tab:blue'
    for dim in [25, 100]:
        m = 'dashed' if dim == 25 else 'solid'

        fname = f'publish/antiHebb/HebbNet[{dim},{dim},1]_train=inf{Rtrain}_force{force}.pkl'
        net = load_from_file(fname)
        
        gen_data = lambda R: generate_recog_data_batch(T=1000, d=dim, R=R, P=0.5, multiRep=False)
        res = plotting.get_recog_positive_rates(net, gen_data, stopAtR=25)
        for j, l in enumerate([Rs, accs, tpr, fpr]): l.append(res[j])
        inf.append([bits(p, n) for p, n in zip(accs[i], Rs[i])])

        ax[0].plot(Rs[i], accs[i], c=c, linestyle=m)
        ax[1].plot(Rs[i], inf[i], c=c, linestyle=m, label=f'{force}, size={dim}')
        i += 1
ax[0].set_xlabel('R')
ax[1].set_xlabel('R')
ax[1].legend(loc='best')
ax[0].set_ylabel('Acccuracy')
ax[1].set_ylabel('Information encoded')
ax[1].set_title(f'max={round(np.max(inf), 2)} bits')
plt.tight_layout()
plt.savefig('information/HebbNet_Rtest.png')
plt.show()