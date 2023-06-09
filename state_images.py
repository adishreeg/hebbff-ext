import os
import numpy as np
from sklearn.decomposition import PCA
import torch
# from torch.utils.data import TensorDataset
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt

from net_utils import load_from_file

def pair(N):
    pairs = []
    while len(pairs) < N:
        first = np.random.choice(os.listdir('STATE'))
        matches = [x for x in os.listdir('STATE') if x.startswith(first[:5])]
        if len(matches) == 2 and matches not in pairs: pairs.append(matches)
    return pairs

def generate_sequence(R, T=50): # sequence length T <= 90 for this to work
    # Decide sequence of familiarity and state commonality
    gen = lambda: np.concatenate(
        (np.zeros(R), np.random.randint(2, size=T-R))).astype(int)
    state, familiarity = gen(), gen()
    for i in range(T): 
        if familiarity[i] == 1: state[i] = 0
    
    # Fill sequence according to parameters
    sequence = np.linspace(0, 2 * (T - 1), T).astype(int) # initialize all as novel
    for i in range(R, T):
        b = sequence[i - R]
        if familiarity[i]: sequence[i] = b
        elif state[i]: sequence[i] = b - 1 if b % 2 else b + 1 

    # Cast to real image sequence
    pairs = np.array(pair(T)).flatten()
    images = [f'STATE/{pairs[i]}' for i in sequence]
    
    return images, state, familiarity

def image_to_vector_sequence(images):
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    layer = model._modules.get('avgpool')
    model.eval()

    scaler = transforms.Resize(224)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()

    def get_vector(name):
        img = Image.open(name)
        t_img = Variable(normalize(to_tensor(scaler(img))))
        embedding = torch.zeros(512)
        def copy_data(m, i, o): 
            embedding.copy_(o.flatten())
            return o
        h = layer.register_forward_hook(copy_data)
        with torch.no_grad():
            model(t_img.unsqueeze(0))
        h.remove()
        return embedding.numpy()

    return np.array([get_vector(image) for image in images])

def binarize(data, d):
    pca = PCA(n_components=d)
    data = pca.fit_transform(data)
    return np.where(data > np.mean(data), 1, -1)

def load_model(d):
    fname = f'publish/antiHebb/HebbNet[{d},{d},1]_train=inf6_forceHebb.pkl'
    net = load_from_file(fname)
    return net

def gen_data(R, T=50, d=25):
    images, state, familiarity = generate_sequence(R, T)
    features = image_to_vector_sequence(images)
    binarized = binarize(features, d)
    return state, (
        torch.from_numpy(binarized.reshape(T, 1, d)).float(),  # x
        torch.from_numpy(familiarity).float()                  # y
    )

T, d = 50, 25
Rs, accuracies, states, familiars = [i for i in range(1, 16, 2)], [], [], []
for R in Rs:
    state, data = gen_data(R, T, d)
    net = load_model(d)
    result = net.evaluate(data).detach()
    correctness = np.where(result.numpy() > 0.5, 1, 0) == data[1].numpy()
    accuracies.append(sum(correctness) / len(correctness))
    states.append(sum(correctness[state == 1]) / sum(state))
    familiars.append(sum(correctness[data[1].numpy() == 1]) / sum(data[1].numpy()))

plt.plot(Rs, accuracies, label='All')
plt.plot(Rs, states, label='State')
plt.plot(Rs, familiars, label='Familiar')
plt.xlabel('R')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('state_figs/accuracy_R4.png')
plt.show()

bits = lambda p, n: -np.log2(1 - np.power(2 * p - 1, 1 / n))
inf = [bits(p, n) for p, n in zip(accuracies, Rs)]
plt.plot(Rs, inf)
plt.xlabel('R')
plt.ylabel('Information encoded (bits)')
plt.title(f'max = {round(max(inf), 2)} bits')
plt.savefig('state_figs/inf.png')
plt.show()