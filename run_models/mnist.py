import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from scipy import linalg

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from tqdm import tnrange
import seaborn as sns
import pandas as pd
import numpy as np
from torchsummary import summary

from ..models import CVAE_first, sample
from ..models.CVAE_first import CVAE
from ..models.sample import Sample
from ..metrics import inception, calculate_fid
from ..metrics.inception import InceptionV3
from ..metrics.calculate_fid import get_activations, calculate_frechet_distance

BATCH_SIZE=50
N_EPOCHS = 10           # times to run the model on complete data
INPUT_DIM = 28 * 28 * 3     # size of each input
HIDDEN_DIM = 1024
IMAGE_CHANNELS = 3# hidden dimension
LATENT_DIM = 100        # latent vector dimension
N_CLASSES = 15          # number of classes in the data
lr = 1e-3 


train_dataset = datasets.MNIST(
    './data',
    train=True,
    download=True,
    transform=transforms)

test_dataset = datasets.MNIST(
    './data',
    train=False,
    download=True,
    transform=transforms
)

train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE)


def train(model, optimizer):
    # set the train mode
    model.train()
    # loss of the epoch
    train_loss = 0
    # kld loss
    kld_loss = 0
    # rcl_loss
    rcl_loss = 0

    kl_per_lt = {'Latent_Dimension': [], 'KL_Divergence': [], 'Latent_Mean': [], 'Latent_Variance': []}
    for i, (x, y) in enumerate(train_iterator):
        # reshape the data into [batch_size, 784]
        model.to(device)
#         print(i, x.size(), y.size())
        sm = Sample(x, y)
        x, y = sm.generate_x_y()

        x = x.view(-1,3,28,28)

        # update the gradients to zero
        optimizer.zero_grad()

        # forward pass
        reconstructed_x, z_mu, z_var, _ = model(x, y)

        # loss
        for ii in range(z_mu.size()[-1]):
            _, _, kl_per_lt_temp = calculate_loss(x, reconstructed_x, z_mu[:, ii], z_var[:, ii])
            kl_per_lt['KL_Divergence'].append(kl_per_lt_temp.item())
            kl_per_lt['Latent_Dimension'].append(ii)
            kl_per_lt['Latent_Mean'].append(z_mu[:, ii])
            kl_per_lt['Latent_Variance'].append(z_var[:, ii])
        loss, rcl, kld = calculate_loss(x, reconstructed_x, z_mu, z_var)

        # backward pass
        loss.backward()
        train_loss += loss.item()
        rcl_loss += rcl.item()
        kld_loss += kld.item()

        # update the weights
        optimizer.step()

    return train_loss, rcl_loss, kld_loss, kl_per_lt

inc = InceptionV3([3])
inc = inc.cuda()

def test(model, optimizer):
    # set the evaluation mode
    model.eval()

    # test loss for the data
    test_loss = 0
    # kld loss
    kld_loss = 0
    # rcl_loss
    rcl_loss = 0

    kl_per_lt = {'Latent_Dimension': [], 'KL_Divergence': [], 'Latent_Mean': [], 'Latent_Variance': []}

    # we don't need to track the gradients, since we are not updating the parameters during evaluation / testing
    with torch.no_grad():
        for i, (x, y) in enumerate(test_iterator):
            # reshape the data
            model.to(device)
            sm = Sample(x, y)

            x, y = sm.generate_x_y()

            x = x.view(-1,3,28,28)
            # forward pass
            reconstructed_x, z_mu, z_var, _ = model(x, y)
            blur = calc_blur(reconstructed_x)

            # loss
            for ii in range(z_mu.size()[-1]) :
                _, _, kl_per_lt_temp = calculate_loss(x, reconstructed_x, z_mu[:, ii], z_var[:, ii])
                kl_per_lt['KL_Divergence'].append(kl_per_lt_temp.item())
                kl_per_lt['Latent_Dimension'].append(ii)
                kl_per_lt['Latent_Mean'].append(z_mu[:, ii])
                kl_per_lt['Latent_Variance'].append(z_var[:, ii])
            loss, rcl, kld = calculate_loss(x, reconstructed_x, z_mu, z_var)
            test_loss += loss.item()
            rcl_loss += rcl.item()
            kld_loss += kld.item()

    return test_loss, rcl_loss, kld_loss, kl_per_lt, blur.data.item()

best_test_loss = float('inf')

modelt = CVAE(IMAGE_CHANNELS,N_CLASSES, HIDDEN_DIM, LATENT_DIM, )

optimizert = optim.Adam(modelt.parameters(), lr=lr)

dataframe = {'epoch': [], 'train_losses': [], 'train_rcl_losses': [], 'train_kl_losses': [], 'test_losses': [], 
             'test_rcl_losses': [], 'test_kl_losses': [], 'fid': [], 'blur': []}

data2 = {'epoch': [], 'n_dim': [], 'kld_avg_dim': []}

for e in tnrange(N_EPOCHS,desc='Epochs'):

    train_loss, tr_rcl_loss, tr_kld_loss, tr_kl_per_lt = train(modelt, optimizert)
    test_loss, test_rcl_loss, test_kld_loss, test_kl_per_lt, blur = test(modelt, optimizert)
    
    with torch.no_grad():
        im, lab = iter(test_iterator).next()
        im = im.repeat(1, 3, 1, 1)
 
        im = im.view(-1,3,28,28).cuda()
        
        y = idx2onehot(lab.view(-1, 1), n = 10).cuda()
        
        colors = []
        for j in range(lab.size()[0]):
            color = torch.randint(1, 4, (1,1)).item()
            other_indices = []
            color_index = []
            for a in [1,2,3]:
                if color != a:
                    other_indices.append(a)
                else:
                    color_index = a

            im[j, other_indices[0]-1, :, :].fill_(0)
            im[j, other_indices[1]-1, :, :].fill_(0)
            colors.append(color-1)
            
        colors = torch.FloatTensor(colors)
        y2 = torch.LongTensor(colors.long())
        y2 = idx2onehot(y2.view(-1, 1), n=3)
        
        y = torch.cat((y, torch.zeros(50).view(-1,1).cuda()), dim = 1)        
        y = torch.cat((y, y2.cuda()), dim = 1)
        y = torch.cat((y, torch.zeros(50).view(-1,1).cuda()), dim = 1)
        z_means,z_var = modelt.encoder(im, y)
        
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_means)

        z = torch.cat((x_sample, y), dim=1)

        # decode
        generated_x = modelt.decoder(z, y)

        X_act = get_activations(im.cpu().data.numpy(), inc, 
                                batch_size=BATCH_SIZE, dims=2048, cuda=True)
        recon_act = get_activations(generated_x.cpu().data.numpy(), inc, 
                                    batch_size=BATCH_SIZE, dims=2048, cuda=True)

        X_act_mu = np.mean(X_act, axis=0)
        recon_act_mu = np.mean(recon_act, axis=0)
        X_act_sigma = np.cov(X_act, rowvar=False)
        recon_act_sigma = np.cov(recon_act, rowvar=False)

        fid = calculate_frechet_distance(X_act_mu, X_act_sigma, recon_act_mu,
                                         recon_act_sigma, eps=1e-6)

    train_loss /= len(train_dataset)
    tr_rcl_loss /= len(train_dataset)
    tr_kld_loss /= len(train_dataset)
    test_loss /= len(test_dataset)
    test_rcl_loss /= len(test_dataset)
    test_kld_loss /= len(test_dataset)
    dataframe['epoch'].append(e)
    dataframe['train_losses'].append(train_loss)
    dataframe['train_rcl_losses'].append(tr_rcl_loss)
    dataframe['train_kl_losses'].append(tr_kld_loss)
    dataframe['test_losses'].append(test_loss)
    dataframe['test_rcl_losses'].append(test_rcl_loss)
    dataframe['test_kl_losses'].append(test_kld_loss)

    dataframe['fid'].append(fid)    
    dataframe['blur'].append(blur)

    print(f'Epoch {e}, Train Loss: {train_loss:.2f}, Train KLD Loss: {tr_kld_loss:.2f}, Test Loss: {test_loss:.2f}, Test KLD Loss: {test_kld_loss:.2f}' )
    #sns.pairplot(data = pd.DataFrame(tr_kl_per_lt), height=3, vars=["dimension"])
    print(f'Epoch {e}, Blur: {blur:.2f}, FID: {fid:.2f}')
    
    df = pd.DataFrame(tr_kl_per_lt)
    df = df.sort_values(by=['KL_Divergence'])
    n_dim = np.max(df['Latent_Dimension'])

    kld_avg_dim = np.zeros(n_dim)

    for i in range(n_dim):
        kld_avg_dim[i] = np.mean(df['KL_Divergence'][df['Latent_Dimension'] == i])
    kld_avg_dim = np.sort(kld_avg_dim)[::-1]
    
    data2['epoch'].append(e)
    data2['n_dim'].append(n_dim)
    data2['kld_avg_dim'].append(kld_avg_dim)
    
    #     print(f'Epoch {e}, Train RCL loss: {tr_rcl_loss:.2f}, Test RCL Loss: {test_rcl_loss:.2f}')
    if best_test_loss > test_loss:
        best_test_loss = test_loss
        patience_counter = 1
    else:
        patience_counter += 1

    if patience_counter > 3:
        break
