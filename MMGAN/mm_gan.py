"""

The generator's loss equals to

L(G) = E[log(1-D(G(z)))]
https://arxiv.org/abs/1406.2661

"""

import torch, torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from itertools import product
import numpy as np

def get_data(BATCH_SIZE=100):
    """ Load data for binared MNIST """
    torch.manual_seed(3435)

    # Download our data
    train_dataset = datasets.MNIST(root='./data/',
                                    train=True,
                                    transform=transforms.ToTensor(),
                                    download=True)

    test_dataset = datasets.MNIST(root='./data/',
                                   train=False,
                                   transform=transforms.ToTensor())

    # Use greyscale values as sampling probabilities to get back to [0,1]
    train_img = torch.stack([torch.bernoulli(d[0]) for d in train_dataset])
    train_label = torch.LongTensor([d[1] for d in train_dataset])

    test_img = torch.stack([torch.bernoulli(d[0]) for d in test_dataset])
    test_label = torch.LongTensor([d[1] for d in test_dataset])

    # MNIST has no official train dataset so use last 10000 as validation
    val_img = train_img[-10000:].clone()
    val_label = train_label[-10000:].clone()

    train_img = train_img[:-10000]
    train_label = train_label[:-10000]

    # Create data loaders
    train = torch.utils.data.TensorDataset(train_img, train_label)
    val = torch.utils.data.TensorDataset(val_img, val_label)
    test = torch.utils.data.TensorDataset(test_img, test_label)

    train_iter = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)

    return train_iter, val_iter, test_iter

class Generator(nn.Module):
    """docstring for Generator"""
    def __init__(self, image_size, hidden_dim, z_dim):
        super(Generator, self).__init__()
        # from a input noise to a generated image
        self.linear = nn.Linear(z_dim, hidden_dim)
        self.generate = nn.Linear(hidden_dim, image_size)

    def forward(self, x):
        activated = F.relu(self.linear(x))
        generation = torch.sigmoid(self.generate(activated))
        return generation


class Discriminator(nn.Module):
    """docstring for Discriminator"""
    def __init__(self, image_size, hidden_dim, output_dim):
        super(Discriminator, self).__init__()
        # input a image, then output is a decide P
        self.linear = nn.Linear(image_size, hidden_dim)
        self.discriminate = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        activated = F.relu(self.linear(x))
        discriminate = torch.sigmoid(self.discriminate(activated))
        return discriminate


class MMGAN(nn.Module):
    """docstring for MMGAN"""
    def __init__(self, image_size, hidden_dim, z_dim, output_dim=1):
        super(MMGAN, self).__init__()
        self.__dict__.update(locals())
        self.G = Generator(image_size,hidden_dim, z_dim)
        self.D = Discriminator(image_size, hidden_dim, output_dim)


class MMGANTrainer:
    """docstring for MMGANTrainer"""
    def __init__(self, model, train_iter, val_iter, test_iter, viz=False):

        self.model = model
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.test_iter = test_iter
        self.viz = viz
        self.Glosses = []
        self.Dlosses = []

    def train(self, num_epochs, G_lr=2e-4, D_lr=2e-4, D_steps=1, G_init=5):
        '''
            INPUT:
                G_lr : learning rate for generator's Adam optimizer
                D_lr: learning rate for discriminator's Adam optimizer
                D_steps: training step ratio for how ofter to train D compared to G
                G_init: number of training steps to pre-train G
        '''
        G_opt = optim.Adam(params=[p for p in self.model.G.parameters() if p.requires_grad], lr=G_lr)
        D_opt = optim.Adam(params=[p for p in self.model.D.parameters() if p.requires_grad], lr=D_lr)

        epoch_steps = int(np.ceil(len(self.train_iter)/D_steps))
        # Let G train for a few steps, the MM GANs have trouble learning very early on in training
        if G_init > 0:
            images = self.process_batch(self.train_iter)
            G_opt.zero_grad()
            G_loss = self.trainG(images)
            G_loss.backward()
            G_opt.step()

        for epoch in range(1, num_epochs+1):
            self.model.train()
            G_losses, D_losses = [], []
            for x in range(epoch_steps):
                D_step_loss = []
                for j in range(D_steps):
                    images = self.process_batch(self.train_iter)
                    D_opt.zero_grad()
                    D_loss = self.trainD(images)
                    D_loss.backward()
                    D_opt.step()
                    D_step_loss.append(D_loss.item())
                D_losses.append(np.mean(D_step_loss))
                G_opt.zero_grad()
                G_loss = self.trainG(images)
                G_losses.append(G_loss.item())
                G_loss.backward()
                G_opt.step()

            self.Glosses.extend(G_losses)
            self.Dlosses.extend(D_losses)
            print ("Epoch[%d/%d], G Loss: %.4f, D Loss: %.4f"
                   %(epoch, num_epochs, np.mean(G_losses), np.mean(D_losses)))
            self.num_epochs = epoch

            if self.viz:
                self.generate_images(epoch)
                plt.show()


    def trainD(self, images):
        DX_score = self.model.D(images)
        noise = self.compute_noise(images.shape[0], self.model.z_dim)
        G_output = self.model.G(noise)
        DG_score = self.model.D(G_output)
        D_loss = -torch.mean(torch.log(DX_score+1e-8) + torch.log(1-DG_score+1e-8))
        return torch.sum(D_loss)

    def trainG(self, images):
        noise = self.compute_noise(images.shape[0], self.model.z_dim)
        G_output = self.model.G(noise)
        DG_score = self.model.D(G_output)
        G_loss = torch.mean(torch.log((1-DG_score) +1e-8))
        return G_loss

    def compute_noise(self, batch_size, z_dim):
        return torch.randn(batch_size, z_dim)

    def process_batch(self, iterator):
        images, _ = next(iter(iterator))
        images = images.view(images.shape[0],-1)
        return images

    def generate_images(self, epoch, num_outputs=36, save=True):
        self.model.eval()
        noise = self.compute_noise(num_outputs, self.model.z_dim)
        images = self.model.G(noise)
        images = images.view(images.shape[0], 28, 28, -1).squeeze()
        plt.close()
        size_figure_grid, k = int(num_outputs**0.5), 0
        fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
        for i, j in product(range(size_figure_grid), range(size_figure_grid)):
            ax[i,j].get_xaxis().set_visible(False)
            ax[i,j].get_yaxis().set_visible(False)
            ax[i,j].imshow(images[k].data.numpy(), cmap='gray')
            k += 1




if __name__ == '__main__':
    train_iter, val_iter, test_iter = get_data()
    model = MMGAN(image_size=784, hidden_dim=256, z_dim=128)
    trainer = MMGANTrainer(model=model, train_iter=train_iter, val_iter=val_iter, test_iter=test_iter, viz=True)
    trainer.train(num_epochs=25,
                  G_lr=2e-4,
                  D_lr=2e-4,
                  D_steps=1,
                  G_init=5)











