import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchsummary import summary

from random import randint

from IPython.display import Image
from IPython.core.display import Image, display



class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

##############################################################################
class UnFlatten(nn.Module):
    def forward(self, input, size=512):
        return input.view(input.size(0), size, 1, 1)

##############################################################################
class VAE(nn.Module):
    def __init__(self, device, image_channels=1, h_dim=512, z_dim=128):
        super(VAE, self).__init__()
        self.device = device        

        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=2, stride=2),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
    
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, image_channels, kernel_size=5, stride=2),
            nn.Sigmoid(),
        )
    

    def push_to_device(self):
        self.encoder.to(self.device)
        self.fc1.to(self.device)
        self.fc2.to(self.device)
        self.fc3.to(self.device)
        self.decoder.to(self.device)


    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z
       
    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        z = F.pad(z, (1,0,1,0), 'constant', 0) #take care of off-by-one error with zero padding (top left)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

##############################################################################
def loss_fn(recon_x, x, mu, logvar, KL_multiplier):    
    rec_loss = F.mse_loss(recon_x, x, size_average=False) #log-likelihood for gaussian
    KL_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) #KL divergence for gaussian, derived in Kingma paper
    return rec_loss + KL_multiplier*KL_loss, rec_loss, KL_loss


def train_VAE(train_dataset, device, epochs=250, bs=256):
    train_iterator = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    model = VAE(device)
    model.push_to_device()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        loss, bce, kld = torch.zeros(3)
        for idx, (images, _) in enumerate(train_iterator):
            images = images.to(device)
            recon_images, mu, logvar = model(images)
            loss, bce, kld = loss_fn(recon_images, images, mu, logvar, KL_multiplier=10)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        to_print = "Epoch[{}/{}] Loss: {:.3f} {:.3f} {:.3f}".format(epoch+1, epochs, loss.data.item()/bs, bce.data.item()/bs, kld.data.item()/bs)
        print(to_print)

    torch.save(model.state_dict(), 'vae.torch')


def test_VAE(model, test_dataset, device):
    ret = torch.tensor([])
    for i in range(8):
        x = test_dataset[randint(1, 100)][0].unsqueeze(0).to(device)
        recon_x, _, _ = model(x)
        compare_x = torch.cat([x, recon_x], dim=2)
        ret = torch.cat([ret, compare_x], dim = 0)
    save_image(ret.data.cpu(), 'sample_image.png')
    display(Image('sample_image.png', width=700, unconfined=True))
