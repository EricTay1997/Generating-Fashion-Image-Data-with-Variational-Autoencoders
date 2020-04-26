import os
import sys
import h5py
import cv2
import math
import random, string

import numpy as np
from scipy.stats import norm
from sklearn import manifold
import torch
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from random import uniform


# Scatter with images instead of points
def imscatter(x, y, imageSize, ax, imageData, zoom):
    images = []
    for i in range(len(x)):
        x0, y0 = x[i], y[i]
        
        # Convert to image
        img = imageData[i]*255.
        img = img.astype(np.uint8).reshape([imageSize,imageSize])
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    
        #img = cv2.resize(img, (imageSize, imageSize), interpolation = cv2.INTER_LINEAR)

        # Note: OpenCV uses BGR and plt uses RGB
        image = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(image, (x0, y0), xycoords='data', frameon=False)
        images.append(ax.add_artist(ab))
    
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()

    
def plotTSNE(images, tsneInput, display=True):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(tsneInput)

    # Plot images according to t-SNE embedding
    if display:
        fig, ax = plt.subplots()
        fig.set_figheight(15)
        fig.set_figwidth(25)
        imscatter(X_tsne[:,0], X_tsne[:,1], imageSize=images.shape[-1], imageData=images, ax=ax, zoom=1.5)
        plt.show()
    else:
        return X_tsne
    
    
#start and end should be pytorch (cuda/gpu) tensors corresponding to two images that we are interpolating between
def visualizeInterpolation(start, end, model, device, save=False, nbSteps=5):
    start_np = start.detach().cpu().numpy()
    end_np = end.detach().cpu().numpy()
    
    # Compute latent space projection
    start_latent, m, logv = model.encode(start)
    end_latent, m, logv = model.encode(end)
    
    
    start_latent_np = start_latent.detach().cpu().numpy()
    end_latent_np = end_latent.detach().cpu().numpy()

    # Get original image for comparison
    vectors = []
    normalImages = []
    
    #Linear interpolation
    alphaValues = np.linspace(0, 1, nbSteps)
    
    for alpha in alphaValues:
        # Latent space interpolation
        vector = start_latent_np*(1-alpha) + end_latent_np*alpha
        vectors.append(vector)
        
        # Image space interpolation
        blendImage = cv2.addWeighted(start_np,1-alpha,end_np,alpha,0)
        normalImages.append(blendImage)

    # Decode latent space vectors
    vectors = torch.from_numpy(np.asarray(vectors)).to(device)
    reconstructions = model.decode(vectors).detach().cpu().numpy()
    

    # Put final image together
    resultLatent = None
    resultImage = None

    if save:
        hashName = ''.join(random.choice(string.lowercase) for i in range(3))

    for i in range(len(reconstructions)):
        interpolatedImage = normalImages[i][0,0,:,:]*255
        interpolatedImage = cv2.resize(interpolatedImage,(50,50))
        interpolatedImage = interpolatedImage.astype(np.uint8)
        resultImage = interpolatedImage if resultImage is None else np.hstack([resultImage,interpolatedImage])

        reconstructedImage = reconstructions[i][0,:,:]*255.
        reconstructedImage = cv2.resize(reconstructedImage,(50,50))
        reconstructedImage = reconstructedImage.astype(np.uint8)
        resultLatent = reconstructedImage if resultLatent is None else np.hstack([resultLatent,reconstructedImage])
    
        if save:
            cv2.imwrite(visualsPath+"{}_{}.png".format(hashName,i),np.hstack([interpolatedImage,reconstructedImage]))

        result = np.vstack([resultImage,resultLatent])

    
    if not save:
        plt.imshow(result, cmap='gray')


def genTSNEplots(data_loader, device, model = None):
    dataset_tensor = next(iter(data_loader))[0]
    dataset_array = dataset_tensor.numpy()
    dataset_tensor = dataset_tensor.to(device)
    
    if not model:
        print("TSNE Visualization of Raw Fashion MNIST Training Data: ")
        plotTSNE(dataset_array, dataset_array.reshape([-1, dataset_array.shape[-1]**2]))
    else:
        print("TSNE Visualization of Fashion MNIST Training Data Latent Space: ")
        z, mu, logvar = model.encode(dataset_tensor)
        z = z.detach().cpu().numpy()
        plotTSNE(dataset_array, z) #latent space of fashion MNIST learned by autoencoder


def genRandomInterpolation(data_loader, model, device): 
    dataset_tensor = next(iter(data_loader))[0]
    dataset_tensor = dataset_tensor.to(device)

    a = int(uniform(0, len(dataset_tensor) - 1))
    b = int(uniform(0, len(dataset_tensor) - 1))

    start = dataset_tensor[a,:,:,:].unsqueeze_(0)
    end = dataset_tensor[b,:,:,:].unsqueeze_(0)

    print("Linear Interpolation vs. Latent Space Interpolation")
    visualizeInterpolation(start, end, model, device, save=False, nbSteps=5) #linear interpolation of two random images vs. interoplation of latent vectors


def plotLatentByCat(k, data_loader, model, device):
    x = []
    y = []

    for idx, (image, label) in enumerate(data_loader):
        if label[0] == k:
           z, m, logv = model.encode(image.to(device))
           m = m.detach().cpu().numpy()
           x.append(m[0,0])
           y.append(m[0,1])

    #visualization of what the learned mean vector (just first 2 dimensions) looks like in the latent space
    print("Distribution of First 2 Latent Dims (out of 128) for Category", str(k), ":")
    plt.scatter(x,y);


def calcLatentStatsByCat(data_loader, model, device):
    sum_means = np.zeros([10,128])
    sum_logvar = np.zeros([10,128])
  
    cnt_means = np.zeros([10])
    cnt_logvar = np.zeros([10])

    for idx, (image, label) in enumerate(data_loader):
        z, m, logv = model.encode(image.to(device))
        
        label = label.detach().cpu().numpy()
        m = m.detach().cpu().numpy()
        logv = logv.detach().cpu().numpy()
    
        m = np.transpose(m)[:,0]
        logv = np.transpose(logv)[:,0]
        
        sum_means[label[0],:] += m
        sum_logvar[label[0],:] += logv
        
        cnt_means[label[0]] += 1
        cnt_logvar[label[0]] += 1
        
    for i in range(len(sum_means)):
        sum_means[i,:] /= cnt_means[i]
        sum_logvar[i,:] /= cnt_logvar[i]

    return sum_means, sum_logvar


def genNewImByCat(num_images, k, multiplier, latent_stats, data_loader, model, device):
    [avg_means, avg_logvar] = latent_stats
    m = torch.from_numpy(avg_means[k,:]).float().to(device)
    logvar = torch.from_numpy(avg_logvar[k,:]).float().to(device)
    
    for i in range(num_images):
        z_new = getRandomSample(m, logvar, multiplier, device).unsqueeze_(0)
        if i == 0:
            im_new = model.decode(z_new)[0,0,:,:].detach().cpu().numpy()
        else:
            im_new = np.concatenate([im_new, model.decode(z_new)[0,0,:,:].detach().cpu().numpy()], axis=1)

    print("generated image(s) from category",str(k), ":")
    plt.imshow(im_new, cmap='gray');


def getRandomSample(mu, logvar, multiplier, device):
    std = logvar.mul(0.5).exp_()
    esp = torch.randn(*mu.size()).to(device)
    z = mu + multiplier * std * esp
    return z
    




