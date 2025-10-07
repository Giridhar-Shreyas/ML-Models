import numpy as np
import re
import matplotlib.pyplot as plt
import math
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim
from sklearn.preprocessing import normalize
from io import StringIO
import pandas as pd
import os
import sys

imgPWD="/scratch/project_2000403/sgiridha/img"

def load_sclice_local(filename, domain_size):
    df = pd.read_csv(filename)

    df['it'] = df['it'].astype(int)

    df['TAU_xx'] = df['TAU_xx'].astype(float)
    df['TAU_xx_inferred'] = df['TAU_xx_inferred'].astype(float)

    df['TAU_yy'] = df['TAU_yy'].astype(float)
    df['TAU_yy_inferred'] = df['TAU_yy_inferred'].astype(float)

    df['TAU_zz'] = df['TAU_zz'].astype(float)
    df['TAU_zz_inferred'] = df['TAU_zz_inferred'].astype(float)
    df['TAU_xy'] = df['TAU_xy'].astype(float)
    df['TAU_xy_inferred'] = df['TAU_xy_inferred'].astype(float)

    df['TAU_yz'] = df['TAU_yz'].astype(float)
    df['TAU_yz_inferred'] = df['TAU_yz_inferred'].astype(float)

    df['TAU_xz'] = df['TAU_xz'].astype(float)
    df['TAU_xz_inferred'] = df['TAU_xz_inferred'].astype(float)

    df['UUMEAN_x'] = df['UUMEAN_x'].astype(float)
    df['UUMEAN_y'] = df['UUMEAN_y'].astype(float)
    df['UUMEAN_z'] = df['UUMEAN_z'].astype(float)


    df['UUX'] = df['UUX'].astype(float)
    df['UUY'] = df['UUY'].astype(float)
    df['UUZ'] = df['UUZ'].astype(float)

    true_tau_xx = df['TAU_xx'].to_numpy().reshape((domain_size,domain_size,domain_size), order='F')
    predicted_tau_xx = df['TAU_xx_inferred'].to_numpy().reshape((domain_size,domain_size,domain_size), order='F')

    true_tau_yy = df['TAU_yy'].to_numpy().reshape((domain_size,domain_size,domain_size), order='F')
    predicted_tau_yy = df['TAU_yy_inferred'].to_numpy().reshape((domain_size,domain_size,domain_size), order='F')

    true_tau_zz = df['TAU_zz'].to_numpy().reshape((domain_size,domain_size,domain_size), order='F')
    predicted_tau_zz = df['TAU_zz_inferred'].to_numpy().reshape((domain_size,domain_size,domain_size), order='F')

    true_tau_xy = df['TAU_xy'].to_numpy().reshape((domain_size,domain_size,domain_size), order='F')
    predicted_tau_xy = df['TAU_xy_inferred'].to_numpy().reshape((domain_size,domain_size,domain_size), order='F')

    true_tau_yz = df['TAU_yz'].to_numpy().reshape((domain_size,domain_size,domain_size), order='F')
    predicted_tau_yz = df['TAU_yz_inferred'].to_numpy().reshape((domain_size,domain_size,domain_size), order='F')

    true_tau_xz = df['TAU_xz'].to_numpy().reshape((domain_size,domain_size,domain_size), order='F')
    predicted_tau_xz = df['TAU_xz_inferred'].to_numpy().reshape((domain_size,domain_size,domain_size), order='F')

    uumean_x = df['UUMEAN_x'].to_numpy().reshape((domain_size,domain_size,domain_size), order='F')
    uumean_y = df['UUMEAN_y'].to_numpy().reshape((domain_size,domain_size,domain_size), order='F')
    uumean_z = df['UUMEAN_z'].to_numpy().reshape((domain_size,domain_size,domain_size), order='F')

    uu_x = df['UUX'].to_numpy().reshape((domain_size,domain_size,domain_size), order='F')
    uu_y = df['UUY'].to_numpy().reshape((domain_size,domain_size,domain_size), order='F')
    uu_z = df['UUZ'].to_numpy().reshape((domain_size,domain_size,domain_size), order='F')

    true_taus = np.stack([true_tau_xx, true_tau_yy, true_tau_zz, true_tau_xy, true_tau_yz, true_tau_xz])
    pred_taus = np.stack([predicted_tau_xx, predicted_tau_yy, predicted_tau_zz, predicted_tau_xy, predicted_tau_yz, predicted_tau_xz])
    uumeans = np.stack([uumean_x, uumean_y, uumean_z])
    uus = np.stack([uu_x, uu_y, uu_z])

    return true_taus, pred_taus, uumeans, uus



def plot_uumean_vs_uu(uumeans, uus, save, k, domain_size, filename=None):
    x_coords = range(0, domain_size)
    y_coords = range(0, domain_size)
    X, Y = np.meshgrid(
        x_coords,
        y_coords,
        indexing='ij'
    )

    fig, axes = plt.subplots(3,2)  # 1 row, 2 columns



    c = axes[0,0].contourf(X, Y, uumeans[0][:,:,k], levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[0,0])
    axes[0,0].set_title(r'UUMEAN X')
    axes[0,0].set_xlabel('x')
    axes[0,0].set_ylabel('y')
    axes[0,0].axis('equal')

    c = axes[0,1].contourf(X, Y, uus[0][:,:,k], levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[0,1])
    axes[0,1].set_title(r'UUX')
    axes[0,1].set_xlabel('x')
    axes[0,1].set_ylabel('y')
    axes[0,1].axis('equal')


    c = axes[1,0].contourf(X, Y, uumeans[1][:,:,k], levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[1,0])
    axes[1,0].set_title(r'UUMEAN Y')
    axes[1,0].set_xlabel('x')
    axes[1,0].set_ylabel('y')
    axes[1,0].axis('equal')

    c = axes[1,1].contourf(X, Y, uus[1][:,:,k], levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[1,1])
    axes[1,1].set_title(r'UUY')
    axes[1,1].set_xlabel('x')
    axes[1,1].set_ylabel('y')
    axes[1,1].axis('equal')


    c = axes[2,0].contourf(X, Y, uumeans[2][:,:,k], levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[2,0])
    axes[2,0].set_title(r'UUMEAN Z')
    axes[2,0].set_xlabel('x')
    axes[2,0].set_ylabel('y')
    axes[2,0].axis('equal')


    c = axes[2,1].contourf(X, Y, uus[2][:,:,k], levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[2,1])
    axes[2,1].set_title(r'UUZ')
    axes[2,1].set_xlabel('x')
    axes[2,1].set_ylabel('y')
    axes[2,1].axis('equal')


    fig.suptitle(f'UUMEANs vs UUX, UUY, UUZ')

    plt.tight_layout()
    if(save == True):
        plt.savefig(filename, dpi=700)
    plt.show()

def plot_tau_vs_inferre_tau(true_taus, pred_taus, save, domain_size, title_name, i = None, j = None, k = None, filename = None):

    true_taus_slice = true_taus
    pred_taus_slice = pred_taus

    if i is not None:
        true_taus_slice = true_taus[:,i,:,:]
        pred_taus_slice = pred_taus[:,i,:,:]
        
    if j is not None:
        true_taus_slice = true_taus[:,:,j,:]
        pred_taus_slice = pred_taus[:,:,j,:]
    
    if k is not None:
        true_taus_slice = true_taus[:,:,:,k]
        pred_taus_slice = pred_taus[:,:,:,k]

    true_tau_xx = true_taus_slice[0]
    true_tau_yy = true_taus_slice[1]
    true_tau_zz = true_taus_slice[2]
    true_tau_xy = true_taus_slice[3]
    true_tau_yz = true_taus_slice[4]
    true_tau_xz = true_taus_slice[5]

    predicted_tau_xx = pred_taus_slice[0]
    predicted_tau_yy = pred_taus_slice[1]
    predicted_tau_zz = pred_taus_slice[2]
    predicted_tau_xy = pred_taus_slice[3]
    predicted_tau_yz = pred_taus_slice[4]
    predicted_tau_xz = pred_taus_slice[5]

    x_coords = range(0, domain_size)
    y_coords = range(0, domain_size)
    X, Y = np.meshgrid(
        x_coords,
        y_coords,
        indexing='ij'
    )

    fig, axes = plt.subplots(6, 2, figsize=(5, 10))


    c = axes[0,0].contourf(X, Y, true_tau_xx, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[0,0])
    axes[0,0].set_title(r'True $\tau_{xx}$')
    axes[0,0].axis('equal')

    c = axes[0,1].contourf(X, Y, predicted_tau_xx, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[0,1])
    axes[0,1].set_title(r'Infered $\tau_{xx}$')
    axes[0,1].axis('equal')


    
    c = axes[1,0].contourf(X, Y, true_tau_yy, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[1,0])
    axes[1,0].set_title(r'True $\tau_{yy}$')
    axes[1,0].axis('equal')

    c = axes[1,1].contourf(X, Y, predicted_tau_yy, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[1,1])
    axes[1,1].set_title(r'Infered $\tau_{yy}$')
    axes[1,1].axis('equal')



    c = axes[2,0].contourf(X, Y, true_tau_zz, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[2,0])
    axes[2,0].set_title(r'True $\tau_{zz}$')
    axes[2,0].axis('equal')

    c = axes[2,1].contourf(X, Y, predicted_tau_zz, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[2,1])
    axes[2,1].set_title(r'Infered $\tau_{zz}$')
    axes[2,1].axis('equal')



    c = axes[3,0].contourf(X, Y, true_tau_xy, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[3,0])
    axes[3,0].set_title(r'True $\tau_{xy}$')
    axes[3,0].axis('equal')

    c = axes[3,1].contourf(X, Y, predicted_tau_xy, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[3,1])
    axes[3,1].set_title(r'Infered $\tau_{xy}$')
    axes[3,1].axis('equal')



    c = axes[4,0].contourf(X, Y, true_tau_yz, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[4,0])
    axes[4,0].set_title(r'True $\tau_{yz}$')
    axes[4,0].axis('equal')

    c = axes[4,1].contourf(X, Y, predicted_tau_yz, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[4,1])
    axes[4,1].set_title(r'Infered $\tau_{yz}$')
    axes[4,1].axis('equal')



    c = axes[5,0].contourf(X, Y, true_tau_xz, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[5,0])
    axes[5,0].set_title(r'True $\tau_{xz}$')
    axes[5,0].axis('equal')

    c = axes[5,1].contourf(X, Y, predicted_tau_xz, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[5,1])
    axes[5,1].set_title(r'Infered $\tau_{xz}$')
    axes[5,1].axis('equal')
    

    fig.suptitle(f'{title_name}')

    plt.tight_layout()
    if(save == True):
        plt.savefig(filename, dpi=700)
    plt.show()

def plot_tau_vs_inferre_tau_l2(true_taus, pred_taus, save, domain_size, title_name, i = None, j = None, k = None, filename=None):

    true_taus_slice = true_taus
    pred_taus_slice = pred_taus

    if i is not None:
        true_taus_slice = true_taus[:,i,:,:]
        pred_taus_slice = pred_taus[:,i,:,:]
        
    if j is not None:
        true_taus_slice = true_taus[:,:,j,:]
        pred_taus_slice = pred_taus[:,:,j,:]
    
    if k is not None:
        true_taus_slice = true_taus[:,:,:,k]
        pred_taus_slice = pred_taus[:,:,:,k]

    true_tau_xx = true_taus_slice[0]
    true_tau_yy = true_taus_slice[1]
    true_tau_zz = true_taus_slice[2]
    true_tau_xy = true_taus_slice[3]
    true_tau_yz = true_taus_slice[4]
    true_tau_xz = true_taus_slice[5]

    predicted_tau_xx = pred_taus_slice[0]
    predicted_tau_yy = pred_taus_slice[1]
    predicted_tau_zz = pred_taus_slice[2]
    predicted_tau_xy = pred_taus_slice[3]
    predicted_tau_yz = pred_taus_slice[4]
    predicted_tau_xz = pred_taus_slice[5]
    
    x_coords = range(0, domain_size)
    y_coords = range(0, domain_size)
    X, Y = np.meshgrid(
        x_coords,
        y_coords,
        indexing='ij'
    )

    x_coords = range(0, 5)
    y_coords = range(0, 22)
    X_zz, Y_zz = np.meshgrid(
        x_coords,
        y_coords,
        indexing='ij'
    )

    fig, axes = plt.subplots(6, 3, figsize=(15, 20))


    c = axes[0,0].contourf(X, Y, true_tau_xx, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[0,0])
    axes[0,0].set_title(r'True $\tau_{xx}$')
    axes[0,0].axis('equal')

    c = axes[0,1].contourf(X, Y, predicted_tau_xx, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[0,1])
    axes[0,1].set_title(r'Infered $\tau_{xx}$')
    axes[0,1].axis('equal')

    c = axes[0,2].contourf(X, Y, np.sqrt((true_tau_xx - predicted_tau_xx)**2), levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[0,2])
    axes[0,2].set_title(r'l2 difference $\tau_{xx}$')
    axes[0,2].axis('equal')


    
    c = axes[1,0].contourf(X, Y, true_tau_yy, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[1,0])
    axes[1,0].set_title(r'True $\tau_{yy}$')
    axes[1,0].axis('equal')

    c = axes[1,1].contourf(X, Y, predicted_tau_yy, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[1,1])
    axes[1,1].set_title(r'Infered $\tau_{yy}$')
    axes[1,1].axis('equal')

    c = axes[1,2].contourf(X, Y, np.sqrt((true_tau_yy - predicted_tau_yy)**2), levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[1,2])
    axes[1,2].set_title(r'l2 difference $\tau_{yy}$')
    axes[1,2].axis('equal')



    c = axes[2,0].contourf(X, Y, true_tau_zz, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[2,0])
    axes[2,0].set_title(r'True $\tau_{zz}$')
    axes[2,0].axis('equal')

    c = axes[2,1].contourf(X, Y, predicted_tau_zz, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[2,1])
    axes[2,1].set_title(r'Infered $\tau_{zz}$')
    axes[2,1].axis('equal')

    c = axes[2,2].contourf(X, Y, np.sqrt((true_tau_zz - predicted_tau_zz)**2), levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[2,2])
    axes[2,2].set_title(r'l2 difference $\tau_{zz}$')
    axes[2,2].axis('equal')


    c = axes[3,0].contourf(X, Y, true_tau_xy, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[3,0])
    axes[3,0].set_title(r'True $\tau_{xy}$')
    axes[3,0].axis('equal')

    c = axes[3,1].contourf(X, Y, predicted_tau_xy, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[3,1])
    axes[3,1].set_title(r'Infered $\tau_{xy}$')
    axes[3,1].axis('equal')

    c = axes[3,2].contourf(X, Y, np.sqrt((true_tau_xy - predicted_tau_xy)**2), levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[3,2])
    axes[3,2].set_title(r'l2 difference $\tau_{xy}$')
    axes[3,2].axis('equal')


    c = axes[4,0].contourf(X, Y, true_tau_yz, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[4,0])
    axes[4,0].set_title(r'True $\tau_{yz}$')
    axes[4,0].axis('equal')

    c = axes[4,1].contourf(X, Y, predicted_tau_yz, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[4,1])
    axes[4,1].set_title(r'Infered $\tau_{yz}$')
    axes[4,1].axis('equal')

    c = axes[4,2].contourf(X, Y, np.sqrt((true_tau_yz - predicted_tau_yz)**2), levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[4,2])
    axes[4,2].set_title(r'l2 difference $\tau_{yz}$')
    axes[4,2].axis('equal')


    c = axes[5,0].contourf(X, Y, true_tau_xz, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[5,0])
    axes[5,0].set_title(r'True $\tau_{xz}$')
    axes[5,0].axis('equal')

    c = axes[5,1].contourf(X, Y, predicted_tau_xz, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[5,1])
    axes[5,1].set_title(r'Infered $\tau_{xz}$')
    axes[5,1].axis('equal')

    c = axes[5,2].contourf(X, Y, np.sqrt((true_tau_xz - predicted_tau_xz)**2), levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[5,2])
    axes[5,2].set_title(r'l2 difference $\tau_{xz}$')
    axes[5,2].axis('equal')
    

    
    fig.suptitle(f'{title_name}')
    plt.tight_layout()
    
    if(save == True):
        plt.savefig(filename, dpi=700)
    plt.show()




def per_pixel_psnr(true, pred, data_range=1.0, eps=1e-10):
    mse = (true - pred) ** 2
    return 10 * np.log10((data_range ** 2) / (mse + eps))

def per_pixel_mse(true, pred):
    return (true - pred) ** 2


def plot_tau_vs_inferre_tau_full(true_taus, pred_taus, save, domain_size, title_name, i = None, j = None, k = None, filename=None):


    true_taus_slice = true_taus
    pred_taus_slice = pred_taus

    if i is not None:
        true_taus_slice = true_taus[:,i,:,:]
        pred_taus_slice = pred_taus[:,i,:,:]
        
    if j is not None:
        true_taus_slice = true_taus[:,:,j,:]
        pred_taus_slice = pred_taus[:,:,j,:]
    
    if k is not None:
        true_taus_slice = true_taus[:,:,:,k]
        pred_taus_slice = pred_taus[:,:,:,k]

    true_tau_xx = true_taus_slice[0]
    true_tau_yy = true_taus_slice[1]
    true_tau_zz = true_taus_slice[2]
    true_tau_xy = true_taus_slice[3]
    true_tau_yz = true_taus_slice[4]
    true_tau_xz = true_taus_slice[5]

    predicted_tau_xx = pred_taus_slice[0]
    predicted_tau_yy = pred_taus_slice[1]
    predicted_tau_zz = pred_taus_slice[2]
    predicted_tau_xy = pred_taus_slice[3]
    predicted_tau_yz = pred_taus_slice[4]
    predicted_tau_xz = pred_taus_slice[5]


    eps = min(arr.min() for arr in [true_tau_xx, true_tau_yy, true_tau_zz, true_tau_xy, true_tau_yz, true_tau_xz])

    x_coords = range(0, domain_size)
    y_coords = range(0, domain_size)
    X, Y = np.meshgrid(
        x_coords,
        y_coords,
        indexing='ij'
    )

    fig, axes = plt.subplots(6, 7, figsize=(25, 30))
    true = true_tau_xx
    pred = predicted_tau_xx
    c = axes[0,0].contourf(X, Y, true, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[0,0])
    axes[0,0].set_title(r'True $\tau_{xx}$')
    axes[0,0].axis('equal')

    c = axes[0,1].contourf(X, Y, pred, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[0,1])
    axes[0,1].set_title(r'Infered $\tau_{xx}$')
    axes[0,1].axis('equal')

    mse = per_pixel_mse(true, pred)

    c = axes[0,2].contourf(X, Y, ssim(true, pred, full=True, data_range=true.max() - true.min())[1], levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[0,2])
    axes[0,2].set_title(r'Structural Similarity in $\tau_{xx}$')
    axes[0,2].axis('equal')

    
    c = axes[0,3].contourf(X, Y, per_pixel_psnr(true, pred, data_range=true.max() - true.min()), levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[0,3])
    axes[0,3].set_title(r'PSNR in $\tau_{xx}$')
    axes[0,3].axis('equal')
    
    

    c = axes[0,4].contourf(X, Y, ((pred - true)/true + eps), levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[0,4])
    axes[0,4].set_title(r'Fractional Difference $\tau_{xx}$')
    axes[0,4].axis('equal')

    c = axes[0,5].contourf(X, Y, (true - pred), levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[0,5])
    axes[0,5].set_title(r'Difference $\tau_{xx}$')
    axes[0,5].axis('equal')

    c = axes[0,6].contourf(X, Y, mse, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[0,6])
    axes[0,6].set_title(r'MSE $\tau_{xx}$')
    axes[0,6].axis('equal')


    true = true_tau_yy
    pred = predicted_tau_yy
    c = axes[1,0].contourf(X, Y, true, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[1,0])
    axes[1,0].set_title(r'True $\tau_{yy}$')
    axes[1,0].axis('equal')

    c = axes[1,1].contourf(X, Y, pred, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[1,1])
    axes[1,1].set_title(r'Infered $\tau_{yy}$')
    axes[1,1].axis('equal')

    mse = per_pixel_mse(true, pred)

    c = axes[1,2].contourf(X, Y, ssim(true, pred, full=True, data_range=true.max() - true.min())[1], levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[1,2])
    axes[1,2].set_title(r'Structural Similarity in $\tau_{yy}$')
    axes[1,2].axis('equal')


    c = axes[1,3].contourf(X, Y, per_pixel_psnr(true, pred, data_range=true.max() - true.min()), levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[1,3])
    axes[1,3].set_title(r'Structural Similarity in $\tau_{yy}$')
    axes[1,3].axis('equal')

    c = axes[1,4].contourf(X, Y, ((pred - true)/true + eps), levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[1,4])
    axes[1,4].set_title(r'Fractional Difference $\tau_{yy}$')
    axes[1,4].axis('equal')

    c = axes[1,5].contourf(X, Y, (true - pred), levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[1,5])
    axes[1,5].set_title(r'Difference $\tau_{yy}$')
    axes[1,5].axis('equal')

    c = axes[1,6].contourf(X, Y, mse, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[1,6])
    axes[1,6].set_title(r'MSE $\tau_{yy}$')
    axes[1,6].axis('equal')


    true = true_tau_zz
    pred = predicted_tau_zz
    c = axes[2,0].contourf(X, Y, true, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[2,0])
    axes[2,0].set_title(r'True $\tau_{zz}$')
    axes[2,0].axis('equal')

    c = axes[2,1].contourf(X, Y, pred, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[2,1])
    axes[2,1].set_title(r'Infered $\tau_{zz}$')
    axes[2,1].axis('equal')

    mse = per_pixel_mse(true, pred)

    c = axes[2,2].contourf(X, Y, ssim(true, pred, full=True, data_range=true.max() - true.min())[1], levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[2,2])
    axes[2,2].set_title(r'Structural Similarity in $\tau_{zz}$')
    axes[2,2].axis('equal')


    c = axes[2,3].contourf(X, Y, per_pixel_psnr(true, pred, data_range=true.max() - true.min()), levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[2,3])
    axes[2,3].set_title(r'Structural Similarity in $\tau_{zz}$')
    axes[2,3].axis('equal')

    c = axes[2,4].contourf(X, Y, ((pred - true)/true + eps), levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[2,4])
    axes[2,4].set_title(r'Fractional Difference $\tau_{zz}$')
    axes[2,4].axis('equal')

    c = axes[2,5].contourf(X, Y, (true - pred), levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[2,5])
    axes[2,5].set_title(r'Difference $\tau_{zz}$')
    axes[2,5].axis('equal')

    c = axes[2,6].contourf(X, Y, mse, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[2,6])
    axes[2,6].set_title(r'MSE $\tau_{zz}$')
    axes[2,6].axis('equal')





    true = true_tau_xy
    pred = predicted_tau_xy
    c = axes[3,0].contourf(X, Y, true, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[3,0])
    axes[3,0].set_title(r'True $\tau_{xy}$')
    axes[3,0].axis('equal')

    c = axes[3,1].contourf(X, Y, pred, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[3,1])
    axes[3,1].set_title(r'Infered $\tau_{xy}$')
    axes[3,1].axis('equal')

    mse = per_pixel_mse(true, pred)

    c = axes[3,2].contourf(X, Y, ssim(true, pred, full=True, data_range=true.max() - true.min())[1], levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[3,2])
    axes[3,2].set_title(r'Structural Similarity in $\tau_{xy}$')
    axes[3,2].axis('equal')


    c = axes[3,3].contourf(X, Y, per_pixel_psnr(true, pred, data_range=true.max() - true.min()), levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[3,3])
    axes[3,3].set_title(r'Structural Similarity in $\tau_{xy}$')
    axes[3,3].axis('equal')

    c = axes[3,4].contourf(X, Y, ((pred - true)/true + eps), levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[3,4])
    axes[3,4].set_title(r'Fractional Difference $\tau_{xy}$')
    axes[3,4].axis('equal')

    c = axes[3,5].contourf(X, Y, (true - pred), levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[3,5])
    axes[3,5].set_title(r'Difference $\tau_{xy}$')
    axes[3,5].axis('equal')

    c = axes[3,6].contourf(X, Y, mse, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[3,6])
    axes[3,6].set_title(r'MSE $\tau_{xy}$')
    axes[3,6].axis('equal')





    true = true_tau_yz
    pred = predicted_tau_yz
    c = axes[4,0].contourf(X, Y, true, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[4,0])
    axes[4,0].set_title(r'True $\tau_{yz}$')
    axes[4,0].axis('equal')

    c = axes[4,1].contourf(X, Y, pred, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[4,1])
    axes[4,1].set_title(r'Infered $\tau_{yz}$')
    axes[4,1].axis('equal')

    mse = per_pixel_mse(true, pred)

    c = axes[4,2].contourf(X, Y, ssim(true, pred, full=True, data_range=true.max() - true.min())[1], levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[4,2])
    axes[4,2].set_title(r'Structural Similarity in $\tau_{yz}$')
    axes[4,2].axis('equal')


    c = axes[4,3].contourf(X, Y, per_pixel_psnr(true, pred, data_range=true.max() - true.min()), levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[4,3])
    axes[4,3].set_title(r'Structural Similarity in $\tau_{yz}$')
    axes[4,3].axis('equal')

    c = axes[4,4].contourf(X, Y, ((pred - true)/true + eps), levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[4,4])
    axes[4,4].set_title(r'Fractional Difference $\tau_{yz}$')
    axes[4,4].axis('equal')

    c = axes[4,5].contourf(X, Y, (true - pred), levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[4,5])
    axes[4,5].set_title(r'Difference $\tau_{yz}$')
    axes[4,5].axis('equal')

    c = axes[4,6].contourf(X, Y, mse, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[4,6])
    axes[4,6].set_title(r'MSE $\tau_{yz}$')
    axes[4,6].axis('equal')



    true = true_tau_xz
    pred = predicted_tau_xz
    c = axes[5,0].contourf(X, Y, true, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[5,0])
    axes[5,0].set_title(r'True $\tau_{xz}$')
    axes[5,0].axis('equal')

    c = axes[5,1].contourf(X, Y, pred, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[5,1])
    axes[5,1].set_title(r'Infered $\tau_{xz}$')
    axes[5,1].axis('equal')

    mse = per_pixel_mse(true, pred)

    c = axes[5,2].contourf(X, Y, ssim(true, pred, full=True, data_range=true.max() - true.min())[1], levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[5,2])
    axes[5,2].set_title(r'Structural Similarity in $\tau_{xz}$')
    axes[5,2].axis('equal')


    c = axes[5,3].contourf(X, Y, per_pixel_psnr(true, pred, data_range=true.max() - true.min()), levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[5,3])
    axes[5,3].set_title(r'Structural Similarity in $\tau_{xz}$')
    axes[5,3].axis('equal')

    c = axes[5,4].contourf(X, Y, ((pred - true)/true + eps), levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[5,4])
    axes[5,4].set_title(r'Fractional Difference $\tau_{xz}$')
    axes[5,4].axis('equal')

    c = axes[5,5].contourf(X, Y, (true - pred), levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[5,5])
    axes[5,5].set_title(r'Difference $\tau_{xz}$')
    axes[5,5].axis('equal')

    c = axes[5,6].contourf(X, Y, mse, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=axes[5,6])
    axes[5,6].set_title(r'MSE $\tau_{xz}$')
    axes[5,6].axis('equal')
    

    fig.suptitle(f'{title_name}')
    plt.tight_layout()

    if(save == True):
        plt.savefig(filename, dpi=700)
    plt.show()



def print_losses(train_loss_file_name=None, val_loss_file_name=None, out_plot=None):
    plt.figure(figsize=(8, 5))

    if train_loss_file_name is not None:
        train_df = pd.read_csv(train_loss_file_name)
        plt.plot(train_df["epoch"], train_df["train_loss"], label="Training Loss")

    if val_loss_file_name is not None:
        val_df = pd.read_csv(val_loss_file_name)
        plt.plot(val_df["epoch"], val_df["val_loss"], label="Validation Loss")

    plt.xlabel("it")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)

    if out_plot is not None:
        plt.savefig(out_plot, dpi=300, bbox_inches="tight")
        plt.close()
    plt.show()
    












