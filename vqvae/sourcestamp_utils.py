from astropy.io import fits
from astropy.stats import sigma_clip, sigma_clipped_stats
from matplotlib import pyplot as plt
import numpy as np
from numpy import mean
from os import listdir
import PIL
from PIL import Image
import scipy.ndimage as ndimage
from sklearn.utils import shuffle


def loadSourceDetails(infile):
    return np.genfromtxt(infile, dtype='str', delimiter="#")


def getUniqueSources(dr):
    coords = []
    files =  listdir(dr)
    for fn in files:
        c = fn.split('_')[0]
        if not c in coords:
            coords.append(c)
    return coords


def loadImg(f, dr = "", dim = 128):
    #Load images from the given file locations and resize to the correct dimensions
    h = fits.open(dr + '/' + f)
    d = h[0].data
    d = np.array(d)
    d = np.nan_to_num(d)
    
    #m = sigma_clip(d, sigma=5, sigma_upper=10, cenfunc=mean, copy=False)
    pilim = Image.fromarray(d)
    pilim = pilim.resize((dim, dim))
    
    imdat = np.array(pilim)
    imdat = np.reshape(imdat, (dim, dim, 1))
    return imdat


def loadImageArray(filenames, dr = "", dim=128, sigma_clip = False, sigma_coeff=3.0):
    img_out = []
    fns     = []
    
    try:
        # Load if previously saved
        t = np.load(dr + "_" + str(dim) + ".npz")
        i = t.files
        img_out = t[i[0]] 
        fns     = t[i[1]]
        
        # Check if this is the right cache
        print("Loaded " + dr + " at resolution " + str(dim) + "x" + str(dim) + " from cache.")
        
    except IOError:
        
        # Does not exist, create and save
        print("Loading " + dr + " images for the first time, this might take a while. ")
        fns = filenames
        for f in fns:
            img_out.append(loadImg(f, dr, dim))
        np.savez(dr + "_" + str(dim), img_out, fns)

    # Sigma clip if required
    if sigma_clip:
        for i in range(len(img_out)):
            im = img_out[i]
            mean, median, sigma = sigma_clipped_stats(im, sigma=sigma_coeff, maxiters=10)
            im[im < median + sigma_coeff*sigma] = np.nanmin(im)
            img_out[i] = im
    
    # Normalise and remove NaNs
    img_out = np.array(img_out)
    for i in range(len(img_out)):
        im = img_out[i] 
        im = np.nan_to_num(im)
        im -= np.nanmin(im)
        m = np.nanmax(im)
        if m != 0:
            im /= m
            im -= 0.5
        img_out[i] = im
    
    
    return img_out, fns



def loadMultiImageArrays(fn_arrs, dr_arr, dim=150, do_shuffle=True, sigma_clip=True, sigma_coeff=3.0):
    out  = []
    fns = []
    for i in range(len(fn_arrs)):
        f = fn_arrs[i]
        d = dr_arr[i]
        o, f = loadImageArray(f, d, dim, sigma_clip, sigma_coeff)
        if do_shuffle:
            o, f = shuffle(o, f)
        out.append(o)
        fns.append(f)
    return out, fns

