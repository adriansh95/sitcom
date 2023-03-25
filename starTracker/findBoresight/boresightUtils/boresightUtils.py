import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import os

from astride import Streak
from astropy.io import fits
from scipy.optimize import differential_evolution, minimize
from scipy.stats import binned_statistic
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter

def gaussian_blur_image(arcImg, configDict):
    gaussianBlurSigma = configDict['gaussianBlurSigma']
    arcImgCp = gaussian_filter(arcImg, sigma=gaussianBlurSigma)
    
    return arcImgCp

def make_threshold_image(arcImg, configDict):
    threshold = configDict['threshold']
    arcImgCp = gaussian_blur_image(arcImg, configDict)
    brightPxs = np.zeros(arcImg.shape)
    brightPxs[arcImgCp > threshold] = 1
    
    return brightPxs

def convert_fov_degs_px(whDeg, plateScale):
    wDeg, hDeg = whDeg
    nXPx = wDeg * 3600 / plateScale
    nYPx = hDeg * 3600 / plateScale
    whPx = (int(nXPx), int(nYPx))
    
    return whPx
    
def xy_image_coord_transform(r, img, inverse=False):
    # Works with arrays of shape (2,) and (2, N)
    height, width = img.shape
    
    if inverse:
        rPrime0 = r[1]
        rPrime1 = height - r[0]
    else:
        rPrime0 = height - r[1]
        rPrime1 = r[0]

    rPrime = np.array([rPrime0, rPrime1]).reshape(r.shape)
    
    return rPrime

def score_center(center, img, dr):
    # Center in image coords (row, col)
    height, width = img.shape
    maxR = np.linalg.norm(np.array([height, width]))
    rBins = np.arange(0, int(maxR + 1), dr)
    
    # Calculate bright px distances from given center
    center = center.reshape((2, 1))
    rowBright, colBright = np.where(img > 0)
    brightIdx = np.vstack([rowBright, colBright])
    rBright = brightIdx - center
    rBright = np.linalg.norm(rBright, axis=0)
    
    # Calculate score
    nBrightInBin, binEdges = np.histogram(rBright, bins=rBins)
    brightRadii = binEdges[1:][nBrightInBin > 0]
        
    score = len(brightRadii)
    
    return score
    
def make_arc_image(configDict, **kwargs):
    thetaInit = kwargs.get('thetaInit', 0)
    driftTime = kwargs.get('driftTime', 0)
    thetaInit *= np.pi / 180
    expTime = configDict['expTime']
    slewRate = configDict['slewRate']
    width = configDict['imageWidth']
    height = configDict['imageHeight']
    driftVel = configDict['driftVel']
    rng = configDict['rng']
    xMu = configDict['xMu']
    yMu = configDict['yMu']
    pointingSigma = configDict['pointingSigma']
    nStars = configDict['nStars'] 
    rotM_init = np.array([[np.cos(thetaInit), -np.sin(thetaInit)], [np.sin(thetaInit), np.cos(thetaInit)]])

    fCircle = np.abs(expTime * slewRate / 360)
    phiDot = slewRate * np.pi / 180
    
    vals = rng.standard_normal(2) # Sample gaussian twice

    # Define pointing center
    pointingCenter = np.array([xMu + pointingSigma * vals[0], yMu + pointingSigma * vals[1]]) # x and y

    tSteps = int(20000 * fCircle * 4)
    
    try:
        dt = expTime / tSteps
        times = np.arange(0, expTime + dt, dt)
    except ZeroDivisionError:
        times = np.array([0])

    # Make an array for later use
    arcImg = np.zeros((height, width))

    for i in range(nStars): # Pretty sure this entire operation can be vectorized (tensorized)
        xExt0 = width * rng.random()
        yExt0 = height * rng.random()
        rExt0 = np.array([xExt0, yExt0])
        integratedDrift_t = (times + driftTime) * driftVel
        rExt_t = rExt0.reshape((2, 1)) + integratedDrift_t
        rotM_t = np.array([[[np.cos(phiDot * t), -np.sin(phiDot * t)], 
                            [np.sin(phiDot * t), np.cos(phiDot * t)]] for t in times])
        
        rPrime_t = rExt_t - pointingCenter.reshape((2, 1))
        rPrime_t = rPrime_t.transpose().reshape((rotM_t.shape[0], 2, 1))
        rCam_t = np.matmul(rotM_init, np.matmul(rotM_t, rPrime_t))
        rCam_t = rCam_t.transpose(0, 2, 1).squeeze()
        rCam_t = np.round(rCam_t)
        
        dataType = np.dtype([('x', rCam_t.dtype), ('y', rCam_t.dtype)])
        dataVectors = rCam_t.view(dtype=dataType).squeeze()
        uniqueVectors = np.array(np.unique(dataVectors).tolist())
        
        for rPrime in uniqueVectors:            
            r = pointingCenter + rPrime
            imgIdx = xy_image_coord_transform(r, arcImg)
            row = int(imgIdx[0])
            col = int(imgIdx[1])

            # Some arcs will go out of the image
            if row >= 0 and col >= 0:   
                try:
                    arcImg[(row, col)] += 1
                except IndexError:
                    pass
                
    return arcImg, pointingCenter

def find_boresight_symmetric(im0, im1, **kwargs):
    nTrials = kwargs.get('nTrials', 10)
    optimizerResults = np.zeros((2, nTrials, 2))
    
    for i, im in enumerate([im0, im1]):    
        for iTrial in range(nTrials):
            optimizerResults[i, iTrial] = find_boresight_naive(im, **kwargs)
            
    foo = (optimizerResults[0] + optimizerResults[1]) / 2
    mu = foo.mean(axis=0)
    sigma = foo.std(axis=0)
    result = {'result': mu, 'sigma': sigma}
    
    return result

def find_boresight_naive(im, **kwargs):
    dr = kwargs.get('dr', .1)
    tol = kwargs.get('tol', 0.01)
    doPowell = kwargs.get('doPowell', False)
    height, width = im.shape
    args = (im, dr)
    result = differential_evolution(score_center, ((0, height), (0, width)), 
                                      args=args, tol=tol)
    if doPowell:
        lowBound = result.x - 3
        upBound = result.x + 3
        bounds = ((lowBound[0], upBound[0]), (lowBound[1], upBound[1]))
        result = minimize(score_center, result.x, args=args, method='Powell',
                          bounds=bounds)
    return result.x

def measure_drift(im0, im1):
    autoCorr = fftconvolve(im0, im0[::-1,::-1], mode='same')
    imCorr = fftconvolve(im0, im1[::-1, ::-1], mode='same')
    imCenter = np.array(np.unravel_index(np.argmax(autoCorr), autoCorr.shape))
    peakCorr = np.array(np.unravel_index(np.argmax(imCorr), imCorr.shape))
    
    return imCenter - peakCorr

def calculate_drift_bias(configDict, **kwargs):
    nTrials = kwargs.get('nTrials', 10)
    results = np.zeros((nTrials, 2))
    
    for iTrial in range(nTrials):
        arcImg, pointingCenter = make_arc_image(configDict, **kwargs)
        pcIdx = xy_image_coord_transform(pointingCenter, arcImg)
        brightPxs = make_threshold_image(arcImg, configDict)
        naive_center = find_boresight_naive(brightPxs, **kwargs)
        results[iTrial] = naive_center - pcIdx
        
    return results

def plot_arc_image(img, boresight=None, saveFig=False, figsDir='', figName='arc_image.png'):
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(img, cmap='gray')
    
    if boresight is not None:
        x, y = boresight[1], boresight[0]
        ax.scatter(x, y, marker='x', c='r')
    
    if saveFig:
        filename = os.path.join(figsDir, figName)
        fig.savefig(filename)

    plt.show()

def find_streaks(fname, **kwargs):
    contour_threshold = kwargs.get("contour_threshold", 1.5)
    area_cut = kwargs.get("area_cut", 25)
    radius_dev_cut = kwargs.get("radius_dev_cut", 0.5)
    remove_bkg = kwargs.get("remove_bkg", "map")

    # Read a fits image and create a Streak instance.
    streak = Streak(fname, contour_threshold=contour_threshold,
                    area_cut=area_cut, radius_dev_cut=radius_dev_cut,
                    remove_bkg=remove_bkg) 

    # Detect streaks.
    streak.detect()

    return streak.streaks

def plot_streaks(inFileName, saveFig=False, outPath="./", **kwargs):
    nSelect = kwargs.get("nSelect", 100)

    with fits.open(filename) as hdu:
        img = hdu[0].data

    edges = find_streaks(inFileName, **kwargs)

    df = pd.DataFrame(edges)
    df = df.sort_values(by='perimeter', ascending=False)
    selection = df.index.to_numpy()[:]

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), sharex='all', sharey='all')

    # Plot all contours.
    for n in selection[:nSelect]:
        edge = edges[n]
        ax1.plot(edge['x'], edge['y'], mfc='o', mec='r', ms=5)
        ax1.text(edge['x'][0], edge['y'][1],
                '%d' % (edge['index']), color='b', fontsize=15)

    display(img, ax=ax1, fig=fig)
    fig.tight_layout()
    ax1.set_title("Star Trails")
    fig.savefig(f"{fname}_streaks.png" dpi=120)
