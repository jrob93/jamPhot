#!/usr/bin/env python
# coding: utf-8

# In[1]:

print("launch and load modules")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
import os
from optparse import OptionParser

from astropy.time import Time
import datetime
from astroquery.jplhorizons import Horizons
from astropy.stats import sigma_clipped_stats
from astropy.io import fits
from photutils.detection import DAOStarFinder
from photutils.psf import (IntegratedGaussianPRF, DAOGroup, IterativelySubtractedPSFPhotometry)
from photutils.background import MMMBackground
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.visualization.wcsaxes import WCSAxes
from astropy import visualization as aviz
from photutils.aperture import (ApertureStats, CircularAnnulus, CircularAperture)
from astropy.stats import SigmaClip
from ccdproc import ImageFileCollection
from astropy.wcs import WCS

# Script to do PSF and aperture photometry on images from the Danish Telescope
# Run this script in the Science directory and declare which night is to be analysed using the command line argument:
parser = OptionParser()
parser.add_option( "-n", "--night", dest="night", help="observing night", metavar="NSIDE" )

(options,args)=parser.parse_args()

if options.night:
    night=options.night
else:
    night="20230725"


# In[2]:


data_path = os.getcwd()
print("analyse {}/{}".format(data_path,night))

# In[3]:

# Danish Telescope pixel scale
pixel_scale = 0.39 # arcsec


# In[4]:


files = glob.glob("{}/{}/*wcs.fits".format(data_path,night))


# In[5]:


# get all fits file data and save to file
im_collection=ImageFileCollection(filenames=files)
df_fits = im_collection.summary.to_pandas()
df_fits["frame"] = df_fits["file"].str.split("/").str[-1]
df_fits["night"] = df_fits["file"].str.split("/").str[-2]
df_fits["jd-mid"] = df_fits["jd"] + ((df_fits["exptime"]/2.0)/(24.0*60.0*60.0))
# drop any random flat field that snuck through (imagetyp = LIGHT)
df_fits = df_fits[~df_fits["frame"].str.contains("flat")]
df_fits = df_fits.reset_index(drop=True)
df_fits.to_csv("{}/{}/df_fits_{}.csv".format(data_path,night,night))


# In[6]:


# do psf fitting to get source positions and seeing for the frame
df_phot = pd.DataFrame()
fwhm = []

for f in files:

    print(f)

    fname_phot = "{}/df_phot_{}.csv".format("/".join(f.split("/")[:-1]),f.split("/")[-1].split(".fits")[0])

    if os.path.isfile(fname_phot):
        print("load {}".format(fname_phot))
        df = pd.read_csv(fname_phot, index_col = 0)

        ### if file is loaded and the ra/dec of detections are not present, calculate
        if not np.isin(["ra_0","dec_0","ra_fit","dec_fit"],df.columns).all():

            print("determine ra/dec of detections")

            hdu = fits.open(f)
            hdr = hdu[0].header
            wcs = WCS(hdr)

            # add the ra and dec of detections
            for x,y in zip(["x_0","x_fit"],["y_0","y_fit"]):
                c = wcs.pixel_to_world(df[x], df[y])
                df[["ra_"+x.split("_")[-1],"dec_"+y.split("_")[-1]]] = np.array([c.ra.degree,c.dec.degree]).T

            df.to_csv(fname_phot)


    else:

        hdu = fits.open(f)
        img = hdu[0].data
        hdr = hdu[0].header
        wcs = WCS(hdr)

        print(img.shape)

        #output stats
        mean, median, std = sigma_clipped_stats(img, sigma=2.5)
        print(mean,median,std)

        # Set up psf fitting parmeters and routines
        # TODO: adjust these for different instruments
        fwhm_psf=4.0
        sigma_psf=fwhm_psf/2.354
    #     star_find = DAOStarFinder(fwhm=fwhm_psf, threshold=10.0*std,peakmax=60000.0,sigma_radius=5.0,sharphi=1.0)
        star_find = DAOStarFinder(fwhm=fwhm_psf, threshold=5.0*std, peakmax=60000.0)
    #     starfind = DAOStarFinder(10*std, fwhm_psf)

        daogroup = DAOGroup(9.0)
        med_bkg = MMMBackground()

        # define mdel psf as a gaussian
        model_psf = IntegratedGaussianPRF(sigma=sigma_psf)
        model_psf.sigma.fixed = False

        # define fitting routine to use
        fitter = LevMarLSQFitter()

        # create function to perform psf fitting on frame
        my_photometry = IterativelySubtractedPSFPhotometry(finder=star_find, group_maker=daogroup, bkg_estimator=med_bkg, psf_model=model_psf, fitter=fitter, fitshape=(11,11),niters=2)

        # set psf model, run source detection and photometry
        table = my_photometry(image=img-median)

        # calculate fwhm values
        table['fwhmarcsec']=pixel_scale*2.354*table['sigma_fit']
        table['fwhmpix']=2.354*table['sigma_fit']
        df = table.to_pandas()
        df["file"] = f

        # add the ra and dec of detections
        for x,y in zip(["x_0","x_fit"],["y_0","y_fit"]):
            c = wcs.pixel_to_world(df[x], df[y])
            df[["ra_"+x.split("_")[-1],"dec_"+y.split("_")[-1]]] = np.array([c.ra.degree,c.dec.degree]).T

        df.to_csv(fname_phot)

    df_phot = pd.concat([df_phot,df]).reset_index(drop=True)

    indiv_fwhm = df_phot['fwhmpix']
    fwhm.append((np.median(indiv_fwhm))) #get average fwhm for each image

fwhm = np.array(fwhm)


# In[7]:


# do aperture phot

df_stats = pd.DataFrame()

for i,f in enumerate(files):

    print(f)

    fname_stats = "{}/df_stats_{}.csv".format("/".join(f.split("/")[:-1]),f.split("/")[-1].split(".fits")[0])

    if os.path.isfile(fname_stats):
        print("load {}".format(fname_stats))
        df = pd.read_csv(fname_stats, index_col = 0)

        ### if file is loaded and the ra/dec of detections are not present, calculate
        if not np.isin(["ra_centroid","dec_centroid"],df.columns).all():

            print("determine ra/dec of detections")

            hdu = fits.open(f)
            hdr = hdu[0].header
            wcs = WCS(hdr)

            # add the ra and dec of detections
            c = wcs.pixel_to_world(df["xcentroid"], df["ycentroid"])
            df[["ra_centroid","dec_centroid"]] = np.array([c.ra.degree,c.dec.degree]).T

            df.to_csv(fname_stats)

    else:

        hdu = fits.open(f)
        img = hdu[0].data
        hdr = hdu[0].header
        wcs = WCS(hdr)
        print(img.shape)

        df = df_phot[df_phot["file"]==f]

        #define aperture and annuli positions
        positions = np.array([df['x_fit'], df['y_fit']])
        positions = np.swapaxes(positions, 0, 1)
        # use the median fwhm of each frame for aperture size
        apertures = CircularAperture(positions, r = fwhm[i])
        annuli = CircularAnnulus(positions, r_in=fwhm[i]+5, r_out=fwhm[i]+10)

        #run photometry on apertures
        sigclip = SigmaClip(sigma = 3.0, maxiters = 10)
        aper_stats = ApertureStats(img, apertures, sigma_clip=None)
        bkg_stats = ApertureStats(img, annuli, sigma_clip=sigclip)

        #calculate and save background subtracted flux values
        total_bkg = bkg_stats.median * aper_stats.sum_aper_area.value
        apersum_bkgsub = aper_stats.sum - total_bkg
        aper_stats = aper_stats.to_table()
        aper_stats['bkgsub'] = apersum_bkgsub  #save background subtracted count
        aper_stats['flux_fit_aper'] = apersum_bkgsub #save counts before distortion correction
        aper_stats['sum_err'] = total_bkg
        aper_stats.remove_columns(['sky_centroid'])

        df = aper_stats.to_pandas()
        df = df.rename({"id":"id_aper"},axis=1)

        ### calculate the ra/dec of detections
        c = wcs.pixel_to_world(df["xcentroid"], df["ycentroid"])
        df[["ra_centroid","dec_centroid"]] = np.array([c.ra.degree,c.dec.degree]).T

        df.to_csv(fname_stats)

    df_stats = pd.concat([df_stats,df]).reset_index(drop=True)

df_phot = pd.concat([df_phot,df_stats],axis=1)


# In[ ]:
