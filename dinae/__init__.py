"""
DINAE (new modifications by mbeaucha) 
"""

__author		= "Maxime Beauchamp"
__version__ 		= "0.0.1"
__last_modification__	= "2019-12-10"

##################################
# Standard lib
##################################
import sys
import os
import shutil
import time as timer
from os.path import join as join_paths
from datetime import date, datetime, timedelta
import itertools
import warnings
import traceback
import re
import functools
import configparser
import builtins
import time
from time import sleep
import multiprocessing
import mkl
import cv2
from tqdm import tqdm
from collections import OrderedDict
import pickle
import argparse

assert sys.version_info >= (3,5), "Need Python>=3.6"

##################################
# Config
##################################
dirs = {}

# Define paths
datapath="/mnt/groupadiag302/WG8/data/gridded_data_swot_wocorr"
basepath="/home/AI4OAC-VM8/DINAE"

print("Initializing DINAE libraries...",flush=True)

##################################
# Scientific and mapping
##################################
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import pandas as pd
import shapely
from shapely import wkt
#import geopandas as gpd
from cartopy import crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.sparse import diags
from scipy.stats import multivariate_normal
from scipy.ndimage.morphology import distance_transform_edt as bwdist
import scipy.ndimage as nd
from scipy.interpolate import RegularGridInterpolator
import skill_metrics as sm
import xarray as xr
from netCDF4 import Dataset
from pyflann import *

##################################
# Tools
##################################
import tensorflow as tf
import keras
from keras.constraints import Constraint
from keras import backend as K
from .mods.import_Datasets    import *
from .mods.define_Models      import *
from .mods.define_Classifiers import *
from .mods.FP_Learning        import *
from .mods.GB_Learning        import *
from .mods.tools              import *
from .mods.graphics           import *

print("...Done") # ... initializing Libraries



