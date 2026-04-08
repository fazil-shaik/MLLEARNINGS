import argparse
import json
import os
import sys
import time
from datetime import datetime,timedelta,timezone
import warnings
from pathlib import Path



#importing needed models
import matplotlib
matplotlib.use('Agg')  #non interactive backend 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sgp4.api import Satrec,jday 
from skyfield.api import EarthSatellite,load,wgs84
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression,ElasticNet,Lasso,Ridge
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor



warnings.filterwarnings("ignore")



#data constants!

DATA_DIR = Path(__file__).parent/"data"
TLE_CACHE = DATA_DIR /"tle_cache.json"

SATTELEITES={
    25544:'ISS (ZARYA)',
    20580:'HST (HUBBLE SPACE TELESCOPE)',
}


CELESTARL_URL = 'https://celestark.org/NORAD/elements/gp.php'

#phase 1. ---fetching the data from api

def fetch_tle(catnr:int)->dict | None:
    """Fetch Tle data from satelite"""

    import requests


    results = None

    try:
        resp = requests.get(f"{CELESTARL_URL}?CATNR={catnr}&format=JSON",timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if data and len(data) > 0:
            result = data[0]
    except Exception as e:
        print(f" Failed to fetch {catnr}:{e}") 


    #fetch data from satellite in TLE formate and in TLE lines

    try:
        resp = requests.get(f"{CELESTARL_URL}?CATNR={catnr}&format=TLE",timeout=15) 
        resp.raise_for_status()
        lines = [l.strip() for l in resp.text.splitlines().splitlines() if l.strip()]
        if len(lines)>=2:
            if lines[0].startswith("1 "):
                tle_line1,tle_line2 = lines[0],lines[1]
            else:
                tle_line1,tle_line2 = lines[1],lines[2]


            if result is None:

                result = {"OBJECT_NAME":lines[0] if not lines[0].startswith("1 ") else f"SAT-{catnr}"}

            result["TLE_LINE1"] = tle_line1
            result["TLE_LINE2"] = tle_line2 

    except Exception  as e:
        print(f" Failed to fetch {catnr}:{e}") 

    if result and "TLE_LINE1" not in result:
        print(f"NO Tle result data found for catnr:{catnr}")

        return None
    return result


#Fetch the data and cache it locally

#phase2:



        

