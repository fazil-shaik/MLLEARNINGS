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
def fetch_all_tle(catalog_numbers: list[int]) -> dict:
    print("Fetching TLE data for satellites...")
    print("=" * 60)
    tle_data = {}
    fetched_any = False
    for catnr in catalog_numbers:
        satname = SATTELEITES.get(catnr, f"SAT-{catnr}")
        print(f"\n Fetching {satname}  with (CATNR{catnr})...")
        result = fetch_tle(catnr)
        if result:
            tle_data[str(catnr)] = result
            fetched_any = True
        else:
            print(f"Failed to fetch TLE data for {satname} .")
    
    # save cached data to json file
    if fetched_any:
        DATA_DIR.mkdir(exist_ok=True)
        with open(TLE_CACHE, "w") as f:
            json.dump(tle_data, f, indent=2)
        print(f"\n[TLE data] cached to {TLE_CACHE}")
    
    # fallback to cache if failed!
    if not tle_data and TLE_CACHE.exists():
        print("Loading cached TLE data from file...")
        with open(TLE_CACHE) as f:
            tle_data = json.load(f)
        print(f"[ok] loaded TLE data {len(tle_data)} from cache")
    
    if not tle_data:
        print("NO TLE data found!")
        sys.exit(1)
    
    return tle_data


# phase 2: Dataset generation!
def generate_dataset(tle_entry: dict, num_samples: int = 10000) -> pd.DataFrame:
    print("\n" + "=" * 60)
    line1 = tle_entry["TLE_LINE1"]
    line2 = tle_entry["TLE_LINE2"]
    sat_name = tle_entry.get("OBJECT_NAME", "UNKNOWN")

    # parse the TLE data using sgp4
    satellite_sgp4 = Satrec.twoline2rv(line1, line2)
    # convert the lat and lon to ECEF coordinates using skyfield
    ts = load.timescale()
    satellite_sf = EarthSatellite(line1, line2, sat_name, ts)

    # extract the orbital params from TLE
    inclination = tle_entry.get("INCLINATION", satellite_sgp4.inclo * 180 / np.pi)
    eccentricity = tle_entry.get("ECCENTRICITY", satellite_sgp4.ecco)
    raan = tle_entry.get("RAAN", satellite_sgp4.nodeo * 180 / np.pi)
    arg_perigee = tle_entry.get("ARG_OF_PERIGEE", satellite_sgp4.argpo * 180 / np.pi)
    mean_motion = tle_entry.get("MEAN_MOTION", satellite_sgp4.no_kozai * 1440 / (2 * np.pi))
    bstar = tle_entry.get("BSTAR", satellite_sgp4.bstar)
    epoch_str = tle_entry.get("EPOCH", None)
    
    if epoch_str:
        epoch_dt = datetime.fromisoformat(epoch_str.replace("Z", "+00:00"))
        if epoch_dt.tzinfo is None:
            epoch_dt = epoch_dt.replace(tzinfo=timezone.utc)
    else:
        year = satellite_sgp4.epochyr
        if year < 57:
            year += 2000
        else:
            year += 1900
        epoch_dt = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=satellite_sgp4.epochdays - 1)
    
    rows = []
    # for i in range(num_samples):



