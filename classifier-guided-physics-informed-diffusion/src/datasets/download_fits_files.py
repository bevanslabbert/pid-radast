import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.mast import Observations

# 1. Load the data from the text file
# We use sep='\s+' to handle any amount of whitespace (tabs or spaces)
df = pd.read_csv('./combined_catalogue_data.txt', sep='\s+', header=None)

# 2. Extract RA and Dec (Columns 0 and 1)
# Create SkyCoord objects to ensure the format is correct for the query
coords = SkyCoord(ra=df[0].values*u.degree, dec=df[1].values*u.degree)

print(f"Loaded {len(coords)} coordinate pairs. Starting search...")

# 3. Loop through coordinates and download
for i, coord in enumerate(coords):
    print(f"--- Processing Point {i+1}: {coord.ra.deg}, {coord.dec.deg} ---")
    
    # 1. Query the region
    obs_table = Observations.query_region(coord, radius=0.01 * u.deg)
    
    # 2. FILTER: Remove TESS FFIs and empty results
    # TESS FFIs are usually what cause the "Observation list is empty" error on get_product_list
    valid_obs = obs_table[obs_table['obs_collection'] != 'TESS']
    
    if len(valid_obs) > 0:
        try:
            # Pick the first valid non-TESS observation
            products = Observations.get_product_list(valid_obs[0])
            
            # Filter for Science products (prevents downloading logs/previews)
            science_products = products[products['productGroupDescription'] == 'Minimum Recommended Products']
            
            if len(science_products) > 0:
                manifest = Observations.download_products(science_products[0], download_dir='./fits_data')
                print(f"Downloaded: {manifest['Local Path'][0]}")
            else:
                print("No science-grade products found.")
                
        except Exception as e:
            print(f"Error fetching products for point {i}: {e}")
    else:
        print("No valid (non-TESS) observations found at this location.")