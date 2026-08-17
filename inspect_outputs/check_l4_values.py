import rasterio, numpy as np
import json
import os
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    _config = json.load(f)

ages = ["2000", "2006", "2011", "2015", "2020", "2033", "2040", "2048",
        "2056", "2068", "2094", "2100", "2113", "2120", "2133", "2140",
        "2154", "2165", "2180", "2200", "2210", "2220", "2230", "2240",
        "2250", "2270", "2290", "2300", "2315", "2331", "2350", "2370",
        "2383", "2393", "2408", "2420", "2444", "2463", "2475", "2489",
        "2500", "2518", "2535", "2545"]

ages = ["2120"]

le_outputs = _config["LE_outputs"]
unique_vals = []

for age in ages:
    print(age)
    l4_raster = os.path.join(le_outputs,age,"level4_out_LE_DT.tif")
    with rasterio.open(l4_raster) as src:
        print("Band count:", src.count)
        print("Band descriptions:", src.descriptions)  # often carries variable names
        for i in range(1, src.count + 1):
            band = src.read(i)
            vals, counts = np.unique(band, return_counts=True)
            print(f"Band {i}:", dict(zip(vals.tolist(), counts.tolist())))

print(f"Unique values: {unique_vals}")