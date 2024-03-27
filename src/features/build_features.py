# merging the data based on region and date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

path = "../../data/interim/pickle/nhs_region_data.pkl"
path2 = "../../data/raw/pickle/covid19_data.pkl"

data1 = pd.read_pickle(path)
data2 = pd.read_pickle(path2)

data1["date"] = pd.to_datetime(data1["date"])
data2["date"] = pd.to_datetime(data2["date"])

# NHS region mapping 
data1.areaName.unique()
data2.region.unique()

region_mapping = {
    "East of England": "East of England",
    "London Region": "London",
    "West Midlands": "Midlands",
    "East Midlands": "Midlands",
    "North East England": "North East and Yorkshire",
    "Yorkshire and the Humber": "North East and Yorkshire",
    "North West England": "North West",
    "South East England": "South East",
    "South West England": "South West",
}

data2["region"] = data2["region"].map(region_mapping)

# merging the data based on region and date
merged_data = data1.merge(data2, left_on=["areaName", "date"], right_on=["region", "date"], how="inner")

merged_data["areaName"].unique()

merged_data["date"] = pd.to_datetime(merged_data["date"])

# drop the columns that are not needed
merged_data.drop(["areaType", "openstreetmap_id", "region"], axis=1, inplace=True)


os.makedirs("../../data/processed", exist_ok=True)
# save the merged data
merged_data.to_pickle("../../data/processed/merged_nhs_covid_data.pkl")
merged_data.to_csv("../../data/processed/merged_nhs_covid_data.csv")

# Aggregating the data to make an England wide data for all the columns
# only the numerical columns are aggregated
england_data = merged_data.groupby("date").sum().reset_index().drop(columns=["latitude", "longitude", "areaCode", "areaName", "epi_week"], axis=1)

england_data["areaName"] = "England"

# save the england data
england_data.to_pickle("../../data/processed/england_data.pkl")
england_data.to_csv("../../data/processed/england_data.csv")

import tensorflow as tf

tf.__version__
tf.config.list_physical_devices('GPU')