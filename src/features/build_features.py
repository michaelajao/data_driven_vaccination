# merging the data based on region and date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = "../../data/interim/pickle/nhs_region_data.pkl"
path2 = "../../data/raw/pickle/covid19_data.pkl"

data1 = pd.read_pickle(path)
data2 = pd.read_pickle(path2)

data1["date"] = pd.to_datetime(data1["date"])
data2["date"] = pd.to_datetime(data2["date"])

# NHS region mapping 
data1.areaName.unique()
data2.region.unique()

# [East of England, London, Midlands, North East and Yorkshire, North West, South East, South West]

# in data2, the regions need to be mapped to the NHS regions

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

# plotting the data
lockdown_periods = [
    ("2020-03-23", "2020-05-10"),
    ("2020-11-05", "2020-12-02"),
    ("2021-01-06", "2021-03-08"),
]

merged_data["date"] = pd.to_datetime(merged_data["date"])

# drop the columns that are not needed
merged_data.drop(["areaType", "openstreetmap_id", "region"], axis=1, inplace=True)

# save the merged data
merged_data.to_pickle("../../data/processed/merged_nhs_covid_data.pkl")
merged_data.to_csv("../../data/processed/merged_nhs_covid_data.csv")