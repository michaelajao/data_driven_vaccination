import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


# plt.rcParams.update({
#     # Line styles and markers
#     "lines.linewidth": 2,  # from mpl.rcParams['lines.linewidth']
#     "lines.markersize": 5,  # Default marker size updated from mpl.rcParams['lines.markersize']
    
#     # Figure size and layout
#     "figure.figsize": [15, 10],  # Adjusted to match the specified figure size
#     "figure.autolayout": True,  # Enable auto layout
    
#     # Axes and grid
#     "axes.prop_cycle": cycler('color', ['k']) * cycler('marker', ['', 'o', '^', 's', 'v']) *
#                        cycler('linestyle', ['-', '--', ':', (0, (5, 2, 5, 5, 1, 4))]),
#     "axes.titlesize": 26,  # Title size from your settings
#     "axes.labelsize": 14,  # Label size from your settings
#     "axes.spines.top": False,  # Hide the top spine
#     "axes.spines.right": False,  # Hide the right spine
#     # "axes.grid": True,  # Enable grid
#     # "grid.color": "0.75",  # Grid color
    
#     # Font settings
#     "font.family": "serif",
#     "font.size": 14,  # Default text sizes
#     "font.sans-serif": ["Helvetica"],
    
#     # Legend
#     "legend.fontsize": "medium",  # Legend font size
#     "legend.frameon": False,  # No frame around the legend
#     "legend.loc": "best",  # Automatic legend placement
#     # LaTeX
#     # "text.usetex": True,  # Enable LaTeX rendering
    
# })

plt.style.use('seaborn-v0_8-white')
plt.rcParams.update(
    {
        "lines.linewidth": 2,
        "font.family": "serif",
        "axes.titlesize": 20,
        "axes.labelsize": 12,
        "figure.figsize": [15, 8],
        "figure.autolayout": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # "axes.grid": True,
        # "grid.color": "0.75",
        "legend.fontsize": "medium",
        "legend.frameon": False,
        "legend.loc": "best",
        "font.size": 14,
        "font.sans-serif": ["Helvetica"],
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.format": "png",
        "savefig.bbox": "tight",
        "legend.loc": "best",
        
    }
)


first_data = "../../data/interim/pickle/nhs_region_data.pkl"
second_data = "../../data/raw/pickle/covid19_data.pkl"
data_column = "date"

data1 = pd.read_pickle(first_data)
data2 = pd.read_pickle(second_data)

data1["date"] = pd.to_datetime(data1["date"])
data2["date"] = pd.to_datetime(data2["date"])

#national lockdown periods
lockdown_periods = [
    ("2020-03-23", "2020-05-10"),
    ("2020-11-05", "2020-12-02"),
    ("2021-01-06", "2021-03-08"),
]
# heatmap of the covid-19 ICU hospalisation data

pivot_ICU_data = data1.pivot_table(index="date", columns="areaName", values="covidOccupiedMVBeds", aggfunc=np.sum)
pivot_ICU_data.index = pd.to_datetime(pivot_ICU_data.index)

# Use seaborn's heatmap function for better color scaling and integration
sns.heatmap(pivot_ICU_data, cmap="viridis", cbar_kws={'label': 'Covid-19 ICU Hospitalisation'})

# Title and labels
plt.title("Covid-19 ICU Hospitalisation")
plt.xlabel("Region")
plt.ylabel("Date")

# Rotate the x-axis tick labels for better readability
plt.xticks(rotation=45)

# Sparse y-axis ticks for better readability
plt.yticks(np.arange(0, len(pivot_ICU_data.index), step=max(1, len(pivot_ICU_data.index)//10)), pivot_ICU_data.index.strftime("%Y-%m-%d")[::max(1, len(pivot_ICU_data.index)//10)], rotation=0)

# Use a tight layout
plt.tight_layout()

# Show the plot
plt.show()


    
# Plot the data
for region in data2["region"].unique():
    region_data = data2[data2["region"] == region]
    plt.plot(region_data["date"], region_data["new_confirmed"], label=region)
    
for start, end in lockdown_periods:
    plt.axvspan(start, end, alpha=0.3, color="grey", label='Lockdown Periods' if start == '2020-03-23' else "")
    
    
plt.title("New Confirmed COVID-19 Cases Over Time by NHS Region")
plt.xlabel("Date")
plt.ylabel("New Confirmed Cases per 100,000 Population")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../../reports/figures/new_confirmed_cases.png")
plt.show()


# Plot the data for all the regions in ICU
for region in data1["areaName"].unique():
    region_data = data1[data1["areaName"] == region]
    plt.plot(region_data["date"], region_data["covidOccupiedMVBeds"], label=region)
    
for start, end in lockdown_periods:
    plt.axvspan(start, end, alpha=0.3, color="grey", label='Lockdown Periods' if start == '2020-03-23' else "")
    
plt.title("Covid-19 ICU Hospitalisation Over Time by NHS Region")
plt.xlabel("Date")
plt.ylabel("Covid-19 ICU Hospitalisation")
plt.legend(loc="upper right")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../../reports/figures/covid_19_icu_hospitalisation.png")
plt.show()



# # function to load the data
# def load_data(data, file_name):
#     if os.path.exists(data):
#         data = pd.read_pickle(data)
#         return data
#     else:
#         print(f"File {file_name} does not exist")
        
#     return pd.DataFrame()
    
    
# map the region to the data and merge the data

# def map_regions_and_merge(df1, df2, mapping, merge_columns, drop_columns):
#     """Map regions in one dataframe to another and merge them."""
#     df2["mapped_region"] = df2["region"].map(mapping)
#     merged_df = pd.merge(
#         df1, df2, how="inner", left_on=merge_columns, right_on=["date", "mapped_region"]
#     )
#     merged_df.drop_duplicates(inplace=True)
#     merged_df.drop(columns=drop_columns, inplace=True, errors="ignore")
#     return merged_df

# merge_columns = ["date", "areaName"]
# drop_columns = ["mapped_region", "areaType", "areaCode"]

# merge_data = map_regions_and_merge(data1, data2, region_mapping, merge_columns, drop_columns)

