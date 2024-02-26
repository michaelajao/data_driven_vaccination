import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
# plt.style.available
plt.rcParams.update({
    "lines.linewidth": 2,
    "font.family": "serif",
    "axes.titlesize": 20,
    "axes.labelsize": 12,
    "figure.figsize": [15, 8],
    "figure.autolayout": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "0.75",
    "legend.fontsize": "medium",
    "legend.frameon": False,
    "legend.loc": "best",
    "font.size": 14,
    "font.sans-serif": ["Helvetica"],
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 300
})
plt.rcParams.keys()
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

colunms = ["new_confirmed", "new_deceased"]


for column in colunms:
    for region in data2["region"].unique():
        region_data = data2[data2["region"] == region]
        plt.plot(region_data["date"], region_data[column], label=region)
        
    for start, end in lockdown_periods:
        plt.axvspan(start, end, color="grey", alpha=0.2, label='Lockdown Periods' if start == '2020-03-23' else "")
        
        
    plt.title(f"{column.replace('_', ' ').title()} Over Time by NHS Region")
    plt.xlabel("Date")
    plt.ylabel(f"{column.replace('_', ' ').title()}")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"../../reports/figures/{column}_over_time_by_nhs_region.png")
    plt.show()

columns = ["covidOccupiedMVBeds", "hospitalCases", "newAdmissions"]

for column in columns:
    for region in data1["areaName"].unique():
        region_data = data1[data1["areaName"] == region]
        plt.plot(region_data["date"], region_data[column], label=region)
        
    for start, end in lockdown_periods:
        plt.axvspan(start, end, color="grey", alpha=0.2, label='Lockdown Periods' if start == '2020-03-23' else "")
    plt.title(f"{column.replace('_', ' ').title()} Over Time by NHS Region")
    plt.xlabel("Date")
    plt.ylabel(f"{column.replace('_', ' ').title()}")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"../../reports/figures/{column}_over_time_by_nhs_region.png")
    plt.show()

pivot_ICU_data = data1.pivot_table(index="date", columns="areaName", values="covidOccupiedMVBeds", aggfunc=np.sum)
pivot_ICU_data.index = pd.to_datetime(pivot_ICU_data.index)

# Use seaborn's heatmap function for better color scaling and integration
sns.heatmap(pivot_ICU_data, cmap="viridis", cbar_kws={'label': 'Covid-19 ICU Hospitalisation'})
plt.title("Covid-19 ICU Hospitalisation")
plt.xlabel("Region")
plt.ylabel("Date")
plt.xticks(rotation=45)
plt.yticks(np.arange(0, len(pivot_ICU_data.index), step=max(1, len(pivot_ICU_data.index)//10)), pivot_ICU_data.index.strftime("%Y-%m-%d")[::max(1, len(pivot_ICU_data.index)//10)], rotation=0)
plt.tight_layout()
plt.savefig(f"../../reports/figures/covid-19_icu_hospitalisation_heatmap.png")
plt.show()

pivot_hospital_cases_data = data1.pivot_table(index="date", columns="areaName", values="hospitalCases", aggfunc=np.sum)
pivot_hospital_cases_data.index = pd.to_datetime(pivot_hospital_cases_data.index)

# Use seaborn's heatmap function for better color scaling and integration
sns.heatmap(pivot_hospital_cases_data, cmap="viridis", cbar_kws={'label': 'Hospital Cases'})
plt.title("Hospital Cases")
plt.xlabel("Region")
plt.ylabel("Date")
plt.xticks(rotation=45)
plt.yticks(np.arange(0, len(pivot_hospital_cases_data.index), step=max(1, len(pivot_hospital_cases_data.index)//10)), pivot_hospital_cases_data.index.strftime("%Y-%m-%d")[::max(1, len(pivot_hospital_cases_data.index)//10)], rotation=0)
plt.tight_layout()
plt.savefig(f"../../reports/figures/hospital_cases_heatmap.png")
plt.show()


pivot_new_admissions_data = data1.pivot_table(index="date", columns="areaName", values="newAdmissions", aggfunc=np.sum)
pivot_new_admissions_data.index = pd.to_datetime(pivot_new_admissions_data.index)

# Use seaborn's heatmap function for better color scaling and integration
sns.heatmap(pivot_new_admissions_data, cmap="viridis", cbar_kws={'label': 'New Admissions'})
plt.title("New Admissions")
plt.xlabel("Region")
plt.ylabel("Date")
plt.xticks(rotation=45)
plt.yticks(np.arange(0, len(pivot_new_admissions_data.index), step=max(1, len(pivot_new_admissions_data.index)//10)), pivot_new_admissions_data.index.strftime("%Y-%m-%d")[::max(1, len(pivot_new_admissions_data.index)//10)], rotation=0)
plt.tight_layout()
plt.savefig(f"../../reports/figures/new_admissions_heatmap.png")
plt.show()