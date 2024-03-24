import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# plt.style.use('seaborn-v0_8-whitegrid')
# plt.style.available
# plt.rcParams.update({
#     "font.family": "serif",
#     "figure.facecolor": "white",
#     "axes.facecolor": "white",
#     "axes.titlesize": 26,
#     "axes.labelsize": 12,
#     "figure.figsize": [15, 8],
#     "figure.autolayout": True,
#     "legend.fontsize": "medium",
#     "legend.frameon": False,
#     "legend.loc": "best",
#     "lines.linewidth": 2.5,
#     "lines.markersize": 10,
#     "font.size": 14,
#     "xtick.labelsize": 12,
#     "ytick.labelsize": 12,
#     "figure.dpi": 300,
#     "savefig.dpi": 300,
#     "savefig.format": "pdf",
#     "savefig.bbox": "tight",
    
# })

plt.rcParams.update({
    # Font settings for clarity and compatibility with academic publications
    "font.family": "serif",  # Consistent font family
    # "font.serif": ["Times", "Computer Modern Roman"],  # Preferred serif fonts
    "font.size": 14,  # Base font size for better readability
    "text.usetex": False,  # Enable LaTeX for text rendering for a professional look

    # Figure aesthetics & size for detailed visuals and fit on publication pages
    "figure.figsize": (15, 8),  # Adjusted figure size for a balance between detail and fit
    "figure.facecolor": "white",  # White figure background for clean print
    "figure.autolayout": True,  # Enable automatic layout adjustments
    "figure.dpi": 400,  # High resolution figures
    "savefig.dpi": 400,  # High resolution saving settings
    "savefig.format": "pdf",  # Save figures in PDF format for publications
    "savefig.bbox": "tight",  # Tight bounding box around figures

    # Axes aesthetics for clarity and precision
    "axes.labelsize": 14,  # Clear labeling with larger font size
    "axes.titlesize": 20,  # Prominent titles for immediate recognition
    "axes.facecolor": "white",  # White axes background

    # Legend aesthetics for distinguishing plot elements
    "legend.fontsize": 12,  # Readable legend font size
    "legend.frameon": False,  # No frame around legend for cleaner look
    "legend.loc": "best",  # Optimal legend positioning

    # Line aesthetics for clear visual distinctions
    "lines.linewidth": 2,  # Thicker lines for visibility
    "lines.markersize": 8,  # Slightly smaller markers for balance

    # Tick label sizes for readability
    "xtick.labelsize": 12, 
    "ytick.labelsize": 12,
    "xtick.direction": "in",  # Ticks inside the plot
    "ytick.direction": "in",  # Ticks inside the plot
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
    plt.savefig(f"../../reports/figures/{column}_over_time_by_nhs_region.pdf")
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
    plt.savefig(f"../../reports/figures/{column}_over_time_by_nhs_region.pdf")
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
plt.savefig(f"../../reports/figures/covid-19_icu_hospitalisation_heatmap.pdf")
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


# plot england data with the lockdown periods
england_data = 