import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import PchipInterpolator


def plot_timeseries_simple(ages, metric, metric_name, title, color):
    # Convert to numpy arrays
    ages_array = np.array(ages)
    metric_array = np.array(metric)

    # Create a mask for finite values (removes NaNs and Infs)
    valid_mask = np.isfinite(ages_array) & np.isfinite(metric_array)

    # Apply mask
    clean_ages = ages_array[valid_mask]
    clean_metric = metric_array[valid_mask]

    # Check if we have enough data points to plot (need at least 2 for a line)
    if len(clean_ages) < 2:
        print(f"Warning: Not enough valid data points for {title}. Skipping.")
        return

    # Interpolation on clean data
    x_smooth = np.linspace(clean_ages.min(), clean_ages.max(), 300)
    pchip = PchipInterpolator(clean_ages, clean_metric)
    y_smooth = pchip(x_smooth)

    plt.figure(figsize=(10, 6))
    plt.scatter(clean_ages, clean_metric, marker='o', linestyle='-', color=color)
    plt.plot(x_smooth, y_smooth, color=color, linewidth=2)

    plt.xlabel('Age (Ma)')
    plt.ylabel(metric_name)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.gca().invert_xaxis()

    os.makedirs('output', exist_ok=True)
    filename = title.replace(' ', '_') + '.png'
    plt.savefig(os.path.join('output', filename), dpi=300)
    plt.show()


def plot_timeseries_double(ages, metric1, metric1_name, metric2, metric2_name, title):
    ages_array = np.array(ages)
    m1_array = np.array(metric1)
    m2_array = np.array(metric2)

    # Create a mask where BOTH metrics AND age are finite
    # This ensures we only plot points where we have complete data pairs
    valid_mask = np.isfinite(ages_array) & np.isfinite(m1_array) & np.isfinite(m2_array)

    # Apply mask
    clean_ages = ages_array[valid_mask]
    clean_m1 = m1_array[valid_mask]
    clean_m2 = m2_array[valid_mask]

    if len(clean_ages) < 2:
        print(f"Warning: Not enough valid data points for {title}. Skipping.")
        return

    # Interpolation on clean data
    x_smooth = np.linspace(clean_ages.min(), clean_ages.max(), 300)
    pchip1 = PchipInterpolator(clean_ages, clean_m1)
    pchip2 = PchipInterpolator(clean_ages, clean_m2)

    y_smooth1 = pchip1(x_smooth)
    y_smooth2 = pchip2(x_smooth)
    color1 = '#6D92C7'
    color2 = '#C94A65'
    plt.figure(figsize=(10, 6))
    plt.scatter(clean_ages, clean_m1, marker='o', linestyle='-', color=color1)
    plt.plot(x_smooth, y_smooth1, color=color1, linewidth=2,label=metric1_name)
    plt.scatter(clean_ages, clean_m2, marker='o', linestyle='-', color=color2)
    plt.plot(x_smooth, y_smooth2, color=color2, linewidth=2,label=metric2_name)
    plt.xlabel('Age (Ma)')
    plt.ylabel(title)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.gca().invert_xaxis()
    os.makedirs('output', exist_ok=True)
    filename = title.replace(' ', '_') + '.png'
    plt.savefig(os.path.join('output', filename), dpi=300)
    plt.show()


# Main Execution
try:
    df = pd.read_csv(r"output\geo_indicators.csv")

    # Optional: Inspect where the NaNs are before plotting
    # print(df[df["BQART_TSS_est"].isna()])

    plot_timeseries_simple(
        df["Age"],
        df["Sea_Level_dSL"],
        "Sea-level compared to present-day [m]",
        "Sea-level vs Age",
        "steelblue"
    )

    plot_timeseries_double(
        df["Age"],
        df["BQART_TSS_est"],
        "Annual sediment flux - BQART",
        df["ROBART_TSS_est"],
        "Annual sediment flux - RoBART (MT/yr)",
        "Annual Sediment Flux vs Age"
    )

    plot_timeseries_simple(
        df["Age"],
        df["CO2_ppm"],
        "Atmospheric CO2 (ppm)",
        "CO2 vs Age",
        "indianred"
    )

except Exception as e:
    print(f"An error occurred: {e}")