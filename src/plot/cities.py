import matplotlib.pyplot as plt
import numpy as np

def plot_city_feature_density(
    city_gdf,
    value="value",
    feat_name="Feature",
    breakpoint=None,
    clusters=None,
    cap_percentile=None,
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.get_figure()

    # Remove any existing legends from the axis
    if ax.legend_:
        ax.legend_.remove()

    # Check if clusters column is provided
    if clusters:
        unique_clusters = city_gdf[clusters].unique()
        cmap = plt.cm.get_cmap("tab20", len(unique_clusters))
        city_gdf.plot(
            column=clusters,
            cmap=cmap,
            linewidth=0.8,
            ax=ax,
            edgecolor="0.8",
            legend=False,  # Set legend to False
            categorical=True,
        )
    else:
        # If cap_percentile is provided, adjust the data
        if cap_percentile:
            cap_value = city_gdf[value].quantile(cap_percentile)
            city_gdf[value] = np.where(
                city_gdf[value] > cap_value, cap_value, city_gdf[value]
            )

        max_value = city_gdf[value].max()

        if breakpoint is not None:
            cmap = colors.LinearSegmentedColormap.from_list(
                "custom",
                [
                    (0, "blue"),
                    (breakpoint / max_value, "blue"),
                    (breakpoint / max_value, "red"),
                    (1, "red"),
                ],
                N=256,
            )
            vmin = 0
            vmax = max_value
        else:
            cmap = plt.cm.get_cmap("GnBu")
            vmin = None
            vmax = None

        city_gdf.plot(
            column=value,
            cmap=cmap,
            linewidth=0.8,
            ax=ax,
            edgecolor="0.8",
            legend=False,  # Set legend to False
            vmin=vmin,
            vmax=vmax,
        )

    ax.set_title(
        feat_name,
        fontdict={"fontsize": "20", "fontweight": "3"},
    )

    # Removing grid and axis
    ax.axis("off")
    ax.grid(False)

    return fig, ax


# Example usage:
default_args = {
    "A": {"clusters": "cluster", "feat_name": "Clusters"},
    "B": {"value": "population", "feat_name": "Population", "cap_percentile": 0.95},
    "C": {
        "value": "Commerce and Industry",
        "feat_name": "Commerce and Industry",
        "cap_percentile": 0.95,
    },
    "D": {
        "value": "Transportation",
        "feat_name": "Transportation",
        "cap_percentile": 0.95,
    },
    "E": {
        "value": "Historical and Cultural",
        "feat_name": "Historical and Cultural",
        "cap_percentile": 0.95,
    },
    "F": {
        "value": "Natural Elements",
        "feat_name": "Natural Elements",
        "cap_percentile": 0.95,
    },  # You can leave this empty or assign other default values.
}


def multi_plot_city_feature_density(city_gdf, args_dict=default_args):
    fig = plt.figure(figsize=(14, 14))
    fig.suptitle("City Feature Densities", fontsize=16)

    # Define axes with adjusted spacings
    axA = fig.add_axes([0, 0.33, 0.66, 0.66])
    axB = fig.add_axes(
        [0.67, 0.67, 0.32, 0.32], sharex=axA, sharey=axA
    )  # Increased width
    axC = fig.add_axes([0.67, 0.33, 0.32, 0.32], sharex=axA)  # Increased width
    axD = fig.add_axes(
        [0, 0.01, 0.32, 0.32], sharey=axA
    )  # Increased height & Reduced bottom space
    axE = fig.add_axes(
        [0.33, 0.01, 0.32, 0.32], sharey=axA
    )  # Increased height & Reduced bottom space
    axF = fig.add_axes(
        [0.67, 0.01, 0.32, 0.32], sharex=axA, sharey=axA
    )  # Increased height & Reduced bottom space

    axes = {"A": axA, "B": axB, "C": axC, "D": axD, "E": axE, "F": axF}

    for key, ax in axes.items():
        if key in args_dict:
            args = args_dict[key]
            _, _ = plot_city_feature_density(
                city_gdf=city_gdf,
                value=args.get("value", "value"),
                feat_name=args.get("feat_name", "Feature"),
                breakpoint=args.get("breakpoint"),
                clusters=args.get("clusters"),
                cap_percentile=args.get("cap_percentile"),
                ax=ax,
            )
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)  # add a grid

    plt.tight_layout()
    return fig