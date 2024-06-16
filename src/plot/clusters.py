import polars as pl

import contextily as ctx
import plotly.express as px
import branca.colormap as cm
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "ieee"])


def plot_city(
    service_time_df, h3_df, city, xaxis_title="Service Time [s]", max_service_time=600
):
    # turn the cmap into a list
    city_df = service_time_df.filter(pl.col("city") == city)

    ordering = (
        city_df.group_by("cluster")
        .agg(pl.col("service_time").mean())
        .sort("service_time", descending=True)
    )
    ordering = ordering["cluster"].to_list()

    values, _ = px.colors.convert_colors_to_same_type(
        px.colors.diverging.Tealrose_r, colortype="tuple"
    )
    cmap = cm.LinearColormap(values, vmin=0, vmax=4, max_labels=4)
    cmapper = {cluster: cmap(i) for i, cluster in enumerate(ordering)}

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    plot_df = h3_df.query(f'city == "{city}"').to_crs(epsg=3857)
    plot_df.plot(
        alpha=0.7,
        legend=True,
        color=plot_df.apply(lambda x: cmapper[x["cluster"]], axis=1),
        ax=ax[0],
    )
    ctx.add_basemap(ax[0], source=ctx.providers.CartoDB.Positron)

    ax[0].axes.set_axis_off()

    # plot the box plot
    service_time_arrays = []
    labels = []
    colors = []

    for i, cluster in enumerate(ordering[::-1]):
        service_time_array = city_df.filter(pl.col("cluster") == cluster)[
            "service_time"
        ].to_list()

        label = len(ordering) - i

        service_time_arrays.append(service_time_array)
        labels.append(label)
        colors.append(cmapper[cluster])

    box = ax[1].boxplot(
        service_time_arrays,
        patch_artist=True,
        labels=labels,
        notch=True,
        # dont show outliers
        showfliers=False,
        meanline=True,
        showmeans=True,
        meanprops=dict(color="black", linestyle="--"),
        medianprops=dict(color="black", linestyle="-"),
    )

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)

    ax[1].set_xlabel("Cluster")
    # turn the ticks off for the x-axis
    ax[1].tick_params(axis="x", which="both", bottom=False, top=False)

    ax[1].set_ylabel("Service Time (s)")

    # set y-axis limits to 0 and max_service_time
    ax[1].set_ylim(0, max_service_time)

    # plot the histogram
    # for i, cluster in enumerate(ordering[::-1]):
    #     service_time_array = city_df.filter(pl.col("cluster") == cluster)[
    #         "service_time"
    #     ]

    #     label = (
    #         f"{len(ordering) - i }, $\mu=$"
    #         + f"{int(service_time_array.mean())}s, "
    #         + "$P_{50\%}=$"
    #         + f"{int(service_time_array.median())}s, N={len(service_time_array)}"
    #     )

    #     ax[1].hist(
    #         city_df.filter(pl.col("cluster") == cluster)["service_time"],
    #         alpha=0.7,
    #         # label=cluster,
    #         color=cmapper[cluster],
    #         # normalize the histogram
    #         density=True,
    #         # fix the range of the histogram
    #         range=(0, max_service_time),
    #         # fix the bin size
    #         bins=50,
    #         # add the label
    #         label=label,
    #     )

    #     # plot kde
    #     city_df.filter(pl.col("cluster") == cluster)["service_time"].to_pandas().plot(
    #         kind="kde",
    #         label=cluster,
    #         color=cmapper[cluster],
    #         # remove from the legend
    #         legend=False,
    #     )

    # for i, cluster in enumerate(ordering[::-1]):
    #     service_time_array = city_df.filter(pl.col("cluster") == cluster)[
    #         "service_time"
    #     ]

    #     label = (
    #         f"{len(ordering) - i }, $\mu=$"
    #         + f"{int(service_time_array.mean())}s, "
    #         + "$P_{50\%}=$"
    #         + f"{int(service_time_array.median())}s, N={len(service_time_array)}"
    #     )

    #     ax[1].boxplot(
    #         service_time_array,
    #         patch_artist=True,  # fill with color
    #         labels=[label],  # labels
    #         boxprops=dict(facecolor=cmapper[cluster], color=cmapper[cluster]),
    #     )

    # # reverse the order of the legend
    # handles, labels = ax[1].get_legend_handles_labels()
    # # take every other, starting at 1
    # handles = handles[::2]
    # labels = labels[::2]
    # ax[1].legend(handles[::-1], labels[::-1])

    # ax[1].set_xlabel(xaxis_title)
    # ax[1].set_ylabel("Probability")
    # ax[1].set_xlim(0, max_service_time)

    return fig, ax


def plot_cities(
    cities, service_time_df, h3_df, xaxis_title="Service Time [s]", max_service_time=600
):
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))  # Create a 2x2 subplot grid

    for idx, city in enumerate(cities):
        # Calculate row and column index for the current subplot
        row = idx
        # col = idx % 2

        city_df = service_time_df.filter(pl.col("city") == city)
        ordering = (
            city_df.group_by("cluster")
            .agg(pl.col("service_time").mean())
            .sort("service_time", descending=True)
        )
        ordering = ordering["cluster"].to_list()

        values, _ = px.colors.convert_colors_to_same_type(
            px.colors.diverging.Tealrose_r, colortype="tuple"
        )
        cmap = cm.LinearColormap(values, vmin=0, vmax=4, max_labels=4)
        cmapper = {cluster: cmap(i) for i, cluster in enumerate(ordering)}

        # Plot map
        ax_map = axes[row][0]  # Corrected indexing here
        plot_df = h3_df.query(f'city == "{city}"').to_crs(epsg=3857)
        plot_df.plot(
            alpha=0.7,
            legend=True,
            color=plot_df.apply(lambda x: cmapper[x["cluster"]], axis=1),
            ax=ax_map,
        )
        ctx.add_basemap(ax_map, source=ctx.providers.CartoDB.Positron)
        ax_map.set_title(city)  # Add a title to each map
        ax_map.axes.set_axis_off()

        # Plot histogram
        ax_hist = axes[row][1]  # Corrected indexing here

        # plot the box plot
        service_time_arrays = []
        labels = []
        colors = []

        for i, cluster in enumerate(ordering[::-1]):
            service_time_array = city_df.filter(pl.col("cluster") == cluster)[
                "service_time"
            ].to_list()

            label = len(ordering) - i

            service_time_arrays.append(service_time_array)
            labels.append(label)
            colors.append(cmapper[cluster])

        box = ax_hist.boxplot(
            service_time_arrays,
            patch_artist=True,
            labels=labels,
            notch=True,
            # dont show outliers
            showfliers=False,
            meanline=True,
            showmeans=True,
            meanprops=dict(color="black", linestyle="--"),
            medianprops=dict(color="black", linestyle="-"),
        )

        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)

        ax_hist.set_xlabel("Cluster")
        # turn the ticks off for the x-axis
        ax_hist.tick_params(axis="x", which="both", bottom=False, top=False)

        ax_hist.set_ylabel("Service Time (s)")

        # set y-axis limits to 0 and max_service_time
        ax_hist.set_ylim(0, max_service_time)


        # for i, cluster in enumerate(ordering[::-1]):
        #     service_time_array = city_df.filter(pl.col("cluster") == cluster)[
        #         "service_time"
        #     ]

        #     label = (
        #         f"{len(ordering) - i }, $\mu=$"
        #         + f"{int(service_time_array.mean())}s, "
        #         + "$P_{50\%}=$"
        #         + f"{int(service_time_array.median())}s, N={len(service_time_array)}"
        #     )

        #     ax_hist.hist(
        #         service_time_array,
        #         alpha=0.7,
        #         color=cmapper[cluster],
        #         density=True,
        #         range=(0, max_service_time),
        #         bins=50,
        #         label=label,
        #     )

        #     # KDE plot is omitted for brevity, but can be included if needed
        #     city_df.filter(pl.col("cluster") == cluster)[
        #         "service_time"
        #     ].to_pandas().plot(
        #         kind="kde",
        #         # label=cluster,
        #         color=cmapper[cluster],
        #         # remove from the legend
        #         legend=None,
        #         ax=ax_hist,
        #         label="_nolegend_",
        #     )
        # for i, cluster in enumerate(ordering[::-1]):
        #     service_time_array = city_df.filter(pl.col("cluster") == cluster)[
        #         "service_time"
        #     ]

        #     label = (
        #         f"{len(ordering) - i }, $\mu=$"
        #         + f"{int(service_time_array.mean())}s, "
        #         + "$P_{50\%}=$"
        #         + f"{int(service_time_array.median())}s, N={len(service_time_array)}"
        #     )

        #     ax_hist.boxplot(
        #         service_time_array,
        #         patch_artist=True,  # fill with color
        #         labels=[label],  # labels
        #         boxprops=dict(facecolor=cmapper[cluster], color=cmapper[cluster]),
        #     )

        # ax_hist.set_xlabel(xaxis_title)
        # ax_hist.set_ylabel("Probability")
        # ax_hist.set_xlim(0, max_service_time)
        # ax_hist.set_ylim(0, 0.01)

        # ax_hist.legend()
        # # make the legend a bit smaller
        # ax_hist.legend(fontsize=8)

    # plt.tight_layout()
    return fig, axes
