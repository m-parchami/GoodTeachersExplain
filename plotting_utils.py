from captum.attr import visualization as viz
def visualize_image_attr_custom(
        attr, original_image=None, method='heat_map', sign='absolute_value',
        plt_fig_axis=None, outlier_perc=2, cmap=None,
        alpha_overlay=0.5, show_colorbar=False, 
        title=None, fig_size=(6, 6), use_pyplot=True,
        scale_factor=None, color_original_image=False
):
    """
    fig, axes = plt.subplots(1, 1)
    ax = axes
    visualize_image_attr_custom(attr, sign="positive", show_colorbar=False, 
                        plt_fig_axis=(fig, ax), use_pyplot=False, scale_factor=np.percentile(attr, 99.5))
    plt.imshow()
    Modifies captum.attr.visualization.visualize_image_attr by using value based outlier clamping and by allowing the normalization factor to be specified by the caller. The latter helps in using a common normalization factor for a series of images, as in AggAtt.
    """
    # Create plot if figure, axis not provided
    if plt_fig_axis is not None:
        plt_fig, plt_axis = plt_fig_axis
    else:
        if use_pyplot:
            plt_fig, plt_axis = plt.subplots(figsize=fig_size)
        else:
            plt_fig = matplotlib.figure.Figure(figsize=fig_size)
            plt_axis = plt_fig.subplots()

    if original_image is not None:
        if np.max(original_image) <= 1.0:
            original_image = viz._prepare_image(original_image * 255)
    else:
        assert (
            viz.ImageVisualizationMethod[method] == viz.ImageVisualizationMethod.heat_map
        ), "Original Image must be provided for any visualization other than heatmap."

    # Remove ticks and tick labels from plot.
    plt_axis.xaxis.set_ticks_position("none")
    plt_axis.yaxis.set_ticks_position("none")
    plt_axis.set_yticklabels([])
    plt_axis.set_xticklabels([])
    plt_axis.grid(b=False)

    heat_map = None
    # Show original image
    if viz.ImageVisualizationMethod[method] == viz.ImageVisualizationMethod.original_image:
        if len(original_image.shape) > 2 and original_image.shape[2] == 1:
            original_image = np.squeeze(original_image, axis=2)
        plt_axis.imshow(original_image)
    else:
        # Choose appropriate signed attributions and normalize.
        if scale_factor != 0:
            norm_attr = viz._normalize_scale(attr, scale_factor)
        else:
            norm_attr = attr.copy()

        # Set default colormap and bounds based on sign.
        if viz.VisualizeSign[sign] == viz.VisualizeSign.all:
            default_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                "RdWhGn", ["red", "white", "green"]
            )
            vmin, vmax = -1, 1
        elif viz.VisualizeSign[sign] == viz.VisualizeSign.positive:
            default_cmap = "Greens"
            vmin, vmax = 0, 1
        elif viz.VisualizeSign[sign] == viz.VisualizeSign.negative:
            default_cmap = "Reds"
            vmin, vmax = 0, 1
        elif viz.VisualizeSign[sign] == viz.VisualizeSign.absolute_value:
            default_cmap = "Blues"
            vmin, vmax = 0, 1
        else:
            raise AssertionError("Visualize Sign type is not valid.")
        cmap = cmap if cmap is not None else default_cmap

        # Show appropriate image visualization.
        if viz.ImageVisualizationMethod[method] == viz.ImageVisualizationMethod.heat_map:
            heat_map = plt_axis.imshow(
                norm_attr, cmap=cmap, vmin=vmin, vmax=vmax)
        elif (
            viz.ImageVisualizationMethod[method]
            == viz.ImageVisualizationMethod.blended_heat_map
        ):
            if color_original_image:
                plt_axis.imshow(original_image)
            else:
                plt_axis.imshow(np.mean(original_image, axis=2), cmap="gray")
            heat_map = plt_axis.imshow(
                norm_attr, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha_overlay
            )
        elif viz.ImageVisualizationMethod[method] == viz.ImageVisualizationMethod.masked_image:
            assert viz.VisualizeSign[sign] != viz.VisualizeSign.all, (
                "Cannot display masked image with both positive and negative "
                "attributions, choose a different sign option."
            )
            plt_axis.imshow(
                viz._prepare_image(
                    original_image * np.expand_dims(norm_attr, 2))
            )
        elif viz.ImageVisualizationMethod[method] == viz.ImageVisualizationMethod.alpha_scaling:
            assert viz.VisualizeSign[sign] != viz.VisualizeSign.all, (
                "Cannot display alpha scaling with both positive and negative "
                "attributions, choose a different sign option."
            )
            plt_axis.imshow(
                np.concatenate(
                    [
                        original_image,
                        viz._prepare_image(
                            np.expand_dims(norm_attr, 2) * 255),
                    ],
                    axis=2,
                )
            )
        else:
            raise AssertionError("Visualize Method type is not valid.")

    # Add colorbar. If given method is not a heatmap and no colormap is relevant,
    # then a colormap axis is created and hidden. This is necessary for appropriate
    # alignment when visualizing multiple plots, some with heatmaps and some
    # without.
    if show_colorbar:
        axis_separator = make_axes_locatable(plt_axis)
        colorbar_axis = axis_separator.append_axes(
            "bottom", size="5%", pad=0.1)
        if heat_map:
            plt_fig.colorbar(
                heat_map, orientation="horizontal", cax=colorbar_axis)
        else:
            colorbar_axis.axis("off")
    if title:
        plt_axis.set_title(title)

    if use_pyplot:
        plt.show()

    return plt_fig, plt_axis

