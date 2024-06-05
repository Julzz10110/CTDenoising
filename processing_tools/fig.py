from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def plot_imgs(width=None, vmin=None, vmax=None, **kwargs):
    if width is None:
        width = plt.rcParams["figure.figsize"][0]

    rows = 1
    cols = len(kwargs)

    fig = plt.figure(figsize=(width, rows / cols * width))

    grid = ImageGrid(
        fig,
        111,                
        nrows_ncols=(rows, cols),
        axes_pad=0.025,       
        label_mode="all",   
    )

    for ax, (label, img) in zip(grid, kwargs.items()):
        ax.axis('off')
        for s in ax.spines.values():
            s.set_visible(False)

            ax.set_title(label)
            ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)


def add_zoom_bubble(axes_image,
                    inset_center=(0.75, 0.75),
                    inset_radius=0.2,
                    roi=(0.2, 0.2),
                    zoom=2,
                    edgecolor="red",
                    linewidth=3,
                    alpha=1.0,
                    **kwargs):

    ax = axes_image.axes
    data = axes_image.get_array()
    roi_radius = inset_radius / zoom

    opts = dict(facecolor="none")
    opts["edgecolor"] = edgecolor
    opts["linewidth"] = linewidth
    opts["alpha"] = alpha

  
    axins = ax.inset_axes(
        [
            inset_center[0] - inset_radius,
            inset_center[1] - inset_radius,
            2 * inset_radius,
            2 * inset_radius
        ],
        transform=ax.transAxes,
    )
    axins.axis('off')

    im_inset = axins.imshow(
        data,
        cmap=axes_image.get_cmap(),
        norm=axes_image.norm,
        aspect="auto",            
        interpolation="nearest",
        alpha=1.0,
        vmin=axes_image.get_clim()[0],
        vmax=axes_image.get_clim()[1],
        origin=axes_image.origin,
        extent=axes_image.get_extent(),
        filternorm=axes_image.get_filternorm(),
        filterrad=axes_image.get_filterrad(),
        resample=None,          
        url=None,               
        data=None,              
    )

    axis_to_data = ax.transAxes + ax.transData.inverted()
    lower_left = axis_to_data.transform(np.array(roi) - roi_radius)
    top_right = axis_to_data.transform(np.array(roi) + roi_radius)
    axins.set_xlim(lower_left[0], top_right[0])
    axins.set_ylim(lower_left[1], top_right[1])


    patch = patches.Circle(
        inset_center,
        radius=inset_radius,
        transform=ax.transAxes,
        zorder=axins.get_zorder() + 1,              
        **opts,
    )
    im_inset.set_clip_path(patch)
    ax.add_patch(patch)


    ax.add_patch(
        patches.Circle(
            roi,
            radius=roi_radius,
            transform=ax.transAxes,
            **opts,
        )
    )


    inset_center = np.array(inset_center)
    roi_center = np.array(roi)
    v = inset_center - roi_center
    d = np.linalg.norm(v)

    ax.add_patch(
        patches.ConnectionPatch(
            roi_center + roi_radius / d * v,
            roi_center + (d - inset_radius) / d * v,
            'axes fraction', 'axes fraction',
            axesA=ax, axesB=ax, arrowstyle="-",
            **opts
        )
    )
