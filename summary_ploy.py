import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as pl
from shap.plots import colors
from shap.plots._labels import labels

pl.rcParams['font.sans-serif'] = ['Times New Roman']



def beeswarm(shap_values, features=None, feature_names=None, max_display=None, plot_type=None,
             axis_color="#333333", alpha=1, show=True, color_bar_label=labels["FEATURE_VALUE"],
             cmap=colors.red_blue):

    max_display = max_display
    num_features = shap_values.shape[1]
    feature_order = np.flip(np.arange(min(max_display, num_features)), 0)

    row_height = 0.4

    pl.gcf().set_size_inches(8, len(feature_order) * row_height + 1.5)
    pl.axvline(x=0, color="#999999", zorder=-1)

    for pos, i in enumerate(feature_order):
        pl.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
        shaps = shap_values[:, i]
        values = features[:, i]
        inds = np.arange(len(shaps))
        np.random.shuffle(inds)
        values = values[inds]
        shaps = shaps[inds]
        colored_feature = True
        values = np.array(values, dtype=np.float64)  # make sure this can be numeric
        N = len(shaps)

        nbins = 100
        quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
        inds = np.argsort(quant + np.random.randn(N) * 1e-6)
        layer = 0
        last_bin = -1
        ys = np.zeros(N)
        for ind in inds:
            if quant[ind] != last_bin:
                layer = 0
            ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
            layer += 1
            last_bin = quant[ind]
        ys *= 0.9 * (row_height / np.max(ys + 1))

        if features is not None and colored_feature:
            # trim the color range, but prevent the color range from collapsing
            vmin = np.nanpercentile(values, 5)
            vmax = np.nanpercentile(values, 95)
            if vmin == vmax:
                vmin = np.nanpercentile(values, 1)
                vmax = np.nanpercentile(values, 99)
                if vmin == vmax:
                    vmin = np.min(values)
                    vmax = np.max(values)
            if vmin > vmax:  # fixes rare numerical precision issues
                vmin = vmax

            assert features.shape[0] == len(shaps), "Feature and SHAP matrices must have the same number of rows!"

            # plot the nan values in the interaction feature as grey
            nan_mask = np.isnan(values)
            pl.scatter(shaps[nan_mask], pos + ys[nan_mask], color="#777777",
                       s=16, alpha=alpha, linewidth=0,
                       zorder=3, rasterized=len(shaps) > 500)

            # plot the non-nan values colored by the trimmed feature value
            cvals = values[np.invert(nan_mask)].astype(np.float64)
            cvals_imp = cvals.copy()
            cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
            cvals[cvals_imp > vmax] = vmax
            cvals[cvals_imp < vmin] = vmin
            pl.scatter(shaps[np.invert(nan_mask)], pos + ys[np.invert(nan_mask)],
                       cmap=cmap, vmin=vmin, vmax=vmax, s=16,
                       c=cvals, alpha=alpha, linewidth=0,
                       zorder=3, rasterized=len(shaps) > 500)
    m = cm.ScalarMappable(cmap=cmap)
    m.set_array([0, 1])
    cb = pl.colorbar(m, ax=pl.gca(), ticks=[0, 1], aspect=80)
    cb.set_ticklabels([labels['FEATURE_VALUE_LOW'], labels['FEATURE_VALUE_HIGH']])
    cb.set_label(color_bar_label, size=12, labelpad=0)
    cb.ax.tick_params(labelsize=11, length=0)
    cb.set_alpha(1)
    cb.outline.set_visible(False)

    pl.gca().xaxis.set_ticks_position('bottom')
    pl.gca().yaxis.set_ticks_position('none')
    pl.gca().spines['right'].set_visible(False)
    pl.gca().spines['top'].set_visible(False)
    pl.gca().spines['left'].set_visible(False)
    pl.gca().tick_params(color=axis_color, labelcolor=axis_color)
    pl.yticks(range(len(feature_order)), [feature_names[i] for i in feature_order], fontsize=13)
    if plot_type != "bar":
        pl.gca().tick_params('y', length=20, width=0.5, which='major')
    pl.gca().tick_params('x', labelsize=11)
    pl.ylim(-1, len(feature_order))
    if plot_type == "bar":
        pl.xlabel(labels['GLOBAL_VALUE'], fontsize=13)
    else:
        pl.xlabel(labels['VALUE'], fontsize=13)
    pl.tight_layout()
    if show:
        pl.show()


def beeswarm_alone(all_shap_values, features=None, feature_names=None, shap_values_abs=None, max_display=20,
                   plot_type=None, axis_color="#333333", alpha=1, color_bar_label=labels["FEATURE_VALUE"],
                   cmap=colors.red_blue):
    num_features = all_shap_values.shape[-1]
    feature_order = np.flip(np.arange(min(max_display, num_features)), 0)
    print(feature_order)

    # pl.gcf().set_size_inches(8, 0.4 + 1.5)
    # pl.axvline(x=0, color="#999999", zorder=-1)
    for j in range(5):
        shap_values = all_shap_values[j, :]
        print(shap_values.shape)
        for i in feature_order:
            # pl.axhline(y=0, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
            shaps = shap_values[:, i]
            values = features[:, i]
            inds = np.arange(len(shaps))
            np.random.shuffle(inds)
            values = values[inds]
            shaps = shaps[inds]
            colored_feature = True
            values = np.array(values, dtype=np.float64)  # make sure this can be numeric
            N = len(shaps)

            nbins = 100
            quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
            inds = np.argsort(quant + np.random.randn(N) * 1e-6)
            layer = 0
            last_bin = -1
            ys = np.zeros(N)
            for ind in inds:
                if quant[ind] != last_bin:
                    layer = 0
                ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
                layer += 1
                last_bin = quant[ind]
            ys *= 0.9 * (0.4 / np.max(ys + 1))
            vmin = np.nanpercentile(values, 5)
            vmax = np.nanpercentile(values, 95)
            # trim the color range, but prevent the color range from collapsing
            assert features.shape[0] == len(shaps), "Feature and SHAP matrices must have the same number of rows!"

            # plot the nan values in the interaction feature as grey
            nan_mask = np.isnan(values)
            pl.scatter(shaps[nan_mask], 0 + ys[nan_mask], color="#777777",
                       s=16, alpha=alpha, linewidth=0,
                       zorder=3, rasterized=len(shaps) > 500)

            # plot the non-nan values colored by the trimmed feature value
            cvals = values[np.invert(nan_mask)].astype(np.float64)
            cvals_imp = cvals.copy()
            cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
            cvals[cvals_imp > vmax] = vmax
            cvals[cvals_imp < vmin] = vmin
            pl.scatter(shaps[np.invert(nan_mask)], 0 + ys[np.invert(nan_mask)],
                       cmap=cmap, vmin=vmin, vmax=vmax, s=3,
                       c=cvals, alpha=alpha, linewidth=0,
                       zorder=3, rasterized=len(shaps) > 500)
            pl.setp(pl.gca().get_yticklabels(), visible=False)
            pl.setp(pl.gca().get_xticklabels(), visible=False)
            print(i)
            pl.xlim(-0.6, 0.6)
            pl.tick_params(axis='both', which='both', length=0)
            pl.gca().spines['right'].set_visible(False)
            pl.gca().spines['top'].set_visible(False)
            pl.gca().spines['left'].set_visible(False)
            pl.gca().spines['bottom'].set_visible(False)
            pl.savefig(f"shap{j}{i}.png", bbox_inches='tight')
            plt.clf()
    pl.show()

def beeswarm_all(all_shap_values, features=None, feature_names=None, shap_values_abs=None, max_display=20,
                   plot_type=None, axis_color="#333333", alpha=1, color_bar_label=labels["FEATURE_VALUE"],
                   cmap=colors.red_blue):
    num_features = all_shap_values.shape[-1]
    feature_order = np.flip(np.arange(min(max_display, num_features)), 0)
    print(feature_order)

    # pl.gcf().set_size_inches(8, 0.4 + 1.5)
    # pl.axvline(x=0, color="#999999", zorder=-1)
    for j in range(5):
        shap_values = all_shap_values[j, :]
        print(shap_values.shape)
        j += 1
        for i in feature_order:
            # pl.axhline(y=0, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
            pl.subplot(15, 5, j)
            pl.gcf().set_size_inches(20, 20)
            pl.tight_layout(h_pad=0, w_pad=2)
            print(j)
            shaps = shap_values[:, i]
            values = features[:, i]
            inds = np.arange(len(shaps))
            np.random.shuffle(inds)
            values = values[inds]
            shaps = shaps[inds]
            colored_feature = True
            values = np.array(values, dtype=np.float64)  # make sure this can be numeric
            N = len(shaps)

            nbins = 100
            quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
            inds = np.argsort(quant + np.random.randn(N) * 1e-6)
            layer = 0
            last_bin = -1
            ys = np.zeros(N)
            for ind in inds:
                if quant[ind] != last_bin:
                    layer = 0
                ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
                layer += 1
                last_bin = quant[ind]
            ys *= 0.9 * (0.4 / np.max(ys + 1))
            vmin = np.nanpercentile(values, 5)
            vmax = np.nanpercentile(values, 95)
            # trim the color range, but prevent the color range from collapsing
            assert features.shape[0] == len(shaps), "Feature and SHAP matrices must have the same number of rows!"

            # plot the nan values in the interaction feature as grey
            nan_mask = np.isnan(values)
            pl.scatter(shaps[nan_mask], 0 + ys[nan_mask], color="#777777",
                       s=16, alpha=alpha, linewidth=0,
                       zorder=3, rasterized=len(shaps) > 500)

            # plot the non-nan values colored by the trimmed feature value
            cvals = values[np.invert(nan_mask)].astype(np.float64)
            cvals_imp = cvals.copy()
            cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
            cvals[cvals_imp > vmax] = vmax
            cvals[cvals_imp < vmin] = vmin
            pl.scatter(shaps[np.invert(nan_mask)], 0 + ys[np.invert(nan_mask)],
                       cmap=cmap, vmin=vmin, vmax=vmax, s=5,
                       c=cvals, alpha=alpha, linewidth=0,
                       zorder=3, rasterized=len(shaps) > 500)
            pl.setp(pl.gca().get_yticklabels(), visible=False)
            pl.setp(pl.gca().get_xticklabels(), visible=False)
            pl.xlim(-0.6, 0.6)
            pl.ylim(-1, 1)
            pl.tick_params(axis='both', which='both', length=0)
            print(j-1)
            if shap_values_abs:
                pl.text(-0.6, 0.60, shap_values_abs[j-1], fontsize=25)
            # pl.gca().spines['right'].set_visible(False)
            # pl.gca().spines['left'].set_visible(False)
            pl.gca().spines['bottom'].set_visible(False)
            pl.gca().spines['top'].set_visible(False)
            if j in [1, 2, 3, 4, 5]:
                pl.gca().spines['top'].set_visible(True)
            if j in [71, 72, 73, 74, 75]:
                pl.gca().spines['bottom'].set_visible(True)
            j += 5
    pl.show()