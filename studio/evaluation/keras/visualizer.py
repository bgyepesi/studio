import os
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import matplotlib

from sklearn import metrics
from studio.evaluation.keras import utils
from plotly.offline import iplot
from inspect import signature

dir_fonts = os.path.join(os.path.dirname(__file__), 'fonts')
BOLD_FONT_PROP = matplotlib.font_manager.FontProperties(fname=os.path.join(dir_fonts, "NeuzeitGroBold.ttf"))
NORMAL_FONT_PROP = matplotlib.font_manager.FontProperties(fname=os.path.join(dir_fonts, "NeuzeitGro.ttf"))


def get_aip_color_map():
    cvals = [0, 0.3, 0.6, 1]
    colors = [np.asarray([77 / 255, 0 / 255, 174 / 255]),
              np.asarray([202 / 255, 100 / 255, 223 / 255]),
              np.asarray([18 / 255, 125 / 255, 239 / 255]),
              np.asarray([54 / 255, 189 / 255, 100 / 255])]
    norm = plt.Normalize(min(cvals), max(cvals))
    tuples = list(zip(map(norm, cvals), colors))
    return matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)


def plot_confusion_matrix(cm, concepts=None, normalize=False,
                          show_counts=True,
                          show_labels=True,
                          show_arrow_bars=True,
                          figsize=(12.8, 7.2),
                          label_fontsize=26, ylabel_pad=120, xlabel_pad=40,
                          tick_fontsize=22,
                          cmap='aip', save_path=None,
                          xrotation=0, yrotation=90,
                          line_width=12, line_spacing_color=np.asarray([229 / 255, 225 / 255, 220 / 255]),
                          cb_labelsize=16,
                          cm_font_color='white', cm_font_size=42,
                          ):
    """Plot confusion matrix.

    Args:
        cm: N x N numpy array representing a confusion matrix.
        concepts: If `None`, does not display labels on the axes.
            If not `None`, must be a list of N names following the order within `cm`.
        normalize: If True, display the CM elements between 0 and 1, rather than the actual values
        show_text: If True, display cell values as text. Otherwise only display cell colors.
        show_arrow_bars: If True, display the vertical and horizontal arrow bars.
        figsize: Tuple indicating the size of the figure.
        label_fontsize: Font size of the x and y labels.
        ylabel_pad: Integer indicating padding of the y-axis text label.
        xlabel_pad: Integer indicating padding of the x-axis text label.
        tick_fontsize: Integer indicating the tick font size.
        cmap: Color choice. By default uses the AIP colormap
        save_path: If `save_path` specified, save confusion matrix in that location
        xrotation: Integer indicating the amount to rotate the x axis labels.
        yrotation: Integer indicating the amount to rotate the y axis labels.
        line_width: The width of the grid lines to add between confusion matrix elements. Set to 0 to remove grid.
        line_spacing_color: The color of the line spacing between cells in the matrix.
        cb_labelsize: The size of the labels for the color bar.
        cm_font_color: Font color of the text within the confusion matrix.
        cm_font_size: Font size of the text within the confusion matrix.

    Returns: Nothing. Plots (and optionally saves) the confusion matrix
    """

    if cmap == 'aip':
        cmap = get_aip_color_map()

    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError('Invalid confusion matrix shape, it should be N x N.')

    # Normalize rows (indicating the true label for a specific class) between 0 and 1.
    # This is for the background color and the colorbar.
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if normalize:
        cm = cm_normalized

    fig = plt.figure(figsize=figsize)
    # fig.patch.set_facecolor(line_spacing_color)

    ax = fig.add_subplot(111)
    # Background color of the confusion matrix.
    cax = ax.matshow(cm_normalized, vmin=np.min(cm_normalized), vmax=np.max(cm_normalized), cmap=cmap)

    # Add colorbar with less padding and a fine-grained interpolation.
    cb = plt.colorbar(cax,
                      fraction=0.046, pad=0.04, boundaries=np.linspace(0, 1, 1000),
                      format='%.1f',  # One sig digit.
                      ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],  # Show these numbers on the colorbar.
                      )
    # Set colorbar text font style.
    for label in cb.ax.get_yticklabels():
        label.set_fontproperties(NORMAL_FONT_PROP)

    # Colorbar font size.
    cb.ax.tick_params(labelsize=cb_labelsize)
    # Remove colorbar border.
    cb.outline.set_visible(False)

    # Force the ticks to be on the bottom.
    ax.xaxis.tick_bottom()
    # Remove small ticks on the borders.
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    plt.ylabel('True Label', fontproperties=BOLD_FONT_PROP,
               fontsize=label_fontsize, rotation=0, labelpad=ylabel_pad, y=0.48)
    plt.xlabel('Predicted Label', fontproperties=BOLD_FONT_PROP,
               fontsize=label_fontsize, labelpad=xlabel_pad)

    if concepts is not None and show_labels:
        if cm.shape[0] != len(concepts) or cm.shape[1] != len(concepts):
            raise ValueError('Number of concepts (%i) and dimensions of confusion matrix do not coincide (%i, %i)' %
                             (len(concepts), cm.shape[0], cm.shape[1]))

        n_labels = len(concepts)
        label_indexes = np.arange(0, n_labels, 1.0)
        ax.set_xticklabels(concepts, fontproperties=NORMAL_FONT_PROP, color='darkgrey')
        ax.set_yticklabels(concepts, fontproperties=NORMAL_FONT_PROP, color='darkgrey')
        plt.xticks(label_indexes, rotation=xrotation)
        plt.yticks(label_indexes, rotation=yrotation, va='center')

        # Add padding between squares (only apply if less than five classes).
        if line_width > 0:  # len(concepts) < 5:
            ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
            ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
            ax.tick_params(which="minor", bottom=False, left=False)
            ax.grid(which='minor', color=line_spacing_color, linestyle='-', linewidth=line_width)

        # Color border around the matrix the same as the line spacing.
        # This is to prevent the odd artefacts that may occur on the border by coloring over them.
        for key, spine in ax.spines.items():
            spine.set_color(line_spacing_color)
            spine.set_linewidth(round(line_width / 2))

    else:
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

    if show_counts:
        # http://stackoverflow.com/questions/21712047/matplotlib-imshow-matshow-display-values-on-plot
        min_val, max_val = 0, len(cm)
        ind_array = np.arange(min_val, max_val, 1.0)
        x, y = np.meshgrid(ind_array, ind_array)
        for i, (x_val, y_val) in enumerate(zip(x.flatten(), y.flatten())):
            c = cm[int(x_val), int(y_val)]
            ax.text(y_val, x_val, c, va='center', ha='center', color=cm_font_color,
                    fontproperties=NORMAL_FONT_PROP,
                    fontsize=cm_font_size)

    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    plt.tight_layout()

    if show_arrow_bars:
        # Add arrow bars outside the figure.
        # These were adjusted to try to make the arrow locations occur
        # relative to the size of the confusion matrix.
        # This was tested on a 7x7 matrix, but may have to be adjusted for the 2x2.
        # The original 2x2 settings are commented out below.
        plt.annotate(s='',
                     xy=(-len(cm) * 0.22, -0.5),  # (-0.75, -0.5),
                     xytext=(-len(cm) * 0.22, len(cm) * 0.92),  # (-0.75, 1.5),
                     arrowprops=dict(arrowstyle='<->', color='darkgrey'),
                     annotation_clip=False)
        plt.annotate(s='',
                     xy=(len(cm) * 0.92, len(cm) + 0.25),  # (1.5, 1.75),
                     xytext=(-0.5, len(cm) + 0.25),  # (-0.5, 1.75),
                     arrowprops=dict(arrowstyle='<->', color='darkgrey'),
                     annotation_clip=False)

    if save_path is not None:
        plt.savefig(save_path, transparent=True, bbox_inches="tight")


def plot_ROC_curve(y_probs, y_true, title='ROC curve', save_path=None,
                   figsize=(12.8, 7.2), specificity_curve=False,
                   lw=3, curve_color=np.asarray([71 / 255, 195 / 255, 157 / 255]), fontsize=20,
                   x_label=None, y_label=None,
                   ):
    """
    Plot the Receiver Operating Characteristic (ROC) curve.

    By default this is defined as true positive rate (TPR) against the false positive rate (FPR) at various thresholds.

    Note: this implementation is restricted to the binary classification task.

    Args:
        y_probs: A numpy array containing the probabilities of the positive class.
        y_true: A numpy array of the true binary labels (*not* encoded as 1-hot).
        title: String with the title.
        save_path: If `save_path` specified, save confusion matrix in that location.
        figsize: Tuple indicating the size of the figure.
        specificity_curve: Boolean value indicating the type of curve to show.
                If False, show the traditional ROC curve.
                If True, show the specificity, sensitivity curve.
        lw: Line width of the ROC curve.
        curve_color: Color of the ROC curve.
        fontsize: Font size of the labels.
        x_label: If not None, overwrite the x label.
        y_label: If not None, overwrite the y label.
    Returns: Figure axis.
    """
    utils.check_input_samples(y_probs, y_true)

    if not np.array_equal(y_true, np.asarray(y_true).astype(bool)):
        raise ValueError('y_true must contain the true binary labels.')

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_probs)

    if specificity_curve:
        # Show the specificity and sensitivity curve.
        # Specificity = 1 - false_positive_rate
        xaxis_values = 1 - fpr
        xlabel = 'Specificity'
        ylabel = 'Sensitivity'
    else:
        # Show the traditional FPR and TPR curve.
        xaxis_values = fpr
        xlabel = 'False Positive Rate (FPR)'
        ylabel = 'True Positive Rate (TPR)'

    # Overwrite the labels if given.
    if x_label is not None:
        xlabel = x_label
    if y_label is not None:
        ylabel = y_label

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    # Remove borders except for bottom
    ax.spines['bottom'].set_color('lightgrey')
    ax.spines['top'].set_color(None)
    ax.spines['right'].set_color(None)
    ax.spines['left'].set_color(None)
    ax.plot(xaxis_values, tpr, color=curve_color, lw=lw, zorder=0)
    plt.ylabel(ylabel, rotation=0, fontproperties=BOLD_FONT_PROP, fontsize=fontsize, labelpad=fontsize * 4)
    plt.xlabel(xlabel, fontproperties=BOLD_FONT_PROP, fontsize=fontsize, labelpad=fontsize * 1.5)

    if title:
        plt.title(title)
    if save_path is not None:
        plt.savefig(save_path)
    return ax


def plot_dermatologists_cnn_comparision(y_probs, y_true, derms_specificity_scores, derms_sensitivity_scores,
                                        ai_specificity,
                                        ai_sensitivity, derms_avg_specificity, derms_avg_sensitivity,
                                        label_fontsize=26,
                                        markersize=5,
                                        aip_plot_color=np.asarray([202 / 255, 100 / 255, 223 / 255]),
                                        derms_plot_color=np.asarray([84 / 255, 160 / 255, 200 / 255]),
                                        ai_avg_color=np.asarray([77 / 255, 0 / 255, 174 / 255]),
                                        derms_avg_color=np.asarray([18 / 255, 115 / 255, 255 / 255]),
                                        save_path=None,
                                        derm_avg_marker='D',
                                        human_avg_marker_size=14,
                                        ai_avg_marker='P',
                                        ai_marker_size=18,
                                        x_label=None, y_label=None,
                                        ):
    """
    Plot the comparision of dermatologists and CNN performance for comparision through a ROC curve
    Args:
        y_probs: A numpy array containing the probabilities of the positive class.
        y_true: A numpy array of the true binary labels (*not* encoded as 1-hot).
        derms_specificity_scores: A numpy array containing the specificity scores achieved by individual dermatologists.
        derms_sensitivity_scores:A numpy array containing the sensitivity scores achieved by individual dermatologists.
        ai_specificity: The average specificity achieved by the CNN on the positive class.
        ai_sensitivity: The average sensitivity achieved by the CNN on the positive class.
        derms_avg_specificity: The average specificity achieved by the dermatologists on the positive class.
        derms_avg_sensitivity: The average sensitivity achieved by the dermatologists on the positive class.
        label_fontsize: The size of the font for the axis labels.
        markersize: The size of the dots used to plot the dermatologists performance.
        derms_plot_color: The color to be used to plot dermatologists specificity and sensitivity scores.
        ai_avg_color: The color to be used to to plot the average AI performance.
        derms_avg_color: The color to be used to to plot the average dermatologists performance.
        save_path:
        derm_avg_marker:
        human_avg_marker_size: Integer indicating the size of the human average marker.
        ai_avg_marker:
        ai_marker_size:
        x_label: If not None, overwrite the x label with the given string.
        y_label: If not None, overwrite the y label with the given string.
    Returns: A ROC curve and points that compares human and CNN performance.
    """

    ax = plot_ROC_curve(y_probs, y_true,
                        title=None,
                        curve_color=aip_plot_color,
                        specificity_curve=True,
                        fontsize=label_fontsize,
                        x_label=x_label, y_label=y_label)

    # Plot human individual performance.
    ax.plot(derms_specificity_scores, derms_sensitivity_scores,
            ' o',
            color=derms_plot_color, markersize=markersize,
            markeredgecolor='royalblue', )

    # Plot human average performance.
    ax.plot(derms_avg_specificity, derms_avg_sensitivity,
            color=derms_avg_color,
            markersize=human_avg_marker_size,
            marker=derm_avg_marker,
            markeredgecolor='black',
            alpha=0.7,  # Transparent marker.
            )

    # Plot AI average performance.
    ax.plot(ai_specificity, ai_sensitivity,
            color=ai_avg_color,
            markersize=ai_marker_size,
            marker=ai_avg_marker,
            markeredgecolor='black', zorder=10,
            alpha=0.6,  # Transparent marker.
            )

    ax.grid(color='lightgrey', linestyle='-', linewidth=0.5)
    ax.set_aspect('equal', 'box')
    plt.xlim([0.0, 1.02])
    plt.ylim([0.0, 1.03])

    for label in ax.get_xticklabels():
        label.set_fontproperties(NORMAL_FONT_PROP)

    for label in ax.get_yticklabels():
        label.set_fontproperties(NORMAL_FONT_PROP)

    tick_fontsize = 14
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    # Get rid of the zero's at origin.
    ax.xaxis.get_major_ticks()[0].label1.set_visible(False)
    ax.yaxis.get_major_ticks()[0].label1.set_visible(False)
    # Add a custom 0 at origin to be on the diagonal so only a single zero appears at origin.
    ax.text(-0.0625, -0.0425, "0.0", fontproperties=NORMAL_FONT_PROP, fontsize=tick_fontsize)

    ax.annotate(s='', xy=(-0.1, 1), xytext=(-0.1, 0),
                arrowprops=dict(arrowstyle='<->', color='darkgrey'),
                annotation_clip=False)

    ax.annotate(s='', xy=(1., -.1), xytext=(-0, -0.1),
                arrowprops=dict(arrowstyle='<->', color='darkgrey'),
                annotation_clip=False)

    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    if save_path is not None:
        plt.savefig(save_path, transparent=True, bbox_inches="tight")


def plot_precision_recall_curve(y_probs, y_true, title='2-class Precision-Recall curve', save_path=None):
    """
    Plot Precision-Recall curve for a binary classification task.

    Note: this implementation is restricted to the binary classification task.

    Args:
        y_probs: A numpy array containing the probabilities of the positive class.
        y_true: A numpy array of the true binary labels (*not* encoded as 1-hot).
        title: String with the title.
        save_path: If `save_path` specified, save confusion matrix in that location.

    Returns: Nothing, displays Precision-Recall curve
    """
    if not np.array_equal(y_true, y_true.astype(bool)):
        raise ValueError('y_true must contain the true binary labels.')

    precision, recall, _ = metrics.precision_recall_curve(y_true, y_probs)
    average_precision = metrics.average_precision_score(y_true, y_probs)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title(title + ': AP={0:0.2f}'.format(average_precision))

    if save_path is not None:
        plt.savefig(save_path)


def plot_threshold(threshold, correct, errors, title='Threshold Tuning'):
    '''

    Args:
        threshold: List of thresholds
        correct: List of correct predictions per threshold
        errors: List of error predictions per threshold
        title: Title of the plot

    Returns: Interactive Plot

    '''
    if not len(threshold) == len(correct) == len(errors):
        raise ValueError('The length of the arrays introduced do not coincide (%i), (%i), (%i)'
                         % (len(threshold), len(correct), len(errors)))

    trace1 = go.Scatter(
        x=threshold,
        y=correct,
        name='Correct Predictions'
    )

    trace2 = go.Scatter(
        x=threshold,
        y=errors,
        name='Removed Errors'
    )

    layout = dict(title=title,
                  xaxis=dict(title='Threshold Value'),
                  yaxis=dict(title='Network Predictions (%)'),
                  )

    data = [trace1, trace2]
    fig = dict(data=data, layout=layout)
    iplot(fig, filename='Threshold Tuning')


def plot_images(image_paths, n_images, title='', subtitles=None, n_cols=5, image_res=(20, 20), save_name=None):
    '''

    Args:
        image_paths: List with image_paths
        n_images: Number of images to show in the plot. Upper bounded by len(image_paths).
        title: Title for the plot
        subtitles: Subtitles for plots
        n_cols: Number of columns to split the data
        image_res: Plot image resolution
        save_name: If specified, will save the plot in save_name path

    Returns: Plots images in the screen

    '''
    if subtitles is not None and len(subtitles) != n_images:
        raise ValueError('Number of images and subtitles is different. There are %d images and %d subtitles'
                         % (n_images, len(subtitles)))
    n_row = 0
    n_col = 0
    total_images_plot = min(len(image_paths), n_images)
    if total_images_plot <= n_cols:
        f, axes = plt.subplots(nrows=1, ncols=n_cols, figsize=image_res)
        plt.title(title)

        for i in range(n_cols):
            if i < total_images_plot:
                img = plt.imread(image_paths[i])
                axes[n_col].imshow(img, aspect='equal')
                axes[n_col].grid('off')
                axes[n_col].axis('off')
                if subtitles is not None:
                    axes[n_col].set_title(subtitles[i])
            else:
                f.delaxes(axes[n_col])
            n_col += 1

    else:
        n_rows_total = int(np.ceil(n_images / n_cols))

        f, axes = plt.subplots(nrows=n_rows_total, ncols=n_cols, figsize=image_res)

        for i in range(n_rows_total * n_cols):
            if i < total_images_plot:
                img = plt.imread(image_paths[i])
                axes[n_row, n_col].imshow(img, aspect='equal')
                axes[n_row, n_col].grid('off')
                axes[n_row, n_col].axis('off')
                if subtitles is not None:
                    axes[n_row, n_col].set_title(subtitles[i])
            else:
                axes[n_row, n_col].grid('off')
                f.delaxes(axes[n_row, n_col])

            n_col += 1

            if n_col == n_cols:
                n_col = 0
                n_row += 1

    if save_name is not None:
        plt.savefig(save_name)


def plot_concept_metrics(concepts, metrics, x_axis_label, y_axis_label, title=None):
    if len(concepts) != len(metrics):
        raise ValueError('Dimensions of concepts (%i) and metrics array (%i) do not match' % (len(concepts),
                                                                                              len(metrics)))
    data = [[] for i in range(len(concepts))]
    for i in range(len(concepts)):
        data[i] = go.Scatter(
            x=np.arange(1, len(metrics[i]) + 1),
            y=metrics[i],
            mode='lines',
            name=concepts[i],
        )
    layout = go.Layout(
        title=title,
        xaxis=dict(
            title=x_axis_label
        ),
        yaxis=dict(
            title=y_axis_label
        ),
    )
    fig = go.Figure(data=data, layout=layout)
    iplot(fig, filename='line-mode')


def plot_models_performance(eval_dir, individual=False, class_idx=None, metric=None, save_name=None):
    """
       Enables plotting of a single metric from multiple evaluation metrics files
       Args:
           eval_dir: A directory that contains multiple metrics files
           individual: If True, compare individual metrics. Otherwise, compare average metrics.
           class_idx: The index of class for when comparing individual metrics
           metric: The metric to be plotted for comparison
           save_name:  If `save_path` specified, save plot in that location

       Returns: Nothing. If save_path is provided, plot is stored.

    """
    x_axis = []
    y_axis = []
    tick_label = []
    i = 0
    for result_csv in os.listdir(eval_dir):
        if utils.check_result_type(result_csv, individual):
            df = pd.read_csv(os.path.join(eval_dir, result_csv))
            tick_label.append(result_csv[:result_csv.rfind('_')])
            if individual:
                if isinstance(class_idx, int) and isinstance(metric, str):
                    y_axis.append(df[metric][class_idx])
                    x_axis.append(i)
                else:
                    raise ValueError('Unsupported type: class_idx, metric')
            else:
                if metric:
                    y_axis.append(df[metric][0])
                    x_axis.append(i)
                else:
                    raise ValueError('Missing required option: metric')
            i += 1
    plt.bar(x_axis, y_axis)
    plt.ylabel(str(metric))
    plt.xticks(x_axis, tick_label, rotation='vertical')
    if save_name:
        plt.savefig(save_name)


def plot_confidence_interval(values_x, values_y, lower_bound, upper_bound, title=''):
    if len(values_x) == len(values_y) == len(lower_bound) == len(upper_bound):
        upper_bound = go.Scatter(
            name='Upper Bound',
            x=values_x,
            y=upper_bound,
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            fillcolor='rgba(25, 25, 255, 0.2)',
            fill='tonexty')

        trace = go.Scatter(
            name='Mean',
            x=values_x,
            y=values_y,
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
            fillcolor='rgba(25, 25, 255, 0.2)',
            fill='tonexty')

        lower_bound = go.Scatter(
            name='Lower Bound',
            x=values_x,
            y=lower_bound,
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines')

        data = [lower_bound, trace, upper_bound]

        layout = go.Layout(
            xaxis=dict(title='Top-1 Probability'),
            yaxis=dict(title='Confidence Interval'),
            title=title,
            showlegend=False)

        fig = go.Figure(data=data, layout=layout)
        iplot(fig, filename='confidence_interval')
    else:
        raise ValueError('Arrays "values_x", "values_y", "lower_bound" and '
                         '"upper_bound" should have the same dimension')


def plot_histogram_probabilities(correct, errors, title, bins=200):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=correct,
                               name='correct',
                               nbinsx=bins,
                               marker_color='#08D82E'))
    fig.add_trace(go.Histogram(x=errors,
                               name='errors',
                               nbinsx=bins,
                               marker_color='#FC5529'))

    # Overlay both histograms
    fig.update_layout(
        barmode='overlay',
        title_text=title,  # title of plot
        xaxis_title_text='Top-1 Probability',  # xaxis label
        yaxis_title_text='Counts',  # yaxis label
    )
    fig.update_traces(opacity=0.60)
    fig.show()


def sensitivity_scatter_plot(values_x, values_y, labels, axis_x, axis_y, title):
    if len(values_x) != len(values_y):
        raise ValueError('Both arrays "values_x" and "values_y" should have the same dimension')

    data = [go.Scatter(
        x=values_x,
        y=values_y,
        mode='markers',
        marker=dict(
            color=values_y,
            colorscale="Viridis",
            symbol='circle',
            size=8,
        ),
        text=labels
    )]

    layout = dict(title=title,
                  xaxis=dict(title=axis_x),
                  yaxis=dict(title=axis_y),
                  )

    fig = dict(data=data, layout=layout)
    iplot(fig, filename='scatter-plot')
