from matplotlib.pyplot import figure, show
from seaborn import set_context, heatmap
from numpy import zeros_like, triu_indices_from

def get_corr_matrix(features, dataframe, exclude, encoder_func, reorder=False):
    encoded = encoder_func(features, dataframe, exclude)
    corr_matrix = encoded.corr()
    
    if reorder:
        reordered_index = corr_matrix.columns[[11, 5, 13, 20, 22, 1, 6,
                   8, 26, 27, 7, 24, 25, 3,
                   23, 4, 0, 2, 9, 18, 17,
                   16, 12, 21, 15, 19, 10, 14]]
        corr_matrix = corr_matrix.reindex(index=reordered_index,
                                          columns=reordered_index)
    return corr_matrix

def heat_map(matrix, cmap, size, fontsize):
    mask = zeros_like(matrix)
    mask[triu_indices_from(mask)] = 1
    set_context('poster', font_scale=fontsize)
    size = (size, size)
    fig = figure(figsize=size)
    heatmap(matrix,
            center=0,
            cmap=cmap,
            mask=mask,
            linewidths=0.5,
            linecolor='black',
            annot=True,
            fmt='.1f',
            cbar=False,
            figure=fig)
    show()
    