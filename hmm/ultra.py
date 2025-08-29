import ultraplot as uplt
import numpy as np


def get_exploration_grid_plot(suptitle=""):
    array = [
        [1, 2, 3, 4, 5, 6],
        [7, 8, 9, 10, 11, 12],
        [13, 14, 15, 16, 17, 18],
        [19, 20, 21, 22, 23, 24],
    ]
    fig, axs = uplt.subplots(array, figwidth=5, span=False, sharex=True)
    axs.format(
        suptitle=suptitle, xlabel="xlabel", ylabel="ylabel",
    )
    return fig, axs


def generate_2d_array(rows, cols):
    return [[col + row * cols + 1 for col in range(cols)] for row in range(rows)]


def get_exploration_and_signal_grid(suptitle="", n_conditions=2,
                                    add_row=False, add_merge_row=False,
                                    figsize=(25,11), n_add_rows=2):

    n_rows = 1 + n_add_rows if add_row else 2
    array = generate_2d_array(n_rows, n_conditions)

    if add_merge_row:
        array.append(np.ones(n_conditions) * (max(max(array)) + 1))

    fig, axs = uplt.subplots(array, span=False, share=False, figsize=figsize)
    axs.format(
        suptitle=suptitle, xlabel="xlabel", ylabel="ylabel",
    )

    return fig, axs
