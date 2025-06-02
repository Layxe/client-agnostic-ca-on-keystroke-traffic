import matplotlib as mpl
import matplotlib.pyplot as plt

PLT_COLORS = ["#64378C", "#C42847", "#759FBC", "#288732", "#E7F9A9"]

IEEE_COLORS = [
    "#00629B", "#BA0C2F", "#658D1B"
]

def use_IEEE_colors():
    """
    Set the color cycle to the IEEE colors.
    """
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=IEEE_COLORS)


def set_base_style():
    plt.style.use('ggplot')

    # Set legend box color
    mpl.rcParams['legend.facecolor'] = 'white'
    mpl.rcParams['legend.edgecolor'] = 'gray'
    mpl.rcParams['legend.frameon'] = True

    mpl.rcParams['axes.facecolor'] = 'white'
    mpl.rcParams['axes.grid']      = True
    mpl.rcParams['grid.color']     = 'lightgrey'
    mpl.rcParams['grid.linestyle'] = '-'

    mpl.rcParams['axes.edgecolor']  = 'gray'
    mpl.rcParams['axes.linewidth']  = 1
    mpl.rcParams['axes.titlesize']  = 20
    mpl.rcParams['axes.labelcolor'] = 'black'
    mpl.rcParams['axes.labelsize']  = 12

    # Set ticks color
    mpl.rcParams['xtick.color'] = 'black'
    mpl.rcParams['ytick.color'] = 'black'

    mpl.rcParams['lines.linewidth'] = 2

    mpl.rcParams['errorbar.capsize'] = 5

    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=PLT_COLORS)

def set_eda_style():
    mpl.rcParams['figure.figsize'] = (20, 6)

    set_base_style()


def set_document_style(half_size=True, ratio='square'):
    set_base_style()
    linewidth_pt = 516

    if half_size:
        linewidth_pt = 258

    linewidth_inch = linewidth_pt / 72.0

    if ratio == 'square':
        plt.figure(figsize=(linewidth_inch, linewidth_inch))
    elif ratio == 'golden':
        plt.figure(figsize=(linewidth_inch, linewidth_inch * 0.618))

    # Set font size to 8 pt
    plt.rcParams.update({
        'font.size': 8,
        'axes.titlesize': 8,
        'axes.labelsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.titlesize': 8,
        'lines.linewidth': 1,
    })

def set_presentation_style():
    set_base_style()
    mpl.rcParams['figure.figsize'] = (6, 3)
    mpl.rcParams['figure.dpi'] = 200
    mpl.rcParams['axes.labelsize'] = 10

    # Set font size for x ticks
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    mpl.rcParams['lines.linewidth'] = 2