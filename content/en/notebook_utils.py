import matplotlib.pyplot as plt
from matplotlib import rc

def prepare_notebook_graphics():
    plt.ion()    
    font = {'size'   : 14}
    rc('font', **font)