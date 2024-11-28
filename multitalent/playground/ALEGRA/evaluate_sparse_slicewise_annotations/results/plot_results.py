import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    datasets = {
        'BraTS2021': 'ALEGRA_SLICEWISE_ANNO - plot_BraTS.csv',
        'AMOS2022': 'ALEGRA_SLICEWISE_ANNO - plot_AMOS.csv',
        'ALEGRA': 'ALEGRA_SLICEWISE_ANNO - plot_ALEGRA.csv',
    }
    for d in datasets.keys():
        fig = plt.figure()
        content = np.loadtxt(datasets[d], delimiter=',', dtype=str)
        less_cases = content[content[:, -1] == 'less_cases', :-1].astype(float)
        less_slices = content[content[:, -1] == 'less_slices', :-1].astype(float)
        plt.plot(less_cases[:, 0], less_cases[:, 1], color='r')
        plt.plot(less_slices[:, 0], less_slices[:, 1], color='g')
        plt.legend(('fewer train cases', 'sparse annot. (slices)'))
        plt.xlabel('percent annotated')
        plt.ylabel('Dice')
        plt.title(d)
        plt.savefig(d + '.png')
        plt.close()


