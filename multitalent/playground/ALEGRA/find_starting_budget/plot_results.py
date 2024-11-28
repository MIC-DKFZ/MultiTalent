import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    csv_file = '/home/isensee/Downloads/ALEGRA_STARTING_BUDGET - stuff_for_plotting.csv'  # from https://docs.google.com/spreadsheets/d/1mIdfIE8YwBQ15abrZfy7u1xLqfdZpZGqNC6u8hHeDEg/edit#gid=1199429017

    fig = plt.figure()
    content = np.loadtxt(csv_file, delimiter=',', dtype=str)
    less_cases = content[content[:, -1] == 'fewer_train_cases', :-1].astype(float)
    patches = content[content[:, -1] == 'patches', :-1].astype(float)
    patches_better = content[content[:, -1] == 'patches_improved', :-1].astype(float)
    plt.plot(less_cases[:, 0], less_cases[:, 1], color='r')
    plt.plot(patches[:, 0], patches[:, 1], color='g')
    plt.plot(patches_better[:, 0], patches_better[:, 1], color='b')
    plt.legend(('fewer train cases', 'sparse annot. (patches)', 'sparse annot. (patches_better)'))
    plt.xlabel('percent annotated')
    plt.ylabel('Dice')
    plt.title('ALEGRA Patch Annotation efficiency')
    plt.savefig('ALEGRA_patchAnnos' + '.png')
    plt.close()