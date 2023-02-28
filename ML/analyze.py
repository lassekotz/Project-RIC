import numpy as np
import matplotlib.pyplot as plt

def view_label_distr(filepath, bins = 10):
    with open(filepath) as f:
        lines = f.readlines()
        vals = []
        for line in lines:
            vals.append(float(line.replace('\n', '')))

    vals_np = np.array(vals)
    hist = np.histogram(vals_np)
    mean = np.mean(vals_np)
    plt.axvline(x=mean, color='r', label='axvline - full height')
    plt.hist(vals_np, bins = bins)

    plt.legend(['mean = %.2f' %mean, 'label distr.'])
    plt.title('Distribution of angles in dataset ' + str(filepath))
    plt.show()

def plot_error_distr():
    #TODO: Implement an error distribution tool
    pass

filepath = './Data/BigDataset/labels.txt'
view_label_distr(filepath, bins = 15)