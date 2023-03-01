import numpy as np
import matplotlib.pyplot as plt

def view_label_distr(filepath, bins = 10):
    with open(filepath) as f:
        lines = f.readlines()
        vals = []
        for line in lines:
            vals.append(float(line.replace('\n', '')))

    vals_np = np.array(vals)
    mean = np.mean(vals_np)
    plt.axvline(x=mean, color='r', label='axvline - full height')
    plt.hist(vals_np, bins = bins)

    plt.legend(['mean = %.2f' %mean, 'label distr.'])
    plt.title('Distribution of angles in dataset ' + str(filepath))
    plt.show()

def plot_error_distr(errors_list, bins = 20):
    errors_np = np.array(errors_list)

    fix, (ax1, ax2) = plt.subplots(1, 2)

    ax1.hist(errors_np, bins)
    ax1.set_title('Error distribution')
    ax2.plot(errors_list)
    ax2.set_title('Errors over time')
    plt.show()
