import sys
import numpy as np
import matplotlib.pyplot as plt


def view_label_distr(filepath, bins=10):
    with open(filepath) as f:
        lines = f.readlines()
        vals = []
        for line in lines:
            vals.append(float(line.replace('\n', '')))

    vals_np = np.array(vals)
    mean = np.mean(vals_np)
    plt.axvline(x=mean, color='r', label='axvline - full height')
    plt.hist(vals_np, bins=bins)

    plt.legend(['mean = %.2f' % mean, 'label distr.'])
    plt.title('Distribution of angles in dataset ' + str(filepath))
    plt.show()

    return None


def plot_error_distr(errors_list, bins=20):
    errors_np = np.array(errors_list)

    fix, (ax1, ax2) = plt.subplots(1, 2)

    ax1.hist(errors_np, bins)
    ax1.set_title('Error distribution')
    ax2.plot(errors_list)
    ax2.set_title('Errors over time')
    plt.show()

    return None


def plot_pred_vs_target(targets, preds):
    plt.scatter(preds, targets, .5)
    plt.plot([-30, 30], [-30, 30], 'r-')
    plt.title('Prediction space')
    plt.legend(['Ideal', 'Predictions'])
    plt.xlabel('Predicted angle')
    plt.ylabel('Actual angle')
    plt.grid()
    plt.xticks()
    plt.yticks()
    plt.show()
    # TODO: fix this

    return None


def plot_pred_target_distributions(targets, preds, bins=20):
    fix, (ax1, ax2) = plt.subplots(1, 2)

    ax1.hist(targets, bins)
    ax1.set_title('Target distribution')
    ax2.hist(preds, bins)
    ax2.set_title('Prediction distribution')
    plt.show()


def plot_results(train_losses, val_losses):
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.legend(['train losses', 'val losses'])
    plt.title('Training progress')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.grid()
    plt.show()


if __name__ == '__main__':

    datapath = './Results/VGG/test_results.txt'
    with open(datapath) as f:
        lines = f.readlines()
        targets = []
        preds = []
        for line in lines:
            line = line.replace('(', '').replace(')', '')
            line = tuple(map(float, line.split(', ')))
            targets.append(line[0])
            preds.append(line[1])

    plot_pred_vs_target(targets, preds)
    plot_pred_target_distributions(targets, preds, bins=30)
