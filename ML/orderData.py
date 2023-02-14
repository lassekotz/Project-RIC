

from_path = "Data/"


def order_data(data_path):
    train_path = "Train/"
    val_path = "Val/"
    test_path = "Test/"

    f = open(data_path + "labels.txt")
    #t = open(train_path)
    for line in f:
        print(line)

    f_lines = f.read()
    #for line in f_lines:
        #print(line)






order_data(from_path)