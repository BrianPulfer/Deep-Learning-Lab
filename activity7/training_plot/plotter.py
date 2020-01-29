def plot(y):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.plot(np.arange(len(y)), y,'b-')
    plt.show()


if __name__ == '__main__':
    file = open("./training.txt")
    lines = file.readlines()
    file.close()

    y = []
    for line in lines:
        y.append(float(line.split('Loss: ')[1].split(".\n")[0]))

    plot(y)
    print(len(y))
    print(max(y))
    print(min(y))
