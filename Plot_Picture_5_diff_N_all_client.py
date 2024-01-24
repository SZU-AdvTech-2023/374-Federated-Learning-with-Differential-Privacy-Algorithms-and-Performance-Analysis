import matplotlib.pyplot as plt
import numpy as np

def loaddata(fileName):
    k = []
    l = []
    infile = open(fileName, 'r')
    for line in infile:
        set = line.split(',')
        print(set)
        k.append(float(set[0]))
        l.append(float(set[1]))
    return (k, l)

if __name__ == '__main__':



    fileName01 = 'DPFL_N_50.txt'
    x1, y1 = loaddata(fileName01)

    fileName011 = 'DPFL_N_60.txt'
    x11, y11 = loaddata(fileName011)

    fileName0111 = 'DPFL_N_80.txt'
    x111, y111 = loaddata(fileName0111)


    fileName02 = 'DPFL_N_100.txt'
    x2, y2 = loaddata(fileName02)


    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(1, 1, 1)




    l01, = ax.plot(x1, y1, label='$N = 50$', marker='^', markersize=12, markevery=1, color='blue',markerfacecolor='none',linewidth =2)

    l011, = ax.plot(x11, y11, label='$N = 60$', marker='o', markersize=12, markevery=1, color='orangered',markerfacecolor='none',linewidth =2)

    l0111, = ax.plot(x111, y111, label='$N = 80$', marker='d', markersize=12, markevery=1, color='gold',markerfacecolor='none',linewidth =2)

    l02, = ax.plot(x2, y2, label='$N = 100$', marker='>', markersize=12, markevery=1, color='m',markerfacecolor='none',linewidth =2)


    ax.set_xlabel("Aggregation Time (t)", fontsize=25)
    ax.set_ylabel("Value of the Loss Function", fontsize=25)
    ax.set_ylim(0.4,2.4)
    ax.set_yticks(np.arange(0.4, 2.5, 0.2))
    ax.set_xlim(0.5,25.5)
    ax.set_xticks(np.arange(1,26,3))
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("P5_Different_N_all_client.png")
    plt.show()
