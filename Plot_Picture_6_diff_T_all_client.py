import matplotlib.pyplot as plt
import numpy as np



if __name__ == '__main__':

    x = [5,10,15,20,25,30]

    y1 = [1.53,1.00,1.10,20.39,2693.09,1455938.375]
    y11= [1.52,1.01,0.83,4.00,132.51,20895.78]
    y111=[1.52,1.00,0.77,0.70,1.01,8.69]
    y2= [1.52,1.00,0.75,0.62,0.54,0.49]


    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(1, 1, 1)




    l01, = ax.plot(x, y1, label='$\epsilon=50$', marker='^', markersize=12, markevery=1, color='blue',markerfacecolor='none',linewidth =2)

    l011, = ax.plot(x, y11, label='$\epsilon=60$', marker='o', markersize=12, markevery=1, color='orangered',markerfacecolor='none',linewidth =2)

    l0111, = ax.plot(x, y111, label='$\epsilon=100$', marker='d', markersize=12, markevery=1, color='gold',markerfacecolor='none',linewidth =2)


    l02, = ax.plot(x, y2, label='Non-private', marker='>', markersize=12, markevery=1, color='m',markerfacecolor='none',linewidth =2)


    ax.set_xlabel("Number of Maximum Aggregation Times (T)", fontsize=25)
    ax.set_ylabel("Value of the Loss Function", fontsize=25)
    ax.set_ylim(0.4,2.4)
    ax.set_yticks(np.arange(0.4, 2.5, 0.2))
    ax.set_xlim(5,30.1)
    ax.set_xticks(np.arange(5,30.1,5,))
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("P6_Different_T_different_e_all_client.png")
    plt.show()
