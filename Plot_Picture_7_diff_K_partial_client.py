import matplotlib.pyplot as plt
import numpy as np



if __name__ == '__main__':

    x = [10,15,20,25,30,35,40,45,50]

    y1 = [910.73,64.85,14.58,4.38,2.02,1.21,0.87,0.75,1.16]
    y11= [86.53,7.18,1.79,1.06,0.84,0.68,0.62,0.60,0.68]
    y111=[0.87,0.62,0.59,0.57,0.54,0.54,0.55,0.56,0.57]
    y2= [0.53,0.53,0.53,0.53,0.53,0.53,0.53,0.53,0.53]



    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(1, 1, 1)




    l01, = ax.plot(x, y1, label='$\epsilon=50$', marker='^', markersize=12, markevery=1, color='blue',markerfacecolor='none',linewidth =2)

    l011, = ax.plot(x, y11, label='$\epsilon=60$', marker='o', markersize=12, markevery=1, color='orangered',markerfacecolor='none',linewidth =2)

    l0111, = ax.plot(x, y111, label='$\epsilon=100$', marker='d', markersize=12, markevery=1, color='gold',markerfacecolor='none',linewidth =2)

    l02, = ax.plot(x, y2, label='Non-private', marker='>', markersize=12, markevery=1, color='m',markerfacecolor='none',linewidth =2)


    ax.set_xlabel("Number of the Chosen Clients (K)", fontsize=25)
    ax.set_ylabel("Value of the Loss Function", fontsize=25)
    ax.set_ylim(0.4,2.4)
    ax.set_yticks(np.arange(0.4, 2.5, 0.2))
    ax.set_xlim(10,50.1)
    ax.set_xticks(np.arange(10,50.1,5))
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("P7_Different_K_different_e_partial_client.png")
    plt.show()
