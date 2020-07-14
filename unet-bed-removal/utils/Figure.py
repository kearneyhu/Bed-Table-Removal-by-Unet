import matplotlib.pyplot as plt
import numpy as np
def draw_accuracy(acc, epochs):
    max1 = max(acc)
    min1 = min(acc)
    epoch = np.arange(epochs)+1

    plt.figure(figsize=(20, 8))
    plt.title("Accuracy")
    plt.plot(epoch,acc, color="r", linestyle="-", linewidth=1.0, marker = 'o',markersize = 2)

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim(0.95*min1,1.05*max1 )
    plt.xlim(0,epochs+1)
    for a, b in zip(epoch, acc):
        a = round(a, 3)
        b = round(b, 3)
        plt.text(a, b, b, ha='center', va='bottom', fontsize= 10 )
    plt.savefig('results/mean_accuracy.jpg')
    np.savetxt('results/acc.txt', acc)

    # plt.show()

def draw_f1(f1, epochs):
    max1 = max(f1)
    min1 = min(f1)
    epoch = np.arange(epochs)+1
    # f1[0]=f1[1]
    plt.figure(figsize=(20, 8))
    plt.title("F1-score")
    plt.plot(epoch,f1, color="r", linestyle="-",linewidth=1.0, marker = 'o',markersize = 2)
    plt.xlabel("Epochs")
    plt.ylabel("F1-score")
    plt.xlim(0,epochs+1)
    plt.ylim(0.95*min1,1.05*max1)
    for c, d in zip(epoch, f1):
        c = round(c, 3)
        d = round(d, 3)
        plt.text(c, d, d, ha='center', va='bottom', fontsize=10)

    plt.savefig('results/mean_f1.jpg')
    np.savetxt('results/f1.txt', f1)
    # plt.show()
def draw_dice(dice, epochs):
    max1 = max(dice)
    min1 = min(dice)
    epoch = np.arange(epochs)+1
    plt.figure(figsize=(20, 8))
    plt.title("Dice-coefficient")
    plt.plot(epoch, dice, color="r", linestyle="-", linewidth=1.0,marker = 'o',markersize = 2)
    plt.xlabel("Epochs")
    plt.ylabel("Dice coefficient")
    plt.xlim(0,epochs+1)
    plt.ylim(0.95*min1,1.05*max1)
    for e, f in zip(epoch, dice):
        e = round(e, 3)
        f = round(f, 3)
        plt.text(e, f, f, ha='center', va='bottom', fontsize=10)
    plt.savefig('results/mean_dice.jpg')
    np.savetxt('results/dice.txt', dice)
    # plt.show()

def draw_loss(loss, epochs):
    max1 = max(loss)
    min1 = min(loss)
    epoch = np.arange(epochs)+1
    plt.figure(figsize=(20, 8))
    plt.title("Training-loss")
    plt.plot(epoch, loss, color="r", linestyle="-", linewidth=1.0,marker = 'o',markersize = 2)
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.xlim(0,epochs+1)
    plt.ylim(0.95*min1, 1.05*max1)
    for a, b in zip(epoch, loss):
        a = round(a, 3)
        b = round(b, 3)
        plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
    plt.savefig('results/Traning_loss.jpg')
    np.savetxt('results/loss.txt', loss)
    # plt.show()