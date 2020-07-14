import matplotlib.pyplot as plt


def plot_img_and_mask(img, mask):
    # classes = mask.shape[2] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, 2)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    # if classes > 1:
    #     for i in range(classes):
    #         ax[i+1].set_title(f'Output mask (class {i+1})')
    #         ax[i+1].imshow(mask[:, :, i])
    # else:
    ax[1].set_title(f'Output mask')
    ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()
def plot_worst_acc(img,GT, mask, acc,epoch):
    fig, ax = plt.subplots(1, 3)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    ax[1].set_title('GT')
    ax[1].imshow(GT)
    ax[2].set_title(acc)
    ax[2].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.savefig('data/outcome/%dst epoch minACC.jpg'%(epoch+1))
    # plt.show()

def plot_worst_dice(img, GT, mask, dice,epoch):
    fig, ax = plt.subplots(1, 3)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    ax[1].set_title('GT')
    ax[1].imshow(GT)
    ax[2].set_title(dice)
    ax[2].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.savefig('data/outcome/%dst epoch minDICE.jpg'%(epoch+1))
    # plt.show()
def plot_worst_f1(img, GT, mask, f1,epoch):
    fig, ax = plt.subplots(1, 3)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    ax[1].set_title('GT')
    ax[1].imshow(GT)
    ax[2].set_title(f1)
    ax[2].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.savefig('data/outcome/%dst epoch minF1.jpg'%(epoch+1))
    # plt.show()
