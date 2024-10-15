import random, numpy as np, matplotlib.pyplot as plt


def check_train_loader(train_loader):
    img, msk = next(iter(train_loader))
    img_num = random.randint(0,img.shape[0]-1)
    test_img=img[img_num]
    test_mask=msk[img_num]
    test_mask=np.argmax(test_mask, axis=3)

    n_slice=random.randint(0, test_mask.shape[2])
    plt.figure(figsize=(12, 8))

    plt.subplot(221)
    plt.imshow(test_img[:,:,n_slice, 0], cmap='gray')
    plt.title('Image flair')
    plt.subplot(222)
    plt.imshow(test_img[:,:,n_slice, 1], cmap='gray')
    plt.title('Image t1ce')
    plt.subplot(223)
    plt.imshow(test_img[:,:,n_slice, 2], cmap='gray')
    plt.title('Image t2')
    plt.subplot(224)
    plt.imshow(test_mask[:,:,n_slice])
    plt.title('Mask')
    plt.show()