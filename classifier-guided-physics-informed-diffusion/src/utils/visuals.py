import torchvision
import matplotlib.pyplot as plt

def show_image(img):
    img = torchvision.utils.make_grid(img)
    img = img / 2 + 0.5  # denormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()