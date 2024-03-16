import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

x_test = np.load('cifar10_CTA_dataset/cifar10_x_test.npy')
y_test = np.load('cifar10_CTA_dataset/cifar10_y_test.npy')

trigger = Image.open('gen_c500_0.png')
trigger = np.array(trigger)

trigger = trigger[:32, :32, :]

def blend_images(image1, image2, alpha=0):
    return (alpha * image1 + (1 - alpha) * image2).astype(np.uint8)

x_test_triggered = np.array([blend_images(img, trigger) for img in x_test])

for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_test_triggered[i])
    plt.axis('off')
    plt.savefig('tes_trigger.png', bbox_inches='tight', pad_inches=0)
    
plt.show()

np.save('cifar10_x_test_with_trigger.npy', x_test)

y_test[:] = 0
np.save('cifar10_y_test_modified.npy', y_test)
