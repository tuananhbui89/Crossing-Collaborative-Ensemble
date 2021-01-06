import numpy as np 
import matplotlib.pyplot as plt 

FONT_SIZE = 20
STICK_SIZE = 10

def plot_prediction(figname, image, adv, pred_nor, pred_adv): 
    nb = 2
    size = 5 
    fig = plt.figure(figsize=(size*nb, 2*size))

    ax = fig.add_subplot(2, nb, 1)
    ax.imshow(image)
    ax.grid(False)
    ax.axis('off')
    ax.set_title('Input image', fontsize=FONT_SIZE)

    ax = fig.add_subplot(2, nb, 1+nb)
    ax.imshow(adv)
    ax.grid(False)
    ax.axis('off')
    ax.set_title('Adversarial image', fontsize=FONT_SIZE)

    values = [pred_nor, pred_adv]
    titles = ['Model Ensemble', 'Model Ensemble']
    indexes = [2,4]

    for val, title, idx in zip(values, titles, indexes):
        ax = fig.add_subplot(2, nb, idx)
        ax.bar(np.arange(len(val)), val)
        ax.set_xlabel('Classes', fontsize=FONT_SIZE)
        ax.set_ylabel('Pred', fontsize=FONT_SIZE)
        ax.set_title(title + '; max index={}'.format(np.argmax(val)), fontsize=FONT_SIZE)
        ax.tick_params(labelsize=STICK_SIZE)

    plt.tight_layout()
    plt.savefig(figname)
    plt.close()   

def plot_images(savepath, x, y, x_adv, y_adv):
	class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

	plt.figure(figsize=(8,8))
	nbrow = 4
	nbcol = np.shape(x)[0] // nbrow * 2

	for r in range(nbrow): 
		_x = x if r%2==0 else x_adv 
		_y = y if r%2==0 else y_adv 

		for c in range(nbcol): 
			plt.subplot(nbrow,nbcol,r*nbrow+c+1)
			plt.xticks([])
			plt.yticks([])
			plt.grid(False)
			plt.imshow(_x[r//2+c], cmap=plt.cm.binary)
			# The CIFAR labels happen to be arrays, 
			# which is why you need the extra index
			plt.xlabel(class_names[_y[r//2+c]])			

	plt.tight_layout()
	plt.savefig(savepath, dpi=300)
	plt.close()  

def plot_historgram(figname, h_nat, h_adv, title): 
    plt.figure()
    plt.hist(h_nat, bins=50, color='blue', label='normal')
    plt.hist(h_adv, bins=50, color='red', label='adv')
    plt.legend()
    plt.xlabel('entropy H(f(x))')
    plt.ylabel('count')
    plt.title(title)
    plt.savefig(figname, dpi=300)
    plt.close()