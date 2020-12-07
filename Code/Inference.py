import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets, transforms
from torchvision import models

# define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define transforms
process_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])
                                        ])


def process_image(image, asNumpy=False):
	"""
	Function to read image and perform transforms
	:param image: image path(str)
	:param asNumpy: specification if numpy image be return (bool)
	:return: transformed images (tensor)
	"""

	im = Image.open(image)
	im = im.convert("RGB")
	im = process_transform(im)
	return im if not asNumpy else im.numpy()


def show_image(image, ax=None, title=None):
	"""
	Function to show an image that is a tensor
	:param image: image path (str)
	:param ax: axis plot
	:param title: title of plot
	:return:
	"""
	if ax is None:
		fig, ax = plt.subplots()

	# PyTorch tensors assume the color channel is the first dimension
	# but matplotlib assumes is the third dimension
	image = image.transpose(1, 2, 0)

	# Undo preprocessing transforms on tensor image
	mean = np.array([0.485, 0.456, 0.406])
	std = np.array([0.229, 0.224, 0.225])
	image = std * image + mean

	# Image needs to be clipped between 0 and 1 or it looks like noise when displayed
	image = np.clip(image, 0, 1)

	ax.imshow(image)
	plt.show()

def detect_mask(image_path, model, k=3):
	"""
	 Function to predict the class of an image using the trained model
	:param image_path: directory to where image is stored (str)
	:param model: trained model
	:param k: top class from the model prediction
	:return: k top probabilities and class from model output
	"""

	model.to(device)
	model.eval()
	image = process_image(image_path).unsqueeze_(0)
	output = model(image.to(device)) / 100
	output = output.float().cpu()
	return torch.topk(output, k)


def plot_probs(face_class, face_probs):

	plt.rcdefaults()
	fig, ax = plt.subplots()

	y_pos = np.arange(len(face_class))

	ax.barh(y_pos, face_probs, color='blue', ecolor='black')
	ax.set_yticks(y_pos)
	ax.set_yticklabels(face_class)
	ax.invert_yaxis()
	plt.show()


image_1 = "image1.jpg"
face_image = process_image(image_1, asNumpy=True)
# face_image.shape
show_image(face_image)

# load model from disk
t_model = torch.load("resnet50_2.pt")
base_model = t_model['base_model']
idx_to_class = t_model['class_to_idx']
facemask_model = base_model.load_state_dict(t_model['state_dict'], strict=False)


# get top probabilities and classes
top_probs, top_classes = detect_mask(image_1, facemask_model)

# face_class = [idx_to_class[x.item()] for x in top_classes[0]]
# face_probs = [x.item() for x in top_probs[0]]

