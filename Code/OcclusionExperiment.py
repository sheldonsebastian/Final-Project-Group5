# https://github.com/Niranjankumar-c/DeepLearning-PadhAI/tree/master/DeepLearning_Materials/6_VisualizationCNN_Pytorch
# %% --------------------Imports
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
from matplotlib.pyplot import cm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# %% --------------------
seed = 19

# %% --------------------
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

# %% --------------------
BASE_DIR = "/home/ubuntu/Workspaces/Project/"
save_path = BASE_DIR + "saved_models/"

BATCH_SIZE = 1

# %% --------------------
model_dict = torch.load(save_path + "resnet50_2.pt")

# %% --------------------
# Initialize the model
model = model_dict["base_model"]
model.load_state_dict(model_dict["state_dict"])

# %% --------------------LOAD DATA
input_size = 224
# create transformer
generic_transformer = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# create dataset
# using train dataset, since that is what we used to train the model
train_dataset = datasets.ImageFolder(BASE_DIR + "root_data/train/", transform=generic_transformer)
# train_dataset = datasets.ImageFolder(BASE_DIR + "Inference/", transform=generic_transformer)

# create dataloader
train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)

# %% --------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %% --------------------
model.to(device)
model.eval()


# %% --------------------
def denormalize_image(img):
    # define std correction to be made
    std_correction = np.asarray([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    # define mean correction to be made
    mean_correction = np.asarray([0.485, 0.456, 0.406]).reshape(3, 1, 1)

    # convert the tensor img to numpy img and de normalize
    npimg = np.multiply(img.cpu().numpy(), std_correction) + mean_correction

    return npimg


# %% --------------------
# view the original image by de-normalizing it
def imshow(img, title):
    """Custom function to display the image using matplotlib"""
    npimg = denormalize_image(img)

    # plot the numpy image
    plt.figure(figsize=(BATCH_SIZE * 4, 4))
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()


# %% --------------------
def show_batch_images(dataloader):
    # get next batch of data
    images, _ = next(iter(dataloader))

    # get dataloader classes
    classes = dataloader.dataset.classes

    # send image to device
    images = images.to(device)

    # run the model on the images
    outputs = model(images)

    # get the maximum class
    _, pred = torch.max(outputs.data, 1)

    # make grid
    img = torchvision.utils.make_grid(images)

    # call the function
    imshow(img, title=classes[pred.item()])

    return images, pred


# %% --------------------
images, pred = show_batch_images(train_dataloader)

# perform inference without occlusion to get maximum probability which will be used to normalize the
# heatmap
outputs = model(images)
print(outputs.shape)

# passing the outputs through softmax to interpret them as probability
outputs = nn.functional.softmax(outputs, dim=1)

# getting the maximum predicted label
prob_no_occ, pred = torch.max(outputs.data, 1)

# get the first item
prob_no_occ = prob_no_occ[0].item()

print("Probability of Prediction Class w/o Occlusion::" + str(prob_no_occ))


# custom function to conduct occlusion experiments
def occlusion(model, image, label, occ_size=50, occ_stride=50, occ_pixel=0.5):
    # get the width and height of the image
    width, height = image.shape[-2], image.shape[-1]

    # setting the output image width and height
    output_height = int(np.ceil((height - occ_size) / occ_stride))
    output_width = int(np.ceil((width - occ_size) / occ_stride))

    # create a white image of sizes we defined
    heatmap = torch.zeros((output_height, output_width))

    # iterate all the pixels in each column
    for h in range(0, height):
        for w in range(0, width):

            h_start = h * occ_stride
            w_start = w * occ_stride
            h_end = min(height, h_start + occ_size)
            w_end = min(width, w_start + occ_size)

            if (w_end) >= width or (h_end) >= height:
                continue

            input_image = image.clone().detach()

            # replacing all the pixel information in the image with occ_pixel(grey) in the specified location
            input_image[:, :, w_start:w_end, h_start:h_end] = occ_pixel

            # run inference on modified image
            output = model(input_image)
            output = nn.functional.softmax(output, dim=1)
            prob = output.tolist()[0][label]

            # setting the heatmap location to probability value
            heatmap[h, w] = prob

    return heatmap


# perform occlusion
# heatmap contains probabilities, thus b/w 0 and 1
heatmap = occlusion(model, images, pred.item(), 32, 14)

# visualize the heatmap
# displaying the image using seaborn heatmap and also setting the maximum value of gradient to
# probability
imgplot = sns.heatmap(heatmap, xticklabels=False, yticklabels=False, vmax=prob_no_occ)
plt.show()

# https://discuss.pytorch.org/t/create-heatmap-over-image/41408
# https://discuss.pytorch.org/t/solved-convert-color-of-tensors/58341
color_map = cm.get_cmap("magma")

# apply colormap to single channel heatmap to get 3 channels
hm_image = color_map(heatmap)[:, :, :3]
plt.imshow(hm_image)
plt.show()

hm_image = np.transpose(hm_image, (2, 0, 1))

# convert numpy cmap to tensor
hm_image = torch.from_numpy(hm_image).float()

# denormalize normalized input image
denormalized = denormalize_image(images.squeeze(0))

# convert numpy to tensor
denormalized = torch.from_numpy(denormalized).float()

# convert tensors to PIL
original = TF.to_pil_image(denormalized)
hm_image = TF.to_pil_image(hm_image)

print(original.size)
print(hm_image.size)

# resize heatmap to input image
hm_image = hm_image.resize((original.size[0], original.size[1]))
print(hm_image)

# superimpose
res = Image.blend(hm_image, original, 0.3)

plt.imshow(res)
plt.show()
