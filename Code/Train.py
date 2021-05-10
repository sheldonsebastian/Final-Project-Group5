# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
# %%--------------------------------------Imports
import os
import copy
import random
from sklearn.metrics import accuracy_score
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision import models
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
# %% --------------------
seed = 42

# %% --------------------
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

# %% --------------------
BASE_DIR = "/home/ubuntu/Deep-Learning/Exam2/"
save_path = BASE_DIR + "saved_models/"
os.makedirs(save_path, exist_ok=True)

# %% --------------------Configurable Parameters
model_name = "resnet50"

# covered, uncovered, incorrect
num_of_classes = 3

EPOCHS = 10
LR = 0.001

BATCH_SIZE = 512

# feature_extract_param = True means all layers frozen except the last user added layers
# feature_extract_param = False means all layers unfrozen and entire network learns new weights
# and biases
feature_extract_param = True


# %% --------------------
def set_parameter_requires_grad(model, feature_extracting):
    # feature_extract_param = True means all layers frozen except the last user added layers
    # feature_extract_param = False means all layers unfrozen and entire network learns new weights
    # and biases
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# %% --------------------initialize pretrained model & return input size desired by pretrained model
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(num_ftrs, num_classes))
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def plot_error(train_loss, valid_loss):
    plt.plot([x for x in range(len(train_loss))], train_loss, label='train', color='r')
    plt.plot([x for x in range(len(train_loss))], valid_loss, label='validation', color='g')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Validation Loss')
    plt.legend()
    plt.show()


# %% --------------------
# Initialize the model for this run
model, input_size = initialize_model(model_name, num_of_classes, feature_extract_param,
                                     use_pretrained=True)

# %% --------------------
# Print the model we just instantiated
print(model)

# %% --------------------
# get all parameters from model
params_to_update = model.parameters()

# if feature_extract_param is True, then we freeze all layers except the user added layers
if feature_extract_param:
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)

# %% --------------------
optimizer_ft = optim.Adam(params_to_update, lr=LR)

# %%--------------------------Load Data
# train transformation
# some augmentation
train_transformer = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.ColorJitter(saturation=[0, 1]),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# validation and holdout transformation is generic
# normalization and resize
generic_transformer = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# create dataset using ImageFolder
train_dataset = datasets.ImageFolder(BASE_DIR + "root_data/train/", transform=train_transformer)

val_dataset = datasets.ImageFolder(BASE_DIR + "root_data/validation/",
                                   transform=generic_transformer)

# holdout_dataset = datasets.ImageFolder("/home/ubuntu/Workspaces/Project/root_data/holdout/",
#                                     transform=generic_transformer)

# create dataloader
train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
# holdout_dataloader = DataLoader(holdout_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)

# create dataloader dictionary
dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

# %% --------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %% --------------------Criterion of CrossEntropy since we are doing classification
# CrossEntropyLoss combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
# https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
criterion = nn.CrossEntropyLoss()


# %% --------------------
def train_model(model, dataloaders, criterion, optimizer, num_epochs, is_inception=False):
    val_acc_history = []
    train_loss = []
    valid_loss = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # FORWARD
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output.
                    # In train mode we calculate the loss by summing the final output and the
                    # auxiliary output but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        # make prediction
                        outputs = model(inputs)
                        # find loss
                        loss = criterion(outputs, labels)

                    # select the class with highest value
                    # 1 specifies the axis
                    _, preds = torch.max(outputs, 1)

                    # BACKWARD + OPTIMIZE only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

                # finding accuracy
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # store train and valid loss separately
            if phase == 'train':
                train_loss.append(epoch_loss)
            else:
                valid_loss.append(epoch_loss)

            # plot train and valid loss after after 3 epochs
            if phase == "val":
                plot_error(train_loss, valid_loss)

            # deep copy the best model
            # model checkpoint
            if phase == 'val' and epoch_acc > best_acc:
                # save model state based on best val accuracy per epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, val_acc_history, optimizer


# %% --------------------
model.to(device)
# store class labels
idx_to_class = { v : k for k,v in train_dataset.class_to_idx.items()}
# %%--------------------------
model_resnet50, hist, optimizer_resnet50 = train_model(model, dataloaders_dict, criterion, optimizer_ft,
                                                       num_epochs=EPOCHS, is_inception=(model_name == "inception"))

# %% --------------------
torch.cuda.empty_cache()

checkpoint = {'base_model': model,
              'optim_state_dict': optimizer_resnet50.state_dict(),
              'state_dict': model_resnet50.state_dict(),
              'class_to_idx': idx_to_class
             }

torch.save(checkpoint, 'resnet50_2.pt')


# %% --------------------Load the best model
# model.load_state_dict(torch.load(save_path + "model_" + model_name + ".pt"))


# %% --------------------Evaluation metrics
def find_evaluation_metrics(folder):
    model.eval()

    # use generic transformer
    acc_dataset = datasets.ImageFolder(BASE_DIR + "root_data/" + folder + "/",
                                       transform=generic_transformer)
    acc_dataloader = DataLoader(acc_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)

    pred = []
    actual = []

    # get all outputs
    for inputs, label in acc_dataloader:
        with torch.no_grad():
            # logit
            predictions = model(inputs.to(device))

            # find index with max logit
            val, index = torch.max(predictions, 1)

            # append prediction
            pred.extend(index.tolist())

            # append actual
            actual.extend(label.tolist())

    accuracy = accuracy_score(actual, pred)
    f1 = f1_score(actual, pred, average='micro')
    cm = confusion_matrix(actual, pred)
    print("Accuracy::" + str(accuracy))
    print("F1 Score::" + str(f1))
    print("Confusion Matrix::" + str(cm))


# %% --------------------
print("Train Evaluation Metrics")
print("-" * 10)
find_evaluation_metrics("train")
print()
torch.cuda.empty_cache()

# %% --------------------
print("Validation Evaluation Metrics")
print("-" * 10)
find_evaluation_metrics("validation")
print()
torch.cuda.empty_cache()

# %% --------------------
print("Holdout Evaluation Metrics")
print("-" * 10)
find_evaluation_metrics("holdout")
print()
torch.cuda.empty_cache()
