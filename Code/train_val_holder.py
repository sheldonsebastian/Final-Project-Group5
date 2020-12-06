# %% --------------------
import os

import random

# %% --------------------
seed = 42

# %% --------------------
BASE_DIR = "/home/ubuntu/Deep-Learning/Exam2/"

# %% --------------------
# create 3 folders in specified path
# train
os.makedirs(BASE_DIR + "root_data/train/uncovered", exist_ok=True)
os.makedirs(BASE_DIR + "root_data/train/covered", exist_ok=True)
os.makedirs(BASE_DIR + "root_data/train/incorrect", exist_ok=True)

# val
os.makedirs(BASE_DIR + "root_data/validation/uncovered", exist_ok=True)
os.makedirs(BASE_DIR + "root_data/validation/covered", exist_ok=True)
os.makedirs(BASE_DIR + "root_data/validation/incorrect", exist_ok=True)

# holdout
os.makedirs(BASE_DIR + "root_data/holdout/uncovered", exist_ok=True)
os.makedirs(BASE_DIR + "root_data/holdout/covered", exist_ok=True)
os.makedirs(BASE_DIR + "root_data/holdout/incorrect", exist_ok=True)


# %% --------------------
def splitter_and_mover(label):
    # store file names in uncovered_arr
    arr = []

    # read the uncovered folder
    for path in [f for f in os.listdir(BASE_DIR + "data_masks_ml2/" + label)]:
        arr.append(path)

    # len(uncovered_arr) ==> randomly choose 80% of indices
    train_size = int(len(arr) * 0.8)
    val_size = int(train_size * 0.2)

    print("Total Size for " + label + "::" + str(len(arr)))
    print("Train Size for " + label + "::" + str(train_size))
    print("Validation Size for " + label + "::" + str(val_size))
    print("Holdout Size for " + label + "::" + str(len(arr) - train_size))

    # take random 80% indices from the 80% of original indices and cp to train
    random.seed(seed)
    # random.sample(..) gets elements without replacement
    train_imgs = random.sample(arr, train_size)

    # set difference to get 20 % of indices and cp those index files into holdout folder
    holdout_imgs = set(arr) - set(train_imgs)
    holdout_imgs = list(holdout_imgs)

    # take 20 % of images from train indices
    random.seed(seed)
    # random.sample(..) gets elements without replacement
    val_imgs = random.sample(train_imgs, val_size)

    # subtract val_imgs from train_imgs
    train_imgs = list(set(train_imgs) - set(val_imgs))

    # copy all the files in the indexes to appropriate folder
    for train_id in train_imgs:
        os.system(
            "cp " + BASE_DIR + "data_masks_ml2/" + label + "/" + train_id + " " + BASE_DIR + "root_data/train/" + label)

    print("Finished Train set")

    for val_index in val_imgs:
        os.system(
            "cp " + BASE_DIR + "data_masks_ml2/" + label + "/" + val_index + " " + BASE_DIR + "root_data/validation/" + label)

    print("Finished Validation set")

    for holdout_index in holdout_imgs:
        os.system(
            "cp " + BASE_DIR + "data_masks_ml2/" + label + "/" + holdout_index + " " + BASE_DIR + "root_data/holdout/" + label)

    print("Finished Holdout set")
    print("*" * 25)
    print()


# %% --------------------
splitter_and_mover("uncovered")
print("Finished Uncovered")
splitter_and_mover("covered")
print("Finished Covered")
splitter_and_mover("incorrect")
print("Finished Incorrect")

print("Finished")