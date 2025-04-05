# Last modified: 2025-04-03
# Started: 2025-04-01

# This script trains and evaluates model for MBP1413H final project, in particular, brain tumour segmentation.
# It uses MONAI.
# UNETR is used, instead of UNet.

# ========== modify for each run ==========
run_id = "01"
# Exactly the same as UNETR_binary_run05.py. The bash script that runs this script allocates more time (15h)
# Created this "long" version because UNETR run05 is running, but subsequent tries for running one with longer time allocation failed,
# so the intension is to rerun run05 without modifying the .py code, without, by any chance, interferring with the current UNETR run.

output_dir = "output/UNETR/binary/long/"
output_dir_model = "model/UNETR/binary/long/"
output_nii = "output/nii_output/UNETR/binary/long/run" + str(run_id) + "/"

type_run = "UNETR_binary_long"
best_model_name = "best_metric_model_" + type_run + "_run" + str(run_id) + ".pth"
print("Name of the best model:", best_model_name)

# specify batch size
b_size = 8 #16

# set variables for early stopping
patience_interval = 25 # this is technically 50 because validation interval is 2
min_delta = 0.01 # for experiment

# set the number of epochs
max_epoch = 300

#. specify the number of images
n_images = 369 ##369, 25, 100
n_images_val_test = 55 ##55, 5, 20
# =========================================

from typing_extensions import dataclass_transform
import os
import re
import cv2
import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import imsave
import shutil
import nibabel as nib
import tempfile
import glob
import torch
import nibabel as nib

import monai
print(monai.__version__)
from monai.utils import first, set_determinism
from monai.transforms import (
    Compose,
    LoadImaged, #d for dictionary
    ToTensord, #d for dictionary

    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Resized,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
)

from monai.handlers.utils import from_engine
from monai.networks.nets import UNet, UNETR
from monai.networks.layers import Norm
from monai.metrics import DiceMetric, ConfusionMatrixMetric, HausdorffDistanceMetric
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract

from skimage.util import montage
import skimage.transform as skTrans
from skimage.transform import rotate

#torch.cuda.empty_cache()


#. Specify the path to the data
dataset_path = "/cluster/projects/gaitigroup/Users/Yumika/class/MBP1413H/final_pj/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/"
print(dataset_path)

# Set up data directory
directory = os.environ.get("BraTS2020")

# ===== Split dataset to training and testing =====
#. Create empty lists for each set
train_files = []
val_files = []
test_files = []


# ===== Create a data dictionary of training and validation images=====
# Create a data dictionary

#. create a list of paths to images and labels, respectively
train_test_images = [f"{dataset_path}BraTS20_Training_{volume_number:03d}/BraTS20_Training_{volume_number:03d}_flair.nii" for volume_number in range(1, n_images+1)] # train_images = [f"{dataset_path}BraTS20_Training_{volume_number:03d}/BraTS20_Training_{volume_number:03d}_flair.nii" for volume_number in range(1, n_images+1)]
train_test_labels = [f"{dataset_path}BraTS20_Training_{volume_number:03d}/BraTS20_Training_{volume_number:03d}_seg.nii" for volume_number in range(1, n_images+1)] # train_labels = [f"{dataset_path}BraTS20_Training_{volume_number:03d}/BraTS20_Training_{volume_number:03d}_seg.nii" for volume_number in range(1, n_images+1)]

#. create a data dictionary
data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_test_images, train_test_labels)]

# split the data dictionary into training, validation, and testing
train_val_files, test_files = data_dicts[:-n_images_val_test], data_dicts[-n_images_val_test:] # train_files, val_files = data_dicts[:-9], data_dicts[-9:]
train_files, val_files = train_val_files[:-n_images_val_test], train_val_files[-n_images_val_test:]

print("Length of train_files:", len(train_files))
print("Length of test_files (originall val_files):", len(test_files))
print("Length of val_files:", len(val_files))

#. plot a bar plot showing the number of images in each of training, validation, and testing
plt.figure(figsize=(8, 6))  # Adjust figure size if needed
plt.bar(['Train', 'Validation','Test'], [len(train_files), len(val_files),len(test_files)])
plt.title('Number of Volumes in Each Set')
plt.ylabel('Number of Volums')
plt.xlabel('Dataset Split')
plt.savefig(output_dir + 'n_images_train-val-test_' + type_run + '_' + str(run_id)  + '.png', dpi=300)


# ===== Set deterministic training for reproducibility ===== 
# . make randomness the same
set_determinism(seed=0)


# ===== Set up data loader and augmentation transforms ===== 
# Step 1: Load the images
# Step 2: Do any transforms
# Step 3: Convert transform-applied data to torch tensors in order to do training using pytorch

thrsh_asdis = 0.5 ##
### DO NOT TRUST WHAT I WROTE ON THE RIGHT --> "Equivalent to no transformation"
orig_transforms = Compose(
    [
        # Load images
        LoadImaged(keys=['image', 'label']),

        # Apply transformation
        # (this class) add[s] a new dimension that controls how many batches we have
        EnsureChannelFirstd(keys=['image', 'label']),

        # Binarize the label: all label values > threshold (e.g. 0.5) become 1, else 0
        # AsDiscreted(keys=["label"], threshold = thrsh_asdis),

        # Convert to torch tensors
        ToTensord(keys=['image', 'label'])
    ]
)



train_transforms = Compose(
    [
        # Load images
        LoadImaged(keys=['image', 'label']),

        # Apply transformation

        # add a new dimension that controls how many batches we have using the following class
        EnsureChannelFirstd(keys=['image', 'label']),

        # binarize the label: all label values > threshold (e.g. 0.5) become 1, else 0
        # AsDiscreted(keys=["label"], threshold = thrsh_asdis),

        # Change the dimension.
        Spacingd(keys=['image', 'label'], pixdim=(1.5, 1.5, 2)), # pixdim=(H,W,D)

        # Scale the intensity of the image. Apply it only to the image, not label.
        ScaleIntensityRanged(
            keys='image',
            a_min=-200, ## To be optimized. To select this value, 34:17 in this tutorial: https://www.youtube.com/watch?v=hqgZuatm8eE
            a_max=200, ## To be optimized.
            b_min=0.0, ##
            b_max=1.0
        ),

        # Crop foreground from both the image and label based on the image
        CropForegroundd(keys=['image', 'label'], source_key='image'),

        # Resize the image and the label
        Resized(keys=['image', 'label'], spatial_size=(160, 160, 64)), # the second parameter is new dimension

        # Normalize intensity of the image
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),

        # Apply data augmentation
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),        
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),

        # Convert to torch tensors
        ToTensord(keys=['image', 'label'])
    ]
)

val_transforms = Compose(
    [
        # Load images
        LoadImaged(keys=['image', 'label']),

        # Apply transformation
        # (this class) add[s] a new dimension that controls how many batches we have
        EnsureChannelFirstd(keys=['image', 'label']),

        # Binarize the label: all label values > threshold (e.g. 0.5) become 1, else 0
        # AsDiscreted(keys=["label"], threshold = thrsh_asdis),


        # Change the dimension. Make sure it's applied to both image and label, not just image.
        Spacingd(keys=['image', 'label'], pixdim=(1.5, 1.5, 2)), # pixdim=(H,W,D)

        # Scale the intensity of the image. Apply it only to the image, not label.
        ScaleIntensityRanged(
            keys='image',
            a_min=-200, ## To be optimized. To select this value, 34:17 in this tutorial: https://www.youtube.com/watch?v=hqgZuatm8eE
            a_max=200, ## To be optimized.
            b_min=0.0, ##
            b_max=1.0
        ),

        # Crop foreground from both the image and label based on the image
        CropForegroundd(keys=['image', 'label'], source_key='image'),

        # Resize the image and the label
        Resized(keys=['image', 'label'], spatial_size=(160, 160, 64)), # the second parameter is new dimension

        # Normalize intensity of the image
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),

        # Convert to torch tensors
        ToTensord(keys=['image', 'label'])
    ]
)


# Data loader
print("Data loader: original images------------------")
orig_ds = CacheDataset(data=train_files, transform=orig_transforms, cache_rate=0.5, num_workers=4) ## cache_rate=1.0 is going to load all the images into the memory before even training the model. set it to 0 for both train and val.
orig_loader = DataLoader(orig_ds, batch_size=b_size, shuffle=True, num_workers=1)

print("Data loader: train images----------------------")
train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.5, num_workers=4)
train_loader = DataLoader(train_ds, batch_size=b_size, shuffle=True, num_workers=1)

print("Data loader: validation images-----------------")
val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=0.5, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=b_size, shuffle=True, num_workers=1)

print("\Data loaded")

orig_patient = first(orig_loader) ### no transform
train_transformed_patient = first(train_loader) ## train_loader contains all the image
val_transformed_patient = first(val_loader)

print(torch.min(train_transformed_patient['image']))
print(torch.max(train_transformed_patient['image']))


# Visualize transformation
print("\nVisualizing transformation")
## Check if our function is working

# Extract the 2D slice (z=35) for each image
z1 = 35
z2 = 37
z3 = 40

orig_slice1 = orig_patient['image'][0, 0, :, :, z1]
orig_label1 = orig_patient['label'][0, 0, :, :, z1]
transformed_slice1 = train_transformed_patient['image'][0, 0, :, :, z1]
transformed_label1 = train_transformed_patient['label'][0, 0, :, :, z1]

orig_slice2 = orig_patient['image'][0, 0, :, :, z2]
orig_label2 = orig_patient['label'][0, 0, :, :, z2]
transformed_slice2 = train_transformed_patient['image'][0, 0, :, :, z2]
transformed_label2 = train_transformed_patient['label'][0, 0, :, :, z2]

orig_slice3 = orig_patient['image'][0, 0, :, :, z3]
orig_label3 = orig_patient['label'][0, 0, :, :, z3]
transformed_slice3 = train_transformed_patient['image'][0, 0, :, :, z3]
transformed_label3 = train_transformed_patient['label'][0, 0, :, :, z3]


# ------------
# Initialize the subplot function - 3 rows and 3 columns
fig, ax = plt.subplots(3, 4, figsize=(15, 10))

# first slice:
# 1. plot the original slice
ax[0, 0].imshow(orig_slice1, cmap="grey")
ax[0, 0].set_title("Original slice")

# 2. plot the original label
ax[0, 1].imshow(orig_label1)
ax[0, 1].set_title("Original label")

# 3. plot the transformed slice
ax[0, 2].imshow(transformed_slice1, cmap="grey")
ax[0, 2].set_title("Transformed slice")

# 4. plot the transformed label
ax[0, 3].imshow(transformed_label1)
ax[0, 3].set_title("Transformed label")

# second slice:
# 1. plot the original slice
ax[1, 0].imshow(orig_slice2, cmap="grey")
ax[1, 0].set_title("Original slice")

# 2. plot the original label
ax[1, 1].imshow(orig_label2)
ax[1, 1].set_title("Original label")

# 3. plot the transformed slice
ax[1, 2].imshow(transformed_slice2, cmap="grey")
ax[1, 2].set_title("Transformed slice")

# 4. plot the transformed label
ax[1, 3].imshow(transformed_label2)
ax[1, 3].set_title("Transformed label")

# third slice:
# 1. plot the original slice
ax[2, 0].imshow(orig_slice3, cmap="grey")
ax[2, 0].set_title("Original slice")

# 2. plot the original label
ax[2, 1].imshow(orig_label3)
ax[2, 1].set_title("Original label")

# 3. plot the transformed slice
ax[2, 2].imshow(transformed_slice3, cmap="grey")
ax[2, 2].set_title("Transformed slice")

# 4. plot the transformed label
ax[2, 3].imshow(transformed_label3)
ax[2, 3].set_title("Transformed label")

fig.savefig(output_dir + "originalSlice-originalLabel-transformedSlice-transformedLabel_" + type_run + "_" + str(run_id) + ".png", dpi=300)


# Visualize validation data (transformed)
fig, ax = plt.subplots(3, 2, figsize=(15, 10))

val_slice1 = val_transformed_patient['image'][0, 0, :, :, z1]
val_label1 = val_transformed_patient['label'][0, 0, :, :, z1]

val_slice2 = val_transformed_patient['image'][0, 0, :, :, z2]
val_label2 = val_transformed_patient['label'][0, 0, :, :, z2]

val_slice3 = val_transformed_patient['image'][0, 0, :, :, z3]
val_label3 = val_transformed_patient['label'][0, 0, :, :, z3]

# first slice:
# 1. plot the transformed image - validation
ax[0, 0].imshow(val_slice1, cmap="grey")
ax[0, 0].set_title("Slice - validation transform")

# 2. plot the transformed label - validation
ax[0, 1].imshow(val_label1)
ax[0, 1].set_title("Label - validation transform")

# second slice:
# 1. plot the transformed image - validation
ax[1, 0].imshow(val_slice2, cmap="grey")
ax[1, 0].set_title("Slice - validation transform")

# 2. plot the transformed label - validation
ax[1, 1].imshow(val_label2)
ax[1, 1].set_title("Label - validation transform")

# third slice:
# 1. plot the transformed image - validation
ax[2, 0].imshow(val_slice3, cmap="grey")
ax[2, 0].set_title("Slice - validation transform")

# 2. plot the transformed label - validation
ax[2, 1].imshow(val_label3)
ax[2, 1].set_title("Label - validation transform")

fig.savefig(output_dir + "valTransformedSlice-valTransformedLabel_" + type_run + "_" + str(run_id) + ".png", dpi=300)

#affine = np.eye(4)


# We are performing the following operation because during training, we need to pass each batch to GPU
# Notes: Most of the workstations have >1 GPU => we can parallelize it using multiple GPU cores. We can split tasks by GPU (e.g. do first part in GPU 1 and do the rest in GPU 2). This technique is important in job or job hunting.
device = torch.device("cuda:0")

for batch_data in train_loader:
    inputs, labels = (
        batch_data["image"].to(device),
        batch_data["label"].to(device),
    )
    print(f"Input image shape: {inputs.shape}") # should B x C X H X W X D
    print(f"Input label shape: {labels.shape}")
    break


# CREATE MODEL
# model = UNet(
#     spatial_dims=3, # 3 because we have height width and depth
#     in_channels=1,
#     out_channels=2, # binary segmentation
#     channels=(16, 32, 64, 128, 256), # this is # of convolutional kernal in each ___ block. The fact that there are 5 values tells us there are 5 encoder layers
#     strides=(2, 2, 2, 2), # => After the first encoder layer, we'll have a stride of 2. Stride means that each time it reduces size of input to the encoder layer.
#     num_res_units=2,
#     dropout=0.2, ### This would do what? ==> Increase 0.1 it to 0.3 because the widely fluctuating validation accuracy/metric hints at overfitting.
#     norm=Norm.BATCH,
# )
model = UNETR(
    img_size=(160, 160, 64), # dimension of input image
    in_channels=1, # dimension of input channels
    out_channels=2, # dimension of output channels. in multi-class, should be set to 4.
    feature_size=16, # dimension of network feature size. can set to 32 or 64 depending on GPU capacity
    hidden_size=768, ## dimension of hidden layer. ##WHY is default 768?
    mlp_dim=3072, ## dimension of the feedforward layer.  ## WHY is default 3072?
    num_heads=12, ## number of attention heads. ## what is attention heads? why is default 12?
    #pos_embed="perceptron", ## This threw an error: "TypeError: UNETR.__init__() got an unexpected keyword argument 'pos_embed'"
    norm_name="instance", ## what is "instance" norm? Is there batch norm option?
    res_block=True, ##
    dropout_rate=0.2,
)
print(model)

# Input tensors and model weights have to be on GPU
if torch.cuda.is_available():
    model = model.cuda()

# Check number of parameters
print("-------------------------")
print("Number of parameters:", sum(p.numel() for p in model.parameters()))
print("-------------------------")

# Define loss function, optimizer, and evaluation metric
print("\nDefining loss function, optimizer, and evaluation metric")
#loss_function = DiceLoss(to_onehot_y=True, softmax=True)
loss_function = DiceCELoss(to_onehot_y=True, softmax=True) # What is softmax? They call it logists. Logists are probability map. We need to change probability map to binary or smth, that's why we need softmax=True.
optimizer = torch.optim.Adam(model.parameters(), 1e-4) #1e-4
dice_metric = DiceMetric(include_background=False, reduction="mean") #include_background=False means we don't care if we can detect background (background channel of the output of the model) correctly or not. # reduction="mean" will give mean of all batches
precision_metric = ConfusionMatrixMetric(include_background=False, metric_name="precision")
recall_metric = ConfusionMatrixMetric(include_background=False, metric_name="recall")
HD_metric = HausdorffDistanceMetric(include_background=False)


# Define maximum number of epochs and validation interval
max_epochs = max_epoch # depending on the epoch we set, model training could take a while. # of epochs is the number of times we want to iterate over each data. (e.g. epoch = 100 means we iterate over each data 100 times). # AA had max_epochs=700 because he had a lot of data It took 1 week to train.
val_interval = 2 ### meaning?  # We can set it to 1, 2, 5, etc. It depends on the data size we have.

# Initialize variables for tracking best metric and associated epoch
best_metric = -1
best_metric_epoch = -1

# Initialize variables for tracking best validation loss and associated epoch
best_val_loss = 3
best_val_loss_epoch = 3

# Lists to store epoch loss values and metric values ==> to store accuracy of loss fn and metric values
epoch_loss_values = []
metric_values = []
precision_values = []
recall_values = []
val_loss_values = []
HD_values = []

# Define post-processing transforms for predictions and labels ==> similar to train .. transform but we apply it after prediction and label.
post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)]) # discretize output. Need to make to one hot vector so that we can compare with model output. channel 0: background, channel 1: foreground
post_label = Compose([AsDiscrete(to_onehot=2)])

# Define counter for early stopping
counter_early_stopping = 0

# Iterate over epochs
print("\nIterate over epochs")
for epoch in range(max_epochs): # start from 0
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")

    # Set model to training mode
    model.train() # set all the randomness of the model ... ### The randomness we have in the model is dropout.
    epoch_loss = 0
    step = 0

    # Iterate over batches in the training loader i.e. iterate over all the data in train loader

    for batch_data in train_loader: # 8 x 1 x 96 x 96 x 96 (BCHWD)
        step += 1

        # Option 2: When on GPU
        inputs = batch_data["image"].cuda()
        labels = batch_data["label"].cuda()

        # * Make the segmentation a binary segmentation problem. This step is extremely important. The code can fail if we don't include this step.
        labels[labels > 0] = 1 # This will make class 1-3 all have the value 1

        ## ** Memorize the following 5 steps.
        # Zero the gradients
        optimizer.zero_grad() # clears out ... from previous step
        # Forward pass
        outputs = model(inputs)
        # Compute loss
        loss = loss_function(outputs, labels)
        # Backward pass
        loss.backward() # c.f. tutorial 1. It computes the gradient.
        # Update weights
        optimizer.step() # gets the gradient and applies (gradient to) gradient descent
        # Update epoch loss
        epoch_loss += loss.item()
        print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")

    # Calculate average epoch loss
    epoch_loss /= step # average of all loss with__ the list
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    # Perform validation at specified intervals
    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad(): # I am in the validation mode, so I don't care about gradients.

            val_interval_loss = 0
            step_val = 0
            for val_data in val_loader:
                step_val += 1
                
                val_inputs = val_data["image"].cuda()
                val_labels = val_data["label"].cuda()

                # Make the segmentation a binary segmentation problems
                val_labels[val_labels > 0] = 1

                # Define ROI size and sliding window batch size
                roi_size = (160, 160, 64) #(160, 160, 160)
                sw_batch_size = 4
                # Perform sliding window inference.   WHY is it IMPT to do sliding window inference?: We don't want __ to speak to specific size. Without sw_inference, we'll have to get 96 x 96 for each.
                val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model) #sw: sliding window. This step crops 96 by 96 automatically from validation, applies model, etc.
                #print("past sliding window inference in validation")


                val_loss = loss_function(val_outputs, val_labels).item()
                print("val_loss at point 1:", val_loss)
                val_interval_loss += val_loss

                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)] # iterate over all the batches
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]

                # Compute dice metric, precision, and recall for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)
                precision_metric(y_pred=val_outputs, y=val_labels)
                recall_metric(y_pred=val_outputs, y=val_labels)

                # Compute HD distance
                HD_metric(y_pred=val_outputs, y=val_labels)



            # Calculate average validation interval loss
            val_interval_loss /= step_val
            val_loss_values.append(val_interval_loss)
            print(f"epoch {epoch + 1} validation loss: {val_interval_loss:.4f}")

            # Aggregate the final mean dice result
            metric = dice_metric.aggregate().item()
            precision = precision_metric.aggregate()[0].item()
            recall = recall_metric.aggregate()[0].item()
            HD_dist = HD_metric.aggregate()[0].item()

            print("metric:", metric)
            print("precision:", precision)
            print("recall:", recall)
            print("HD_dist:", HD_dist)

            # Reset the status for the next validation round
            dice_metric.reset()
            precision_metric.reset()
            recall_metric.reset()
            HD_metric.reset()

            # Append metric value to list
            metric_values.append(metric)
            precision_values.append(precision)
            recall_values.append(recall)
            HD_values.append(HD_dist)

            # Check if current metric is better than best metric
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                # Save the model with the best metric
                torch.save(model.state_dict(), os.path.join(output_dir_model, best_model_name)) # type of the model is pt or pth
                # torch.save(model.state_dict(), output_dir_model + best_model_name)
                print("saved new best metric model")

            # Print current and best metric values
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
                f"\nprecision: {precision:.4f}"
                f"\nrecall: {recall:.4f}"
            )


            # Check if validation loss hasn't improved for the number of epochs set by patience_interval 
            if val_interval_loss < best_val_loss - min_delta:
                # if there is significant improvement
                best_val_loss = val_interval_loss
                best_val_loss_epoch = epoch + 1
                counter_early_stopping = 0 # reset the counter given that there was an improvement in loss
            else:
                # if there is no significant improvement
                counter_early_stopping += 1
                print("Early stopping counter " + str(counter_early_stopping) + " out of " + str(patience_interval))
                if counter_early_stopping >= patience_interval:
                    print("Stopping early at epoch " + str(epoch + 1))
                    break

# The 16 shown below is because of batch size.
# Each of 1/16, 2/16, 3/16, .. is a step. (16 steps/epoch, it seems)

# Indicate the end of training and what the best metric was
print(f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")

# <Plot epoch average loss and validation metric>
# define plot parameters
fig, ax = plt.subplots(1, 3, figsize=(20, 5))

# 1. plot epoch average loss
ax[0].set_title("Epoch Average Loss")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
x_loss = [i + 1 for i in range(len(epoch_loss_values))]
y_loss = epoch_loss_values
ax[0].plot(x_loss, y_loss, color='blue', label='Training Loss')

# 2. plot training loss with validation loss
ax[1].set_title("Epoch Average Loss")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Loss")
x_val_loss = [val_interval * (i + 1) for i in range(len(val_loss_values))]
y_val_loss = val_loss_values
ax[1].plot(x_loss, y_loss, color='blue', label='Training Loss')
ax[1].plot(x_val_loss, y_val_loss, color='orange', label='Validation Loss')

# 2. plot validation evaluation metric (e.g. mean dice)
ax[2].set_title("Validation Mean Dice")
ax[2].set_xlabel("Epoch")
ax[2].set_ylabel("Mean Dice")
x_dice = [val_interval * (i + 1) for i in range(len(metric_values))]
y_dice = metric_values
ax[2].plot(x_dice, y_dice, color='green', label='Validation Dice')

# save the figure
fig.savefig(output_dir + "curves_train-val_" + type_run + "_" + str(run_id) + ".png", dpi=300)


# Plot precision, recall, and HD ----------
fig, ax = plt.subplots(1, 3, figsize=(20, 5))

# precision
ax[0].set_title("Validation Precision")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Precision")
x_vals = [val_interval * (i + 1) for i in range(len(precision_values))]
ax[0].plot(x_vals, precision_values, color='purple', label='Precision')

# recall
ax[1].set_title("Validation Recall")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Recall")
ax[1].plot(x_vals, recall_values, color='orange', label='Recall')

# Hausdorff distance
ax[2].set_title("Validation Hausdorff distance")
ax[2].set_xlabel("Epoch")
ax[2].set_ylabel("Hausdorff distance")
ax[2].plot(x_vals, HD_values, color='grey', label='Hausdorff distance')

fig.savefig(output_dir + "val_precision-recall-HD_" + type_run + "_" + str(run_id) + ".png", dpi=300)

# Plot a precision recall curve
### Ideally, we add area under the curve
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.set_title("Precision-Recall Curve")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_xlim([0, 1.0])
ax.set_ylim([0, 1.0])
ax.scatter(recall_values, precision_values, color="orange")
#ax.plot(recall_values, precision_values, color="orange")
fig.savefig(output_dir + "precision-recall_curve_" + type_run + "_" + str(run_id) + ".png", dpi=300)


# ===== Deploy the best saved model ===== 
# Load the best saved model parameters
model.load_state_dict(torch.load(os.path.join(output_dir_model, best_model_name)))

# Set the model to evaluation mode
model.eval()

# Perform inference on validation data using the best model
step = 0
with torch.no_grad():
    for val_data in val_loader:
        step += 1
        print("current step:", step)
        roi_size = (160, 160, 64)
        sw_batch_size = 4

        val_outputs = sliding_window_inference(val_data["image"].cuda(),
                                               roi_size,
                                               sw_batch_size,
                                               model)

        ## Make the segmentation a binary segmentation problems
        val_labels_binary = val_data["label"].cuda()
        val_labels_binary[val_labels_binary > 0] = 1
        ## print("val_labels_binary:", val_labels_binary)
        ## val_labels[val_labels > 0] = 1


        # Plot three slices ----- 
        fig, ax = plt.subplots(3, 3, figsize=(15, 15))

        # Plot the first slice
        # 1. plot the original slice
        idx_test_slice1 = 35
        ax[0, 0].set_title(f"image {step}")
        ax[0, 0].imshow(val_data["image"][0, 0, :, :, idx_test_slice1], cmap="gray")

        # 2. ground truth label
        ax[0, 1].set_title(f"label {step} - true label")
        ax[0, 1].imshow(val_data["label"][0, 0, :, :, idx_test_slice1])
        
        # 3. plot model output
        ax[0, 2].set_title(f"label {step} - predicted")
        ax[0, 2].imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, idx_test_slice1]) 

        # Plot the second slice
        # 1. plot the original slice
        idx_test_slice2 = 25
        ax[1, 0].set_title(f"image {step}")
        ax[1, 0].imshow(val_data["image"][0, 0, :, :, idx_test_slice2], cmap="gray")

        # 2. ground truth label
        ax[1, 1].set_title(f"label {step} - true label")
        ax[1, 1].imshow(val_data["label"][0, 0, :, :, idx_test_slice2])
        
        # 3. plot model output
        ax[1, 2].set_title(f"label {step} - predicted")
        ax[1, 2].imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, idx_test_slice2]) 


        # Plot the second slice
        # 1. plot the original slice
        idx_test_slice3 = 15
        ax[2, 0].set_title(f"image {step}")
        ax[2, 0].imshow(val_data["image"][0, 0, :, :, idx_test_slice3], cmap="gray")

        # 2. ground truth label
        ax[2, 1].set_title(f"label {step} - true label")
        ax[2, 1].imshow(val_data["label"][0, 0, :, :, idx_test_slice3])
        
        # 3. plot model output
        ax[2, 2].set_title(f"label {step} - predicted")
        ax[2, 2].imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, idx_test_slice3]) 

        fig.savefig(output_dir + "best-model-deployed_input-trueLabel-predictedLabel_step" + str(step) + "_" + type_run + "_" + str(run_id) + ".png", dpi=300)


        # Save the images as nifti images (.nii) -----
        val_image_np = val_data["image"][0, 0].cpu().numpy().astype(np.float64) ## val_image_np = val_data["image"][0, 0].cpu().numpy()
        val_label_np = val_data["label"][0, 0].cpu().numpy().astype(np.float64) ## val_label_np = val_data["label"][0, 0].cpu().numpy()
        val_pred_np = torch.argmax(val_outputs, dim=1).detach().cpu()[0].numpy().astype(np.float64) #torch.argmax(val_outputs, dim=1).detach().cpu()[0].numpy() # torch.argmax(val_outputs, dim=1)[0].cpu().numpy().astype(np.float64) ##torch.argmax(val_outputs, dim=1)[0].detach().cpu().numpy() # this is likely to be our prediction
        #val_pred_np_temp = val_outputs.cpu()[0].

        #print("val_image_np:", val_image_np)
        #print("val_label_np:", val_label_np)
        #print("val_pred_np:", val_pred_np)
        #print("torch.argmax(val_outputs, dim=1).detach().cpu()[0].numpy():", torch.argmax(val_outputs, dim=1).detach().cpu()[0].numpy())

        # print("torch.argmax(val_outputs, dim=1)[0].cpu().numpy():", torch.argmax(val_outputs, dim=1)[0].cpu().numpy())
        
        # check the type of the variables to be saved
        #print("type of val_image_np:", type(val_image_np))
        #print("type of val_label_np:", type(val_label_np))
        #print("type of val_pred_np:", type(val_pred_np))
        #print("type of val_pred_np_temp:", type(val_pred_np_temp))

        # save .nii files using nibabel
        nib.save(nib.Nifti1Image(val_image_np, affine=np.eye(4)), output_nii + "val_image_step" + str(step) + "_run" + str(run_id) + ".nii")
        nib.save(nib.Nifti1Image(val_label_np, affine=np.eye(4)), output_nii + "val_label_step" + str(step) + "_run" + str(run_id) + ".nii")
        nib.save(nib.Nifti1Image(val_pred_np, affine=np.eye(4)), output_nii + "val_prediction_step" + str(step) + "_run" + str(run_id) + ".nii")
        #nib.save(nib.Nifti1Image(val_pred_np_temp, affine=np.eye(4)), output_nii + "temp_val_prediction_step" + str(step) + "_run" + str(run_id) + ".nii")

        # Break loop after 3 iterations for visualization
        if step == 3: # i == 2
            break


# ===== Model performance on test set =====
##Define the list of test images
## originally we do this on test set, but here for this dataset we don't have labels for test set because it's a challenge dataset
## so here the mean dice score
## Create test data dictionary

test_data = test_files ## test_data = val_files

# Define original test transforms
test_org_transforms = val_transforms

# Create original test dataset
test_org_ds = Dataset(data=test_data, transform=test_org_transforms)

# Create data loader for original test dataset
test_org_loader = DataLoader(test_org_ds, batch_size=b_size, num_workers=4)

# Define post-processing transforms
post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
post_label = Compose([AsDiscrete(to_onehot=2)])

# Load the best saved model parameters
model.load_state_dict(torch.load(os.path.join(output_dir_model, best_model_name)))
model.eval()

# Evaluate model performance on test set
step_test = 0
with torch.no_grad():
    for test_data in test_org_loader:
        step_test += 1

        test_inputs = test_data["image"].cuda()
        test_labels = test_data["label"].cuda()

        # Make the segmentation a binary segmentation problems
        test_labels[test_labels > 0] = 1

        roi_size = (160, 160, 64)
        sw_batch_size = 4
        # Perform sliding window inference
        test_outputs = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model)
        test_outputs = [post_pred(i) for i in decollate_batch(test_outputs)]
        test_labels = [post_label(i) for i in decollate_batch(test_labels)]
        # Compute metrics for current iteration
        dice_metric(y_pred=test_outputs, y=test_labels)
        precision_metric(y_pred=test_outputs, y=test_labels)
        recall_metric(y_pred=test_outputs, y=test_labels)
        HD_metric(y_pred=val_outputs, y=val_labels)


    # Aggregate the final mean dice result
    metric_org = dice_metric.aggregate().item()
    precision_org = precision_metric.aggregate()[0].item()
    recall_org = recall_metric.aggregate()[0].item()
    HD_dist_org = HD_metric.aggregate()[0].item()


    # Reset the status for the next validation round
    dice_metric.reset()
    precision_metric.reset()
    recall_metric.reset()
    HD_metric.reset()

# Print the metric, precision, and recall on the test set
print("Metric on Test Set: ", metric_org) # as we used validation data now should be the same as best score
print("Precision on Test Set :", precision_org)
print("Recall on Test Set:", recall_org)
print("Hausdorff distance on Test Set:", HD_dist_org)


# Perform inference on test data using the best model
step_test = 0
with torch.no_grad():
    for test_data in test_org_loader:
        step_test += 1
        print("current step:", step)
        roi_size = (160, 160, 64)
        sw_batch_size = 4

        test_outputs = sliding_window_inference(test_data["image"].cuda(),
                                        roi_size,
                                        sw_batch_size,
                                        model)


        # Save test image, label, and predicted label in nifty (.nii) format
        test_image_np = test_data["image"][0, 0].cpu().numpy()
        test_label_np = test_data["label"][0, 0].cpu().numpy()
        test_pred_np = torch.argmax(test_outputs, dim=1).detach().cpu()[0].numpy().astype(np.float64)

        #print("test_image_np:", test_image_np)
        #print("test_label_np:", test_label_np)
        #print("test_pred_np:", test_pred_np)

        #print("type of test_image_np:", type(test_image_np))
        #print("type of test_label_np:", type(test_label_np))
        #print("type of test_pred_np:", type(test_pred_np))

        # save .nii files using nibabel
        nib.save(nib.Nifti1Image(test_image_np, affine=np.eye(4)), output_nii + "test_image_step" + str(step_test) + "_run" + str(run_id) + ".nii")
        nib.save(nib.Nifti1Image(test_label_np, affine=np.eye(4)), output_nii + "test_label_step" + str(step_test) + "_run" + str(run_id) + ".nii")
        nib.save(nib.Nifti1Image(test_pred_np, affine=np.eye(4)), output_nii + "test_prediction_step" + str(step_test) + "_run" + str(run_id) + ".nii")

        # break loop after a certain number of iterations (e.g. 5)
        if step_test == 3:
            break
