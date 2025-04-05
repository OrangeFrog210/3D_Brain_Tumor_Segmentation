# =========================================
# This script is for UNet T1
run_id = "01" 

output_dir = "output/Unet/binary/t1/"
output_dir_model = "model/Unet/binary/t1/"
output_nii = "output/nii_output/Unet/binary/t1/run" + str(run_id) + "/"

type_run = "Unet_binary_t1"
best_model_name = "best_metric_model_" + type_run + "_run" + str(run_id) + ".pth"
print("Name of the best model:", best_model_name)

# specify batch size
b_size = 8 

# set variables for early stopping
patience_interval = 25 # this corresponds to 50 epochs because validation interval is 2
min_delta = 0.01 

# set the number of epochs
max_epoch = 300

#. specify the number of images
n_images = 369 
n_images_val_test = 55 
# =========================================

print("Python script entered.")

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

# Specify the path to the data
dataset_path = "/cluster/projects/gaitigroup/Users/Yumika/class/MBP1413H/final_pj/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/"
print(dataset_path)

# Set up data directory
directory = os.environ.get("BraTS2020")

# ===== Split dataset to training and testing =====
# Create empty lists for each set
train_files = []
val_files = []
test_files = []


# ===== Create a data dictionary of training and validation images=====
# Create a data dictionary

# create a list of paths to images and labels, respectively
train_test_images = [f"{dataset_path}BraTS20_Training_{volume_number:03d}/BraTS20_Training_{volume_number:03d}_t1.nii" for volume_number in range(1, n_images+1)] # train_images = [f"{dataset_path}BraTS20_Training_{volume_number:03d}/BraTS20_Training_{volume_number:03d}_flair.nii" for volume_number in range(1, n_images+1)]
train_test_labels = [f"{dataset_path}BraTS20_Training_{volume_number:03d}/BraTS20_Training_{volume_number:03d}_seg.nii" for volume_number in range(1, n_images+1)] # train_labels = [f"{dataset_path}BraTS20_Training_{volume_number:03d}/BraTS20_Training_{volume_number:03d}_seg.nii" for volume_number in range(1, n_images+1)]

# create a data dictionary
data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_test_images, train_test_labels)]

# split the data dictionary into training, validation, and testing
train_val_files, test_files = data_dicts[:-n_images_val_test], data_dicts[-n_images_val_test:] 
train_files, val_files = train_val_files[:-n_images_val_test], train_val_files[-n_images_val_test:]

print("Length of train_files:", len(train_files))
print("Length of test_files (originall val_files):", len(test_files))
print("Length of val_files:", len(val_files))

# plot a bar plot showing the number of images in each of training, validation, and testing
plt.figure(figsize=(8, 6))
plt.bar(['Train', 'Validation','Test'], [len(train_files), len(val_files),len(test_files)])
plt.title('Number of Volumes in Each Set')
plt.ylabel('Number of Volums')
plt.xlabel('Dataset Split')
plt.savefig(output_dir + 'n_images_train-val-test_' + type_run + '_' + str(run_id)  + '.png', dpi=300)


# ===== Set deterministic training for reproducibility ===== 
set_determinism(seed=0)


# ===== Set up data loader and augmentation transforms ===== 
# Step 1: Load the images
# Step 2: Do any transforms
# Step 3: Convert transform-applied data to torch tensors in order to do training using pytorch

orig_transforms = Compose(
    [
        # load images
        LoadImaged(keys=['image', 'label']),

        EnsureChannelFirstd(keys=['image', 'label']),

        # convert to torch tensors
        ToTensord(keys=['image', 'label'])
    ]
)



train_transforms = Compose(
    [
        # load images
        LoadImaged(keys=['image', 'label']),

        # add a new dimension that controls how many batches we have using the following class
        EnsureChannelFirstd(keys=['image', 'label']),

        # Change the dimension.
        Spacingd(keys=['image', 'label'], pixdim=(1.5, 1.5, 2)),

        # Scale the intensity of the image. Apply it only to the image, not label.
        ScaleIntensityRanged(
            keys='image',
            a_min=-200, 
            a_max=200, 
            b_min=0.0, 
            b_max=1.0
        ),

        # Crop foreground from both the image and label based on the image
        CropForegroundd(keys=['image', 'label'], source_key='image'),

        # Resize the image and the label
        Resized(keys=['image', 'label'], spatial_size=(160, 160, 64)), 

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

        EnsureChannelFirstd(keys=['image', 'label']),

        # Change the dimension. Make sure it's applied to both image and label, not just image.
        Spacingd(keys=['image', 'label'], pixdim=(1.5, 1.5, 2)), 

        # Scale the intensity of the image. Apply it only to the image, not label.
        ScaleIntensityRanged(
            keys='image',
            a_min=-200, 
            a_max=200, 
            b_min=0.0, 
            b_max=1.0
        ),

        # Crop foreground from both the image and label based on the image
        CropForegroundd(keys=['image', 'label'], source_key='image'),

        # Resize the image and the label
        Resized(keys=['image', 'label'], spatial_size=(160, 160, 64)),

        # Normalize intensity of the image
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),

        # Convert to torch tensors
        ToTensord(keys=['image', 'label'])
    ]
)


# Data loader
print("Data loader: original images------------------")
orig_ds = CacheDataset(data=train_files, transform=orig_transforms, cache_rate=1.0, num_workers=8) 
orig_loader = DataLoader(orig_ds, batch_size=b_size, shuffle=True, num_workers=4)

print("Data loader: train images----------------------")
train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=8)
train_loader = DataLoader(train_ds, batch_size=b_size, shuffle=True, num_workers=4)

print("Data loader: validation images-----------------")
val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=8)
val_loader = DataLoader(val_ds, batch_size=b_size, shuffle=True, num_workers=4)

print("\nData loaded")

orig_patient = first(orig_loader) 
train_transformed_patient = first(train_loader) 
val_transformed_patient = first(val_loader)

print(torch.min(train_transformed_patient['image']))
print(torch.max(train_transformed_patient['image']))


# Visualize transformation
print("\nVisualizing transformation")

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


device = torch.device("cuda:0")

for batch_data in train_loader:
    inputs, labels = (
        batch_data["image"].to(device),
        batch_data["label"].to(device),
    )
    print(f"Input image shape: {inputs.shape}") 
    print(f"Input label shape: {labels.shape}")
    break


# CREATE MODEL
model = UNet(
    spatial_dims=3, 
    in_channels=1,
    out_channels=2, 
    channels=(16, 32, 64, 128, 256), 
    strides=(2, 2, 2, 2),
    num_res_units=2,
    dropout=0.2, 
    norm=Norm.BATCH,
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
loss_function = DiceCELoss(to_onehot_y=True, softmax=True) 
optimizer = torch.optim.Adam(model.parameters(), 1e-4) 
dice_metric = DiceMetric(include_background=False, reduction="mean") 
precision_metric = ConfusionMatrixMetric(include_background=False, metric_name="precision")
recall_metric = ConfusionMatrixMetric(include_background=False, metric_name="recall")
HD_metric = HausdorffDistanceMetric(include_background=False)


# Define maximum number of epochs and validation interval
max_epochs = max_epoch 
val_interval = 2 

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

# Define post-processing transforms for predictions and labels
post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)]) 
post_label = Compose([AsDiscrete(to_onehot=2)])

# Define counter for early stopping
counter_early_stopping = 0

# Iterate over epochs
print("\nIterate over epochs")
for epoch in range(max_epochs): 
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")

    # Set model to training mode
    model.train()
    epoch_loss = 0
    step = 0

    # Iterate over batches in the training loader i.e. iterate over all the data in train loader
    for batch_data in train_loader: 
        step += 1

        # Option 2: When on GPU
        inputs = batch_data["image"].cuda()
        labels = batch_data["label"].cuda()

        # * Make the segmentation a binary segmentation problem. 
        labels[labels > 0] = 1 

        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)
        # Compute loss
        loss = loss_function(outputs, labels)
        # Backward pass
        loss.backward() 
        # Update weights
        optimizer.step() 
        # Update epoch loss
        epoch_loss += loss.item()
        print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")

    # Calculate average epoch loss
    epoch_loss /= step 
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    # Perform validation at specified intervals
    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():

            val_interval_loss = 0
            step_val = 0
            for val_data in val_loader:
                step_val += 1

                val_inputs = val_data["image"].cuda()
                val_labels = val_data["label"].cuda()

                # Make the segmentation a binary segmentation problems
                val_labels[val_labels > 0] = 1

                # Define ROI size and sliding window batch size
                roi_size = (160, 160, 64) 
                sw_batch_size = 4
                # Perform sliding window inference.
                val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model) 
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
                torch.save(model.state_dict(), os.path.join(output_dir_model, best_model_name)) 
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
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.set_title("Precision-Recall Curve")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_xlim([0, 1.0])
ax.set_ylim([0, 1.0])
ax.scatter(recall_values, precision_values, color="orange")
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

        val_labels_binary = val_data["label"].cuda()
        val_labels_binary[val_labels_binary > 0] = 1


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
        val_image_np = val_data["image"][0, 0].cpu().numpy().astype(np.float64) 
        val_label_np = val_data["label"][0, 0].cpu().numpy().astype(np.float64) 
        val_pred_np = torch.argmax(val_outputs, dim=1).detach().cpu()[0].numpy().astype(np.float64) 

        # save .nii files using nibabel
        nib.save(nib.Nifti1Image(val_image_np, affine=np.eye(4)), output_nii + "val_image_step" + str(step) + "_run" + str(run_id) + ".nii")
        nib.save(nib.Nifti1Image(val_label_np, affine=np.eye(4)), output_nii + "val_label_step" + str(step) + "_run" + str(run_id) + ".nii")
        nib.save(nib.Nifti1Image(val_pred_np, affine=np.eye(4)), output_nii + "val_prediction_step" + str(step) + "_run" + str(run_id) + ".nii")

        # Break loop after 3 iterations for visualization
        if step == 3: 
            break


# ===== Model performance on test set =====

test_data = test_files

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
print("Metric on Test Set: ", metric_org) 
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

        # save .nii files using nibabel
        nib.save(nib.Nifti1Image(test_image_np, affine=np.eye(4)), output_nii + "test_image_step" + str(step_test) + "_run" + str(run_id) + ".nii")
        nib.save(nib.Nifti1Image(test_label_np, affine=np.eye(4)), output_nii + "test_label_step" + str(step_test) + "_run" + str(run_id) + ".nii")
        nib.save(nib.Nifti1Image(test_pred_np, affine=np.eye(4)), output_nii + "test_prediction_step" + str(step_test) + "_run" + str(run_id) + ".nii")

        # break loop after a certain number of iterations (e.g. 5)
        if step_test == 3:
            break
            
