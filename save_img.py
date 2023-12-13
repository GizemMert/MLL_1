import cv2
import os
import numpy as np
from Dataloader import Dataloader, label_map
from torch.utils.data import DataLoader


def save_class_images(dataloader, classes_of_interest, save_directory):
    # Dictionary to track whether an image for each class has been saved
    saved_classes = {class_label: False for class_label in classes_of_interest}

    for _, roi_cropped, label_tensor, _ in dataloader:
        label = label_tensor.item()  # Convert label tensor to Python integer
        if label in classes_of_interest and not saved_classes[label]:
            # Convert from CHW to HWC format
            roi_cropped_hwc = np.transpose(roi_cropped, (1, 2, 0))

            # Constructing the filename with label information
            label_name = [name for name, number in label_map.items() if number == label][0]
            filename = os.path.join(save_directory, f"class_{label_name}_image.png")

            # Save the roi_cropped image
            cv2.imwrite(filename, roi_cropped_hwc * 255)  # Rescale back to 0-255 range
            saved_classes[label] = True

            # Check if all classes are saved
            if all(saved_classes.values()):
                break

# Usage
train_dataset = Dataloader(split='train')
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

classes_of_interest = [label_map['myeloblast'], label_map['basophil'], label_map['eosinophil'],
                       label_map['monocyte'], label_map['neutrophil_banded'], label_map['neutrophil_segmented']]

current_directory = os.path.dirname(os.path.realpath(__file__))
save_class_images(train_dataloader, classes_of_interest, current_directory)


