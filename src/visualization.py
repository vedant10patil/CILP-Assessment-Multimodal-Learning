import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_predictions(model, loader, device, save_path="results.png"):

    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(12, 8))
    
    classes = ["Cube", "Sphere"]

    with torch.no_grad():
        for rgb, lidar, labels in loader:
            rgb, lidar, labels = rgb.to(device), lidar.to(device), labels.to(device)
            outputs = model(rgb, lidar)
            _, preds = torch.max(outputs, 1)

            for j in range(rgb.size(0)):
                images_so_far += 1
                ax = plt.subplot(3, 4, images_so_far)
                ax.axis('off')

                img_display = rgb[j].cpu().numpy()[:3, :, :] 
                img_display = np.transpose(img_display, (1, 2, 0))

                img_display = np.clip(img_display, 0, 1)

                label_true = classes[labels[j]]
                label_pred = classes[preds[j]]
                
                color = 'green' if label_true == label_pred else 'red'
                
                ax.set_title(f"True: {label_true}\nPred: {label_pred}", color=color)
                ax.imshow(img_display)

                if images_so_far == 12:
                    plt.tight_layout()
                    plt.savefig(save_path)
                    print(f"Saved visualization to {save_path}")
                    return