import cv2
import numpy as np
from data_collection import gestures


def visualize_prediction(image, prediction):
    y_offset = 30
    font_scale = 0.7
    font_thickness = 2
    text_color = (0, 0, 0)

    # Get the top three predictions
    top_3_indices = np.argpartition(prediction, -3)[-3:]
    top_3_indices = top_3_indices[np.argsort(prediction[top_3_indices])][::-1]
    top_3_predictions = prediction[top_3_indices]

    # Normalize the prediction values
    top_3_predictions = top_3_predictions / np.sum(top_3_predictions)

    # Overlay the prediction percentages on top of the image
    for i, pred in enumerate(top_3_predictions):
        if pred > 0.3:
            # Draw the label
            label = f'{gestures[top_3_indices[i]]}: {pred * 100:.2f}%'
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            label_x = 10
            label_y = y_offset + i * label_size[1]
            cv2.putText(image, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color,
                        font_thickness, cv2.LINE_AA)
    return image
