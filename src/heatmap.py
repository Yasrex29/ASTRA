import cv2
import numpy as np

def generate_heatmap(frame, centers):

    heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

    for (x,y) in centers:
        cv2.circle(heatmap,(x,y),40,1,-1)

    heatmap = cv2.GaussianBlur(heatmap,(51,51),0)

    heatmap_norm = cv2.normalize(heatmap,None,0,255,cv2.NORM_MINMAX)

    heatmap_color = cv2.applyColorMap(
        heatmap_norm.astype(np.uint8),
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(frame,0.6,heatmap_color,0.4,0)

    return overlay
