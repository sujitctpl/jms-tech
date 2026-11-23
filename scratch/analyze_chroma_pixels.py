import cv2, numpy as np

im = cv2.imread("scratch/1043_latest_entry_unpadded.jpg")
h, w = im.shape[:2]

# Let's inspect the upper body region: y from 0.15*h to 0.50*h
upper = im[int(h*0.15):int(h*0.50), :]
hsv = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)

# In this upper region, what are the different colors?
# Let's look at pixels with saturation:
s = hsv[:, :, 1]
v = hsv[:, :, 2]
hue = hsv[:, :, 0]

print("Total pixels in upper body band:", s.size)
print("Pixels with S > 30:", np.sum(s > 30), f"({np.sum(s > 30)/s.size*100:.1f}%)")
print("Pixels with S > 25:", np.sum(s > 25), f"({np.sum(s > 25)/s.size*100:.1f}%)")
print("Pixels with S > 20:", np.sum(s > 20), f"({np.sum(s > 20)/s.size*100:.1f}%)")

# What are the hues of pixels with S > 20?
mask = (s > 20) & (v > 40)
print("Hues of S > 20 pixels:")
print("  Red/Maroon (H < 12 or H >= 165):", np.sum((hue[mask] < 12) | (hue[mask] >= 165)))
print("  Orange (12 <= H < 22):", np.sum((hue[mask] >= 12) & (hue[mask] < 22)))
print("  Yellow (22 <= H < 38):", np.sum((hue[mask] >= 22) & (hue[mask] < 38)))
print("  Green (38 <= H < 85):", np.sum((hue[mask] >= 38) & (hue[mask] < 85)))
print("  Blue (85 <= H < 135):", np.sum((hue[mask] >= 85) & (hue[mask] < 135)))

# Now let's do the SAME on the Exit image (man in white shirt)!
im_out = cv2.imread("static/uploads/visitors/VIS-2026-1043_exit_1795418250.jpg")
from app.vision.body_embedder import BodyEmbedder
be = BodyEmbedder()
un_out = be.unpad_canvas(im_out)
h_o, w_o = un_out.shape[:2]
upper_out = un_out[int(h_o*0.15):int(h_o*0.50), :]
hsv_o = cv2.cvtColor(upper_out, cv2.COLOR_BGR2HSV)
s_o = hsv_o[:, :, 1]
v_o = hsv_o[:, :, 2]
hue_o = hsv_o[:, :, 0]
mask_o = (s_o > 20) & (v_o > 40)

print("\n--- Exit Image (White shirt) ---")
print("Total pixels in upper body band:", s_o.size)
print("Pixels with S > 20:", np.sum(s_o > 20), f"({np.sum(s_o > 20)/s_o.size*100:.1f}%)")
print("Red pixels in exit image:", np.sum((hue_o[mask_o] < 12) | (hue_o[mask_o] >= 165)))
