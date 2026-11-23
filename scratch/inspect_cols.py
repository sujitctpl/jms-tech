import cv2, numpy as np

im = cv2.imread("scratch/1043_latest_entry_unpadded.jpg")
h, w = im.shape[:2]

# In this unpadded crop, let's see where the person is!
# We can find the person using:
# 1. Edge/gradient or background subtraction
# 2. Or let's inspect the columns of the image:
# The white wall on the right has very low variation or very high brightness!
# Let's inspect column by column luminance and saturation
col_means_v = np.mean(im[:, :, 2], axis=0)
col_means_s = np.mean(im[:, :, 1], axis=0)
col_means_bgr = np.mean(im, axis=0)

for x in range(0, w, 20):
    print(f"x={x:3d}..{x+20:3d}: BGR={col_means_bgr[x].astype(int)}, S={col_means_s[x]:.1f}, V={col_means_v[x]:.1f}")
