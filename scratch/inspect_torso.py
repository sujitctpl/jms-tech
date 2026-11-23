import cv2
import numpy as np

e_img = cv2.imread("scratch/1046_Entry_(Man_Maroon).jpg")
x_img = cv2.imread("scratch/1046_Exit_(Woman_Brown).jpg")

print(f"e_img shape: {e_img.shape}")
print(f"x_img shape: {x_img.shape}")

# Let's save slices of torso to see what was actually extracted
eh, ew = e_img.shape[:2]
xh, xw = x_img.shape[:2]

e_torso = e_img[int(eh * 0.16):int(eh * 0.48), int(ew * 0.22):int(ew * 0.78)]
x_torso = x_img[int(xh * 0.16):int(xh * 0.48), int(xw * 0.22):int(xw * 0.78)]

cv2.imwrite("scratch/1046_e_torso.jpg", e_torso)
cv2.imwrite("scratch/1046_x_torso.jpg", x_torso)

# Check average colors (BGR and HSV)
print("e_torso BGR mean:", np.mean(e_torso, axis=(0,1)))
print("x_torso BGR mean:", np.mean(x_torso, axis=(0,1)))

hsv_e = cv2.cvtColor(e_torso, cv2.COLOR_BGR2HSV)
hsv_x = cv2.cvtColor(x_torso, cv2.COLOR_BGR2HSV)
print("e_torso HSV mean:", np.mean(hsv_e, axis=(0,1)))
print("x_torso HSV mean:", np.mean(hsv_x, axis=(0,1)))
