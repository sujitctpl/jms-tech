import cv2, numpy as np

im = cv2.imread("scratch/1043_latest_entry_unpadded.jpg")

# The person's torso
# Person box: [90, 6, 348, 260]
# Let's crop tight chest:
x1, y1, x2, y2 = 90, 6, 348, 260
pw = x2 - x1
ph = y2 - y1

# Torso in upper body of person:
chest = im[y1 + int(ph * 0.16) : y1 + int(ph * 0.50), x1 + int(pw * 0.15) : x1 + int(pw * 0.65)]
cv2.imwrite("scratch/1043_person_chest.jpg", chest)

hsv = cv2.cvtColor(chest, cv2.COLOR_BGR2HSV)
print("Chest BGR mean:", np.mean(chest, axis=(0,1)))
print("Chest HSV mean:", np.mean(hsv, axis=(0,1)))
s_chan = hsv[:,:,1].flatten()
v_chan = hsv[:,:,2].flatten()
h_chan = hsv[:,:,0].flatten()

chroma_mask = (s_chan >= 25) & (v_chan >= 35)
print("Chroma pct:", np.sum(chroma_mask)/len(s_chan))
print("Median H of chroma:", np.median(h_chan[chroma_mask]) if np.sum(chroma_mask)>0 else None)
