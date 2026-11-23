import cv2, numpy as np
from app.vision.body_embedder import BodyEmbedder

be = BodyEmbedder()

p_in = "static/uploads/visitors/VIS-2026-1043_1795417949.jpg"
im = cv2.imread(p_in)
un = be.unpad_canvas(im)

print("Original shape:", im.shape)
print("Unpadded shape:", un.shape if un is not None else None)

# Let's save unpadded image and torso
cv2.imwrite("scratch/1043_latest_entry_raw.jpg", im)
cv2.imwrite("scratch/1043_latest_entry_unpadded.jpg", un)

h, w = un.shape[:2]
ty1 = int(h * 0.18)
ty2 = int(h * 0.50)
tx1 = int(w * 0.22)
tx2 = int(w * 0.78)
torso = un[ty1:ty2, tx1:tx2]
cv2.imwrite("scratch/1043_latest_entry_torso.jpg", torso)

print(f"Torso slice coords: y=[{ty1}:{ty2}], x=[{tx1}:{tx2}]")
print("Torso BGR mean:", np.mean(torso, axis=(0,1)))
hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
print("Torso HSV mean:", np.mean(hsv, axis=(0,1)))
s_chan = hsv[:,:,1].flatten()
v_chan = hsv[:,:,2].flatten()
chroma_mask = (s_chan >= 30) & (v_chan >= 40)
print("Chroma pct:", np.sum(chroma_mask)/len(s_chan))
print("Mean S:", np.mean(s_chan), "Mean V:", np.mean(v_chan))
