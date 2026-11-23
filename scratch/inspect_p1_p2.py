import cv2, numpy as np

im_in = cv2.imread("static/uploads/visitors/VIS-2026-1043_1795417949.jpg")
im_out = cv2.imread("static/uploads/visitors/VIS-2026-1043_exit_1795418250.jpg")

from app.vision.body_embedder import BodyEmbedder
be = BodyEmbedder()
c1 = be.unpad_canvas(im_in)
c2 = be.unpad_canvas(im_out)

h1, w1 = c1.shape[:2]
h2, w2 = c2.shape[:2]

p1 = c1[int(h1 * 0.18):int(h1 * 0.50), int(w1 * 0.15):int(w1 * 0.85)]
p2 = c2[int(h2 * 0.18):int(h2 * 0.50), int(w2 * 0.15):int(w2 * 0.85)]

cv2.imwrite("scratch/p1_torso.jpg", p1)
cv2.imwrite("scratch/p2_torso.jpg", p2)

# Check colors in p1 and p2:
print("p1 shape:", p1.shape)
print("p2 shape:", p2.shape)
print("p1 mean BGR:", np.mean(p1, axis=(0,1)))
print("p2 mean BGR:", np.mean(p2, axis=(0,1)))
