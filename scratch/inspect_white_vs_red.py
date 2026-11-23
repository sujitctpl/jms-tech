import cv2, numpy as np
from app.vision.body_embedder import BodyEmbedder

be = BodyEmbedder()

p_out = "static/uploads/visitors/VIS-2026-1043_exit_1795418250.jpg"
im_out = cv2.imread(p_out)
un_out = be.unpad_canvas(im_out)

h, w = un_out.shape[:2]
torso_out = un_out[int(h * 0.18):int(h * 0.50), int(w * 0.22):int(w * 0.78)]
hsv_out = cv2.cvtColor(torso_out, cv2.COLOR_BGR2HSV)

print("Exit Torso HSV mean:", np.mean(hsv_out, axis=(0,1)))
s_out = hsv_out[:,:,1].flatten()
v_out = hsv_out[:,:,2].flatten()
print("Exit Mean S:", np.mean(s_out), "Exit Mean V:", np.mean(v_out))
print("Exit V 25th percentile:", np.percentile(v_out, 25), "75th percentile:", np.percentile(v_out, 75))

im_in = cv2.imread("static/uploads/visitors/VIS-2026-1043_1795417949.jpg")
un_in = be.unpad_canvas(im_in)
h_in, w_in = un_in.shape[:2]
torso_in = un_in[int(h_in * 0.18):int(h_in * 0.50), int(w_in * 0.22):int(w_in * 0.78)]
hsv_in = cv2.cvtColor(torso_in, cv2.COLOR_BGR2HSV)
v_in = hsv_in[:,:,2].flatten()
print("Entry V 25th percentile:", np.percentile(v_in, 25), "75th percentile:", np.percentile(v_in, 75))
