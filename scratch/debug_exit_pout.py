import cv2, numpy as np
from app.vision.body_embedder import BodyEmbedder

be = BodyEmbedder()
p_out = "static/uploads/visitors/VIS-2026-1043_exit_1795418250.jpg"
c = be.unpad_canvas(cv2.imread(p_out))
h, w = c.shape[:2]
torso = c[int(h * 0.18):int(h * 0.50), int(w * 0.15):int(w * 0.85)]
hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
h_chan = hsv[:, :, 0].flatten()
s_chan = hsv[:, :, 1].flatten()
v_chan = hsv[:, :, 2].flatten()

chroma_mask = (s_chan >= 24) & (v_chan >= 35)
chroma_count = int(np.sum(chroma_mask))
chroma_pct = float(chroma_count) / float(len(h_chan))
print(f"p_out len={len(h_chan)}, chroma_count={chroma_count}, chroma_pct={chroma_pct:.3f}")
h_chroma = h_chan[chroma_mask]
red_count = np.sum((h_chroma < 14) | (h_chroma >= 162))
print(f"p_out red_count={red_count}")
print("Mean S:", np.mean(s_chan), "Mean V:", np.mean(v_chan))
