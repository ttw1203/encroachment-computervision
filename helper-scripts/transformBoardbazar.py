import cv2
import numpy as np

# ------- global setting (affects every subsequent print) -------
np.set_printoptions(precision=4,     # decimals shown
                    suppress=True,   # no scientific notation
                    floatmode='fixed')

dx, dy = 0.0643, 99.7644      # ← + sign because we add to every point
T = np.array([[1, 0, dx],
              [0, 1, dy],
              [0, 0, 1 ]], dtype=np.float32)

# pixel corners of the calibration rectangle in the video
SRC = np.float32([[1399, 708], [2238, 696], [2573,1084], [1132,1105]])

# the same four corners in world units (metres)
TGT = np.float32([[0,0],
                  [12,0],          # W = real-world width in m
                  [12,22],          # H = real-world length in m
                  [0,22]])

H = cv2.getPerspectiveTransform(SRC, TGT)    # 3×3 homography

H_shifted = T @ H                 # matrix product; order matters



pts_pix  = np.float32([[1644, 346],[1932, 342],[3502, 2160],[420, 2160]])
pts_m    = cv2.perspectiveTransform(pts_pix.reshape(-1,1,2), H) \
             .reshape(-1,2)        # → metres

print(pts_m)

pts_m_new = cv2.perspectiveTransform(
               pts_pix.reshape(-1,1,2), H_shifted).reshape(-1,2)

print(pts_m_new)

x1, y1 = 12.0498, 0.0387
s      = -y1 / x1            # shear factor  ≈ 0.1058

S = np.array([[1, 0, 0],     # Y' = Y + s·X
              [s, 1, 0],
              [0, 0, 1]], dtype=np.float32)

H_final = S @ H_shifted       # compose once

pts_m   = cv2.perspectiveTransform(pts_pix.reshape(-1,1,2), H_final)\
             .reshape(-1,2)
print(pts_m)
# -> [-0.0000  0.0000]
#    [50.0137  0.0000]      # flattened
# first row should print [ 0.0000  0.0000]

