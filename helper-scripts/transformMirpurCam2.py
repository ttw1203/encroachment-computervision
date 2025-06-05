import cv2
import numpy as np

# ------- global setting (affects every subsequent print) -------
np.set_printoptions(precision=4,     # decimals shown
                    suppress=True,   # no scientific notation
                    floatmode='fixed')

dx, dy = -0.0020, 25.7903      # ← + sign because we add to every point
T = np.array([[1, 0, dx],
              [0, 1, dy],
              [0, 0, 1 ]], dtype=np.float32)

# pixel corners of the calibration rectangle in the video
SRC = np.float32([[1491, 919], [2564, 919], [4554,2160], [-190,2160]])

# the same four corners in world units (metres)
TGT = np.float32([[0,0],
                  [19,0],          # W = real-world width in m
                  [19,32.5],          # H = real-world length in m
                  [0,32.5]])

H = cv2.getPerspectiveTransform(SRC, TGT)    # 3×3 homography

H_shifted = T @ H                 # matrix product; order matters



pts_pix  = np.float32([[1678, 781],[2343, 781],[4554, 2160],[-190, 2160]])
pts_m    = cv2.perspectiveTransform(pts_pix.reshape(-1,1,2), H) \
             .reshape(-1,2)        # → metres

print(pts_m)

pts_m_new = cv2.perspectiveTransform(
               pts_pix.reshape(-1,1,2), H_shifted).reshape(-1,2)

print(pts_m_new)


x1, y1 = 19.0063, 0
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

