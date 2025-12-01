import cv2, yaml, numpy as np, glob, os

# Load calibration
with open(os.path.expanduser('~/stereo_calib_results/left.yaml')) as f: left = yaml.safe_load(f)
with open(os.path.expanduser('~/stereo_calib_results/right.yaml')) as f: right = yaml.safe_load(f)
with open(os.path.expanduser('~/stereo_calib_results/stereo.yaml')) as f: stereo = yaml.safe_load(f)

K_L, D_L = np.array(left['K']), np.array(left['D'])   # left cam intrinsic + distortion
K_R, D_R = np.array(right['K']), np.array(right['D']) # right cam intrinsic + distortion
R, T = np.array(stereo['R']), np.array(stereo['T'])   # rotation + translation between cams

# resolution of live stereo capture
img_size = (640, 480)

# calculate rectification transforms
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K_L, D_L, K_R, D_R, img_size, R, T, alpha=0)

# generate undistortion + rectification maps
mapLx, mapLy = cv2.initUndistortRectifyMap(K_L, D_L, R1, P1, img_size, cv2.CV_32FC1)
mapRx, mapRy = cv2.initUndistortRectifyMap(K_R, D_R, R2, P2, img_size, cv2.CV_32FC1)

# open both camera streams
capL = cv2.VideoCapture(2)
capR = cv2.VideoCapture(4)

while True:
    retL, imgL = capL.read()
    retR, imgR = capR.read()
    if not (retL and retR): break

    # apply rectification (fix distortion & align images)
    imgL = cv2.remap(imgL, mapLx, mapLy, cv2.INTER_LINEAR)
    imgR = cv2.remap(imgR, mapRx, mapRy, cv2.INTER_LINEAR)

    # show both side-by-side
    both = cv2.hconcat([imgL, imgR])

    # horizontal lines to visually check alignment
    for y in range(0, imgL.shape[0], 40):
        cv2.line(both, (0,y), (both.shape[1],y), (0,255,0), 1)

    cv2.imshow("Rectified (Left | Right)", both)
    if cv2.waitKey(1) == 27: break  # stop on ESC

capL.release()
capR.release()
cv2.destroyAllWindows()
