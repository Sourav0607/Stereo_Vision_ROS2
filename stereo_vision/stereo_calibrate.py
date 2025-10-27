import cv2
import numpy as np
import glob
import yaml
import os

# === Calibration parameters ===
CHECKERBOARD = (8, 6)       # number of inner corners in chessboard pattern (columns x rows)
SQUARE_SIZE = 30.0          # physical size of each square in mm

# === Image paths ===
# load all calibration images from left & right folders
left_path = os.path.expanduser('~/stereo_calib_images_auto/left/*.png')
right_path = os.path.expanduser('~/stereo_calib_images_auto/right/*.png')
left_images = sorted(glob.glob(left_path))
right_images = sorted(glob.glob(right_path))

# make sure we have images otherwise throw an error
if len(left_images) == 0 or len(right_images) == 0:
    raise Exception("No calibration images found. Check your folder paths!")

# === Prepare object points ===
# this represents 3D real-world coordinates for chessboard corners
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE  # scale the points based on real square size

# === Storage for detected 2D-3D points ===
objpoints = []         # actual world coordinates
imgpoints_left = []    # detected pixel corners in left cam
imgpoints_right = []   # detected pixel corners in right cam

# corner refinement criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

print(" Detecting checkerboard corners...")

# === Loop through each image pair ===
for i, (left_file, right_file) in enumerate(zip(left_images, right_images)):
    imgL = cv2.imread(left_file)
    imgR = cv2.imread(right_file)

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # detect chessboard corners
    retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, None)

    if retL and retR:
        # if both images have valid boards, store object points
        objpoints.append(objp)

        # make detected corners more precise
        cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)

        imgpoints_left.append(cornersL)
        imgpoints_right.append(cornersR)

        # show combined visualization of detected corners
        vis = np.hstack((
            cv2.drawChessboardCorners(imgL.copy(), CHECKERBOARD, cornersL, retL),
            cv2.drawChessboardCorners(imgR.copy(), CHECKERBOARD, cornersR, retR)
        ))
        cv2.imshow("Detected Corners (Left | Right)", vis)
        cv2.waitKey(100)
    else:
        print(f"  Skipped pair {i:03d} (checkerboard not found)")

cv2.destroyAllWindows()
print(f" Detected corners in {len(objpoints)} valid image pairs")

# need enough samples or the result will be unreliable
if len(objpoints) < 10:
    raise Exception("Not enough valid image pairs detected. Capture at least 15-20 with full board visible.")

# === Calibrate each camera individually ===
print("\n Calibrating individual cameras...")
retL, K_L, D_L, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpoints_left, grayL.shape[::-1], None, None)
retR, K_R, D_R, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpoints_right, grayR.shape[::-1], None, None)

# === Stereo calibration ===
# this finds rotation & translation between cameras
print("\n Performing stereo calibration...")
flags = cv2.CALIB_FIX_INTRINSIC  # keep intrinsics fixed while optimizing stereo params
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

retStereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    K_L, D_L, K_R, D_R,
    grayL.shape[::-1],
    criteria=criteria_stereo,
    flags=flags
)

print(f"\n Stereo calibration complete.")
print(f"Reprojection error: {retStereo:.4f}")  # lower = better calibration
print(f"Rotation (R):\n{R}")  # rotation from left to right camera
print(f"Translation (T):\n{T}")  # translation between cameras

# === Stereo Rectification ===
print("\n🔧 Computing rectification and projection matrices...")
# this aligns both images so rows match for stereo matching
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K_L, D_L, K_R, D_R, grayL.shape[::-1], R, T, alpha=0
)

# === Saving calibration results ===
def save_yaml(filename, data):
    # simple helper to write YAML files
    with open(filename, 'w') as f:
        yaml.dump(data, f)

output_dir = os.path.expanduser('~/stereo_calib_results')
os.makedirs(output_dir, exist_ok=True)

# left camera params
save_yaml(os.path.join(output_dir, 'left.yaml'), {
    'K': K_L.tolist(),
    'D': D_L.tolist(),
    'R': R1.tolist(),
    'P': P1.tolist()
})
# right camera params
save_yaml(os.path.join(output_dir, 'right.yaml'), {
    'K': K_R.tolist(),
    'D': D_R.tolist(),
    'R': R2.tolist(),
    'P': P2.tolist()
})
# stereo / rectification data
save_yaml(os.path.join(output_dir, 'stereo.yaml'), {
    'R': R.tolist(),
    'T': T.tolist(),
    'Q': Q.tolist(),
    'reprojection_error': float(retStereo)
})

print(f"\n Calibration files saved to: {output_dir}")
print("   - left.yaml")
print("   - right.yaml")
print("   - stereo.yaml")
print("\n Done. You can now test rectification and disparity mapping!")
