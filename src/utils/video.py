import cv2
import os

# Make a video with photos
def make_video(images, video_name, fps=30):
    img = cv2.imread(images[0])
    height, width, layers = img.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))
    for image in images:
        video.write(cv2.imread(image))
    cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    images = []
    for f in range(1,6):
        images.append('calibration/calib_'+str(f)+'_resized.jpg')
    for f in range(1,6):
        images.append('calibration/calib_'+str(f)+'_corners.jpg')
    make_video(images, 'calibration/calibration_video.mp4', 1)