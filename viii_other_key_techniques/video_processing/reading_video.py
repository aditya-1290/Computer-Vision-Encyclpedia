"""
Reading Video: Load and process video frames using OpenCV.

Implementation uses OpenCV for video capture and frame extraction.

Theory:
- Video: Sequence of images (frames) over time.
- Frame rate: Frames per second (FPS).
- Resolution: Width x Height of each frame.

Math: Total frames = duration * FPS

Reference:
- OpenCV VideoCapture documentation
"""

import cv2

def read_video(video_path, max_frames=None):
    """
    Read video frames.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_count += 1
        if max_frames and frame_count >= max_frames:
            break
    cap.release()
    return frames

if __name__ == "__main__":
    # Assume a video file exists
    # frames = read_video("sample_video.mp4", max_frames=10)
    # print(f"Read {len(frames)} frames")
    print("Video reading function defined.")
