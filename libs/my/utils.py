import cv2
import numpy as np

class VideoWorker:
    def __init__(self, video_path, masks):
        self.cap = cv2.VideoCapture(video_path)
        self.masks = masks
        self.writer = None

    def create_writer(self, path, fps: float = 60.0, shape = (1920,1080)):
        if self.writer == None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.writer = cv2.VideoWriter(path, fourcc, fps, shape)

    def create_capture(self, path):
        if self.cap == None:
            self.cap = cv2.VideoCapture(video_path)

    def release_writer(self):
        if self.writer != None:
            self.writer.release()
            self.writer = None

    def release_capture(self):
        if self.cap != None:
            self.cap.release()
            self.cap = None


class MasksCreator(VideoWorker):
    def __init__(self, video_path, masks, result_path):
        super().__init__(video_path, masks)
        self.result_path = result_path

    def cutter(self):
        self.create_writer(self.result_path)
        i = 0
        iterator = iter(self.masks)
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = frame.astype(np.uint8)
            mask = next(iterator).astype(np.uint8)
            img = cv2.bitwise_and(frame, frame, mask=mask)
            self.writer.write(img)
            print(f"Processing {i} frame")
            i += 1
        print("Complete!")
        self.release_capture()
        self.release_writer()