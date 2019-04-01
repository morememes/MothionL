import cv2
import numpy as np

class DataLoader:
    def __init__(self, video_path, batch_size):
        self.video_path = video_path
        self.batch_size = batch_size

        self.cap = cv2.VideoCapture(self.video_path)

    def __get_batch__(self):
        ret, batch = self.cap.read()
        if ret:
            batch = batch[None]
            for i in range(1, self.batch_size):
                ret, frame = self.cap.read()
                if ret:
                    batch = np.concatenate((batch, frame[None]), axis=0)
                else:
                    self.cap.release()
                    return ret, batch
        else:
            self.cap.release()
            return ret, None
        return ret, batch


    def __len__(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))