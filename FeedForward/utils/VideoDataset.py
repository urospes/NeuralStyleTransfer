import os
import cv2
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class VideoDataset(Dataset):
    def __init__(self, video_dir: str, seq_length: int = 32):
        super(VideoDataset).__init__()
        self.video_dir = video_dir
        self.seq_length = seq_length
        self.video_paths = [os.path.join(video_dir, video_name) for video_name in os.listdir(video_dir)]
        self.frames_rd_count = {key: 0 for key in self.video_paths}

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        frames = []
        n_rd = 0

        v_rd = cv2.VideoCapture(video_path)
        v_rd.set(cv2.CAP_PROP_POS_FRAMES, self.frames_rd_count[video_path])
        ret, frame = v_rd.read()
        while ret and n_rd < self.seq_length:
            n_rd += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            ret, frame = v_rd.read()
        v_rd.release()

        self.frames_rd_count[video_path] += n_rd
        return frames
