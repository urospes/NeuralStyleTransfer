import os
import time
import torch
import cv2
from utils import utils, stylize

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    MODELS_PATH = "models"
    VIDEOS_PATH = "videos"

    MODEL_NAME = "st_starry_night_tr_ms_coco_40k_1723738077"
    MODEL_PATH = os.path.join(MODELS_PATH, MODEL_NAME, "params.pth")

    VIDEO_NAME = "lions"    # staviti ime foldera u kome se nalazi zeljeni video
    VIDEO_PATH = os.path.join(VIDEOS_PATH, VIDEO_NAME, "video.mp4")
    OUTPUT_VIDEO_NAME = f'st_{VIDEO_NAME}_model_{MODEL_NAME}.mp4'

    model = utils.load_model(MODEL_PATH, device)

    v_rd = cv2.VideoCapture(VIDEO_PATH)
    video_size = (int(v_rd.get(cv2.CAP_PROP_FRAME_WIDTH)), int(v_rd.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = int(v_rd.get(cv2.CAP_PROP_FPS))

    v_wr = cv2.VideoWriter(os.path.join(VIDEOS_PATH, VIDEO_NAME, OUTPUT_VIDEO_NAME),
                           cv2.VideoWriter_fourcc(*'mp4v'), fps, video_size)

    start = time.time()

    i = 1
    ret, frame = v_rd.read()
    while ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stylized_frame = stylize.stylize_frame(frame, model, device)
        stylized_frame = cv2.cvtColor(stylized_frame, cv2.COLOR_RGB2BGR)
        v_wr.write(stylized_frame)

        ret, frame = v_rd.read()
        i += 1

    end = time.time()

    v_rd.release()
    v_wr.release()
    print(f'Finished stylizing {i} frames... Elapsed time: {end - start} seconds')
