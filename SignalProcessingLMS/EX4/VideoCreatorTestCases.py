import cv2
from CustomDataLoaderCarRacing import CustomDataLoaderCarRacing
import torch
import numpy as np


class VideoCreatorTestCases():

    def __init__(self, fps=15, image_size=(96,96), test_case="flipud", street_color=(155, 102, 60)):
        # for avi videos
        #self._fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        # for mp4 videos
        self._fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._fps = fps
        self._image_size = image_size
        self._street_color = street_color
        self._test_case = test_case

        self.create_dataloader_for_case(self._test_case)

    def create_dataloader_for_case(self, test_case):
        self._dataloader = CustomDataLoaderCarRacing(street_color=self._street_color, transforms=test_case,
                                                    dataset_size=1000, data_split="test",
                                               split_ratio=(60, 20, 20), crop_size=72, image_size=96, input_channels=3)

    def create_video(self, videoname):
        self._video = cv2.VideoWriter(filename=videoname, fourcc=self._fourcc, fps=self._fps,
                                     frameSize=self._image_size)
        for i, data in enumerate(self._dataloader):
            img, label = data
            img_numpy_RGB = torch.permute((img), (1, 2, 0)).numpy().astype(dtype=np.uint8)
            img_numpy_BGR = cv2.cvtColor(img_numpy_RGB, cv2.COLOR_RGB2BGR)
            self._video.write(img_numpy_BGR)
        self._video.release()
        cv2.destroyAllWindows()