import warnings
from pathlib import Path

import albumentations as albu
import cv2
import numpy as np
import torch
from cloths_segmentation.pre_trained_models import create_model
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from iglovikov_helper_functions.utils.image_utils import pad, unpad
from loguru import logger

warnings.filterwarnings("ignore")


class GenImageMask:

    def __init__(self, output_path='./output') -> None:
        self.output_path = Path(output_path)

    def prepare(self, model_name: str = "Unet_2020-10-30"):
        self.model = create_model(model_name)
        self.model.eval()
        self.transform = albu.Compose([albu.Normalize(p=1)], p=1)
        if not self.output_path.exists():
            self.output_path.mkdir(parents=True)

    def gen_mask(self, image_path: str, save_to_local: bool = False):
        try:
            b = cv2.imread(image_path)
            # logger.debug('cloth shape -> {}'.format(b.shape))
            if b.shape != (1024, 768, 3):
                b = cv2.resize(b, (768, 1024))
                # logger.debug('cloth shape resize -> {}'.format(b.shape))
            img_rgb = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
            # logger.debug('cloth rbg shape -> {}'.format(img_rgb.shape))
            padded_image, pads = pad(
                img_rgb, factor=32, border=cv2.BORDER_CONSTANT)

            x = self.transform(image=padded_image)["image"]
            x = torch.unsqueeze(tensor_from_rgb_image(x), 0)

            with torch.no_grad():
                prediction = self.model(x)[0][0]

            mask = (prediction > 0).cpu().numpy().astype(np.uint8)
            mask = unpad(mask, pads)
            b_img = mask * 255
            shape = b_img.shape
            # logger.debug('mask shape -> {}'.format(shape))

            if b.shape[0] <= 1024 and b.shape[1] <= 768:
                img = np.full((1024, 768, 3), 255)
                seg_img = np.full((1024, 768), 0)
                img[int((1024 - shape[0]) / 2): 1024 - int((1024 - shape[0]) / 2),
                    int((768 - shape[1]) / 2):768 - int((768 - shape[1]) / 2)] = b
                seg_img[int((1024 - shape[0]) / 2): 1024 - int((1024 - shape[0]) / 2),
                        int((768 - shape[1]) / 2):768 - int((768 - shape[1]) / 2)] = b_img
            else:
                img = np.full((b.shape[0], b.shape[1], 3), 255)
                seg_img = np.full((b.shape[0], b.shape[1]), 0)
                img[int((b.shape[0] - shape[0]) / 2): b.shape[0] - int((b.shape[0] - shape[0]) / 2),
                    int((b.shape[1] - shape[1]) / 2):b.shape[1] - int((b.shape[1] - shape[1]) / 2)] = b
                seg_img[int((b.shape[0] - shape[0]) / 2): b.shape[0] - int((b.shape[0] - shape[0]) / 2),
                        int((b.shape[1] - shape[1]) / 2):b.shape[1] - int((b.shape[1] - shape[1]) / 2)] = b_img

            if save_to_local:
                cv2.imwrite(str(self.output_path / 'cloth.jpg'), img)
                cv2.imwrite(str(self.output_path / "cloth_mask.jpg"), seg_img)
        except Exception as e:
            logger.error(e)
            return False, str(e)

        return img, seg_img
