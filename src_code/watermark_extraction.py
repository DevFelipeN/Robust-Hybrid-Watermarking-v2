import cv2
import torch
import pywt
import numpy as np
import torch.nn as nn

from watermark_embedding import (
    block_dct2d
)

class WatermarkExtraction(
    nn.Module
):

    def __init__(self, watermark_size=16):

        super().__init__()

        self.watermark_size = (
            watermark_size
        )

        self.encoder = nn.Sequential(

            nn.Conv2d(
                12, 32,
                3,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                32, 64,
                3,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                64, 128,
                3,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                128, 256,
                3,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (
                    self.watermark_size,
                    self.watermark_size
                )
            )
        )

        self.head = nn.Sequential(

            nn.Conv2d(
                256,
                128,
                3,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                128,
                64,
                3,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                64,
                1,
                1
            )
        )

    def extract_frequency_tensor(
        self,
        images,
        block_size=8
    ):

        batch_features = []

        for img in images:

            img_np = (
                img.permute(
                    1, 2, 0
                )
                .detach()
                .cpu()
                .numpy()
            )

            img_np = (
                img_np * 255
            ).astype(
                np.uint8
            )

            img_np = cv2.cvtColor(
                img_np,
                cv2.COLOR_RGB2BGR
            )

            channels = cv2.split(
                img_np
            )

            features = []

            for ch in channels:

                dct_img = (
                    block_dct2d(
                        ch.astype(
                            np.float32
                        ),
                        block_size
                    )
                )

                cA, (
                    cH,
                    cV,
                    cD
                ) = pywt.dwt2(
                    dct_img,
                    'haar'
                )

                features.extend([
                    cA,
                    cH,
                    cV,
                    cD
                ])

            features = np.stack(
                features,
                axis=0
            )

            features = (
                features
                -
                features.mean()
            ) / (
                features.std()
                + 1e-8
            )

            batch_features.append(
                features
            )

        batch_features = np.stack(
            batch_features,
            axis=0
        )

        return (
            torch.from_numpy(
                batch_features
            )
            .float()
            .to(images.device)
        )

    def forward(
        self,
        x
    ):

        x = (
            self
            .extract_frequency_tensor(
                x
            )
        )

        x = self.encoder(
            x
        )

        x = self.head(
            x
        )

        return x