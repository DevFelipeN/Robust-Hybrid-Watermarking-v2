import os
import argparse
import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from tqdm import tqdm
from datetime import datetime
from os.path import join

from torch.optim import Adam
from torch.utils.data import (
    Dataset,
    DataLoader
)

from torch.utils.tensorboard import (
    SummaryWriter
)

from torchvision import transforms
from torchvision.datasets import (
    CIFAR100
)
from torchvision.utils import (
    make_grid
)

from watermark_embedding import (
    watermark_embedding_tensor,
    covert_to_binary
)

from watermark_extraction import (
    WatermarkExtraction
)


# ==========================================
# Criar lista de imagens
# ==========================================
def find_images(root):
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(                (
                    ".jpg",
                    ".png",
                    ".jpeg"
                )
            ):
                paths.append(
                    os.path.join(
                        dirpath,
                        f
                    )
                )
    return sorted(paths)


# ==========================================
# Dataset
# ==========================================
class CelebHQDataset(Dataset):
    def __init__(
        self,
        image_paths,
        transform=None,
        scaling_factor=0.01,
        arnold_iterations=15,
        watermark_size=32
    ):
        self.image_paths = image_paths
        self.transform = transform
        self.scaling_factor = (
            scaling_factor
        )
        self.arnold_iterations = (
            arnold_iterations
        )
        self.watermark_size = (
            watermark_size
        )
        # CIFAR-100
        self.cifar_dataset = (
            CIFAR100(
                root="./cifar100_data",
                train=True,
                download=True
            )
        )
        # transform watermark
        self.wm_transform = (
            transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize(
                    (
                        self.watermark_size,
                        self.watermark_size
                    )
                ),

                transforms.ToTensor()
            ])
        )

    def __len__(self):
        return len(
            self.image_paths
        )
    def __getitem__(
        self,
        idx
    ):
        # -------------------------
        # Host image (CelebA-HQ)
        # -------------------------
        path = self.image_paths[
            idx
        ]
        img = Image.open(
            path
        ).convert("RGB")
        if self.transform:
            img = self.transform(
                img
            )
        # -------------------------
        # Random CIFAR watermark
        # -------------------------
        cifar_idx = (
            np.random.randint(
                0,
                len(
                    self.cifar_dataset
                )
            )
        )

        wm_img, _ = (
            self.cifar_dataset[
                cifar_idx
            ]
        )

        # PIL -> tensor
        wm_img = (
            self.wm_transform(
                wm_img
            )
        )

        # tensor -> numpy
        wm_numpy = (
            wm_img
            .squeeze(0)
            .numpy()
        )

        wm_numpy = (
            wm_numpy * 255
        ).astype(
            np.uint8
        )

        # -------------------------
        # watermark embedding
        # -------------------------
        watermarked, target = (
            watermark_embedding_tensor(
                image_tensor=img,

                watermark_image=
                wm_img,

                scaling_factor=
                self.scaling_factor,

                arnold_iterations=
                self.arnold_iterations,

                return_embedded=
                True
            )
        )

        return (
            watermarked,
            target
        )


# ==========================================
# Main
# ==========================================
def main():

    parser = (
        argparse
        .ArgumentParser()
    )

    # -------------------------
    # Arguments
    # -------------------------
    parser.add_argument(
        "--image_resolution",
        type=int,
        required=True
    )

    parser.add_argument(
        "--watermark_size",
        type=int,
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    parser.add_argument(
        "--scaling_factor",
        type=float,
        required=True
    )

    parser.add_argument(
        "--arnold_iterations",
        type=int,
        required=True
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--cuda",
        type=int,
        default=0
    )

    args = (
        parser.parse_args()
    )

    # -------------------------
    # CUDA
    # -------------------------
    os.environ[
        "CUDA_DEVICE_ORDER"
    ] = "PCI_BUS_ID"

    os.environ[
        "CUDA_VISIBLE_DEVICES"
    ] = str(
        args.cuda
    )

    # -------------------------
    # Paths
    # -------------------------
    LOGS_PATH = join(
        args.output,
        "logs"
    )

    CHECKPOINTS_PATH = join(
        args.output,
        "checkpoints"
    )

    for path in [
        LOGS_PATH,
        CHECKPOINTS_PATH
    ]:

        os.makedirs(
            path,
            exist_ok=True
        )

    writer = SummaryWriter(
        LOGS_PATH
    )

    # -------------------------
    # Logging steps
    # -------------------------
    plot_points = (
        list(
            range(
                0,
                1000,
                100
            )
        )
        +
        list(
            range(
                1000,
                3000,
                200
            )
        )
        +
        list(
            range(
                3000,
                100000,
                1000
            )
        )
    )

    now = (
        datetime.now()
    )

    dt_string = (
        now.strftime(
            "%d%m%Y_%H:%M:%S"
        )
    )

    EXP_NAME = (
        "watermark_DCT_DWT_"
        +
        dt_string
    )

    # -------------------------
    # Device
    # -------------------------
    device = (
        torch.device(
            "cuda"
            if torch.cuda
            .is_available()
            else "cpu"
        )
    )

    print(
        f"Using device:"
        f" {device}"
    )

    # -------------------------
    # Dataset
    # -------------------------
    transform = (
        transforms.Compose([
            transforms.Resize(
                (
                    args
                    .image_resolution,

                    args
                    .image_resolution
                )
            ),

            transforms
            .ToTensor()
        ])
    )

    image_paths = (
        find_images(
            "celeba_hq/train"
        )
    )

    dataset = (
        CelebHQDataset(
            image_paths=
            image_paths,

            transform=
            transform,

            scaling_factor=
            args
            .scaling_factor,

            arnold_iterations=
            args
            .arnold_iterations,

            watermark_size=args.watermark_size
        )
    )

    print(
        f"Loaded "
        f"{len(dataset)} "
        f"images."
    )

    dataloader = (
        DataLoader(
            dataset,
            batch_size=
            args.batch_size,

            shuffle=True,

            num_workers=2,

            pin_memory=True
        )
    )

    # -------------------------
    # Model
    # -------------------------
    model = (
        WatermarkExtraction(
            watermark_size=args.watermark_size
        )
        .to(device)
    )

    # -------------------------
    # Optimizer
    # -------------------------
    optimizer = (
        Adam(
            model.parameters(),
            lr=args.lr,
            betas=(
                0.9,
                0.999
            ),
            weight_decay=1e-5
        )
    )

    # -------------------------
    # Loss
    # -------------------------
    criterion = (
        nn
        .BCEWithLogitsLoss()
    )

    global_step = 0

    # ==================================
    # Training
    # ==================================
    for epoch in range(
        args.num_epochs
    ):

        print(
            f"\nEpoch "
            f"{epoch+1}/"
            f"{args.num_epochs}"
        )

        model.train()

        for (
            images,
            watermarks
        ) in tqdm(
            dataloader
        ):

            global_step += 1

            # -----------------
            # Device
            # -----------------
            images = (
                images.to(
                    device,
                    non_blocking=
                    True
                )
            )

            watermarks = (
                watermarks.to(
                    device,
                    non_blocking=
                    True
                )
            )

            # -----------------
            # Forward
            # -----------------
            decoder_output = model(
                images
            )

            # -----------------
            # Loss
            # -----------------
            loss = (
                criterion(
                    decoder_output,
                    watermarks
                )
            )

            # -----------------
            # Backprop
            # -----------------
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            # -----------------
            # Metrics
            # -----------------
            pred_binary = (
                torch.sigmoid(
                    decoder_output
                ) > 0.5
            ).float()

            bitwise_accuracy = (
                (
                    pred_binary
                    ==
                    watermarks
                )
                .float()
                .mean()
            )

            ber = (
                1.0
                -
                bitwise_accuracy
            )

            # -----------------
            # Logging
            # -----------------
            if (
                global_step
                in plot_points
            ):

                writer.add_scalar(
                    "loss",
                    loss.item(),
                    global_step
                )

                writer.add_scalar(
                    "bitwise_accuracy",
                    bitwise_accuracy.item(),
                    global_step
                )

                writer.add_scalar(
                    "BER",
                    ber.item(),
                    global_step
                )

                writer.add_image(
                    "marked_images",
                    make_grid(
                        images,
                        normalize=True
                    ),
                    global_step
                )

                writer.add_image(
                    "watermark_gt",
                    make_grid(
                        watermarks,
                        normalize=True
                    ),
                    global_step
                )

                writer.add_image(
                    "watermark_pred",
                    make_grid(
                        pred_binary,
                        normalize=True
                    ),
                    global_step
                )

                print(
                    f"Step "
                    f"{global_step} | "
                    f"Loss: "
                    f"{loss.item():.4f} | "
                    f"BitAcc: "
                    f"{bitwise_accuracy.item():.4f} | "
                    f"BER: "
                    f"{ber.item():.4f}"
                )

            # -----------------
            # Checkpoint
            # -----------------
            if (
                global_step
                % 5000
                == 0
            ):

                torch.save(
                    optimizer
                    .state_dict(),

                    join(
                        CHECKPOINTS_PATH,
                        EXP_NAME
                        +
                        "_optim.pth"
                    )
                )

                torch.save(
                    model
                    .state_dict(),

                    join(
                        CHECKPOINTS_PATH,
                        EXP_NAME
                        +
                        "_extractor.pth"
                    )
                )
            
    torch.save(
        model.state_dict(),
        join(
            CHECKPOINTS_PATH,
            EXP_NAME
            + "_final.pth"
        )
    )

    torch.save(
        optimizer.state_dict(),
        join(
            CHECKPOINTS_PATH,
            EXP_NAME
            + "_optim_final.pth"
        )
    )

    writer.close()


if __name__ == "__main__":
    main()