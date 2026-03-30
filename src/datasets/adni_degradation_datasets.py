import os
import random
from logging import getLogger

import torch
from torch.utils import data

logger = getLogger()


class ADNI_degradation(data.Dataset):
    def __init__(
        self,
        split='',
        processed_dir='',
        horizon='1y',
        use_normalization=False,
        downsample=False,
        sampling_rate=3,
        num_frames=140,
    ):
        self.use_normalization = use_normalization
        self.downsample = downsample
        self.sampling_rate = sampling_rate
        self.num_frames = num_frames

        self.n_rois = 450
        self.seq_length = 140

        prefix = f"adni_degradation_{horizon}"
        self.input_x_file = os.path.join(processed_dir, f'{prefix}_{split}_x.pt')
        self.label_y_file = os.path.join(processed_dir, f'{prefix}_{split}_y.pt')

        self.input_xs = torch.load(self.input_x_file)
        self.label_ys = torch.load(self.label_y_file)

    def __len__(self):
        return len(self.input_xs)

    def __getitem__(self, idx):
        input_x, label_y = self.input_xs[idx], self.label_ys[idx]
        input_x = input_x.float()

        if self.use_normalization:
            mean = input_x.mean()
            std = input_x.std()
            input_x = (input_x - mean) / (std + 1e-8)

        # Resample to 160 timepoints to match pretrained model (450 x 160)
        if input_x.shape[1] != 160:
            input_x = torch.nn.functional.interpolate(
                input_x.unsqueeze(0), size=160, mode='linear', align_corners=False
            ).squeeze(0)

        input_x = torch.unsqueeze(input_x, 0).to(torch.float32)

        return input_x.to(torch.float32), int(label_y)

    def _get_start_end_idx(self, fmri_size, clip_size):
        delta = max(fmri_size - clip_size, 0)
        start_idx = random.uniform(0, delta)
        end_idx = start_idx + clip_size - 1
        return start_idx, end_idx

    def _temporal_sampling(self, frames, start_idx, end_idx, num_samples):
        index = torch.linspace(start_idx, end_idx, num_samples)
        index = torch.clamp(index, 0, frames.shape[1] - 1).long()
        new_frames = torch.index_select(frames, 1, index)
        return new_frames


def make_adni_degradation(
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    drop_last=True,
    processed_dir='data/processed/adni',
    horizon='1y',
    use_normalization=False,
    label_normalization=False,
    downsample=False,
):
    train_dataset = ADNI_degradation(
        split='train',
        processed_dir=processed_dir,
        horizon=horizon,
        use_normalization=use_normalization,
        downsample=downsample,
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=collator,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False,
    )

    valid_dataset = ADNI_degradation(
        split='valid',
        processed_dir=processed_dir,
        horizon=horizon,
        use_normalization=use_normalization,
        downsample=downsample,
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        collate_fn=collator,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False,
    )

    test_dataset = ADNI_degradation(
        split='test',
        processed_dir=processed_dir,
        horizon=horizon,
        use_normalization=use_normalization,
        downsample=downsample,
    )

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        collate_fn=collator,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False,
    )

    logger.info('ADNI degradation dataset created')

    return train_data_loader, valid_data_loader, test_data_loader, train_dataset, valid_dataset, test_dataset
