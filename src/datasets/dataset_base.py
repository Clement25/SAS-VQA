from torch.utils.data import Dataset
from PIL import Image
import io
import torch
import numpy as np
import h5py
from src.datasets import decoder
from src.datasets.data_utils import (
    ImageResize, ImagePad, image_to_tensor)
from src.utils.load_save import LOGGER


def get_video_decoding_kwargs(container, num_frames, target_fps,
                              num_clips=None, clip_idx=None,
                              sampling_strategy="rand",
                              safeguard_duration=False, video_max_pts=None):
    if num_clips is None:
        three_clip_names = ["start", "middle", "end"]  # uniformly 3 clips
        assert sampling_strategy in ["rand", "uniform"] + three_clip_names
        if sampling_strategy == "rand":
            decoder_kwargs = dict(
                container=container,
                sampling_rate=1,
                num_frames=num_frames,
                clip_idx=-1,  # random sampling
                num_clips=None,  # will not be used when clip_idx is `-1`
                target_fps=target_fps
            )
        elif sampling_strategy == "uniform":
            decoder_kwargs = dict(
                container=container,
                sampling_rate=1,  # will not be used when clip_idx is `-2`
                num_frames=num_frames,
                clip_idx=-2,  # uniformly sampling from the whole video
                num_clips=1,  # will not be used when clip_idx is `-2`
                target_fps=target_fps  # will not be used when clip_idx is `-2`
            )
        else:  # in three_clip_names
            decoder_kwargs = dict(
                container=container,
                sampling_rate=1,
                num_frames=num_frames,
                clip_idx=three_clip_names.index(sampling_strategy),
                num_clips=3,
                target_fps=target_fps
            )
    else:  # multi_clip_ensemble, num_clips and clip_idx are only used here
        assert clip_idx is not None
        # sampling_strategy will not be used, as uniform sampling will be used by default.
        # uniformly sample `num_clips` from the video,
        # each clip sample num_frames frames at target_fps.
        decoder_kwargs = dict(
            container=container,
            sampling_rate=1,
            num_frames=num_frames,
            clip_idx=clip_idx,
            num_clips=num_clips,
            target_fps=target_fps,
            safeguard_duration=safeguard_duration,
            video_max_pts=video_max_pts
        )
    return decoder_kwargs

def load_decompress_img_from_lmdb_value(lmdb_value):
    """
    Args:
        lmdb_value: image binary from
            with open(filepath, "rb") as f:
                lmdb_value = f.read()

    Returns:
        PIL image, (h, w, c)
    """
    io_stream = io.BytesIO(lmdb_value)
    img = Image.open(io_stream, mode="r")
    return img


class BaseDataset(Dataset):
    """
    datalist: list(dicts)  # lightly pre-processed
        {
        "type": "image",
        "filepath": "/abs/path/to/COCO_val2014_000000401092.jpg",
        "text": "A plate of food and a beverage are on a table.",
                # should be tokenized and digitized first?
        ...
        }
    tokenizer:
    max_img_size: int,
    max_txt_len: int, max text sequence length, including special tokens.
    fps: float, frame per second
    num_frm: #frames to use as input.
    """

    def __init__(self, datalist, tokenizer, img_hdf5_dir, fps=3, num_frm=3,
                 frm_sampling_strategy="rand", max_img_size=-1, max_txt_len=20):
        self.fps = fps
        self.num_frm = num_frm
        self.frm_sampling_strategy = frm_sampling_strategy
        self.datalist = datalist
        self.tokenizer = tokenizer
        self.max_txt_len = max_txt_len

        # use hdf5 instead of lmdb
        self.dataset = h5py.File(img_hdf5_dir)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        raise NotImplementedError

    def _load_img(self, img_id):
        """Load and apply transformation to image

        Returns:
            torch.float, in [0, 255], (n_frm=1, c, h, w)
        """
        raw_img = load_decompress_img_from_lmdb_value(
            self.txn.get(str(img_id).encode("utf-8"))
        )
        image_np = np.array(raw_img, dtype=np.uint8)  # (h, w, c)
        raw_img_tensor = image_to_tensor(
            image_np, keepdim=False).float()  # (c, h, w) [0, 255]
        resized_img = self.img_resize(raw_img_tensor)
        transformed_img = self.img_pad(
            resized_img)  # (n_frm=1, c, h, w)
        return transformed_img

    @classmethod
    def _is_extreme_aspect_ratio(cls, tensor, max_ratio=5.):
        """ find extreme aspect ratio, where longer side / shorter side > max_ratio
        Args:
            tensor: (*, H, W)
            max_ratio: float, max ratio (>1).
        """
        h, w = tensor.shape[-2:]
        return h / float(w) > max_ratio or h / float(w) < 1 / max_ratio

    def _load_video(self, video_id, num_clips=None, clip_idx=None,
                    safeguard_duration=False, video_max_pts=None):
        """Load and sample frames from video.
        Apply transformation to the sampled frames.

        Sample a clip:
            - random: set num_clips and clip_idx to be None
            - uniform: set num_clips=N, clip_idx=idx. e.g., num_clips=3
                and clip_idx=1 will first segment the video into 3 clips,
                then sample the 2nd clip.

        Returns:
            torch.float, in [0, 255], (n_frm=T, c, h, w)
        """
        assert (num_clips is None) == (clip_idx is None), "Both None, or both not None"
        # (T, C, H, W) [0, 255]
        io_stream = io.BytesIO(self.txn.get(str(video_id).encode("utf-8")))
        raw_sampled_frms, video_max_pts = extract_frames_from_video_binary(
            io_stream,
            target_fps=self.fps,
            num_frames=self.num_frm,
            multi_thread_decode=False,
            sampling_strategy=self.frm_sampling_strategy,
            num_clips=num_clips,
            clip_idx=clip_idx,
            safeguard_duration=safeguard_duration,
            video_max_pts=video_max_pts
        )

        if raw_sampled_frms is None:
            return None, None
        elif self._is_extreme_aspect_ratio(raw_sampled_frms, max_ratio=5.):
            print(
                f"Found extreme aspect ratio for video id {video_id}. Skip it")
            return None, None

        raw_sampled_frms = raw_sampled_frms.float()
        resized_frms = self.img_resize(raw_sampled_frms)
        padded_frms = self.img_pad(resized_frms)
        return padded_frms, video_max_pts

        

def img_collate(imgs):
    """
    Args:
        imgs:

    Returns:
        torch.tensor, (B, 3, H, W)
    """
    w = imgs[0].width
    h = imgs[0].height
    tensor = torch.zeros(
        (len(imgs), 3, h, w), dtype=torch.uint8).contiguous()
    for i, img in enumerate(imgs):
        nump_array = np.array(img, dtype=np.uint8)
        if (nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        # (H, W, 3) --> (3, H, W)
        nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)
    return tensor
