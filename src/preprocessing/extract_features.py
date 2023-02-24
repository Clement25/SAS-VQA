import argparse, os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import h5py
import cv2
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn
import torchvision
import random
import numpy as np
from models import resnext
from datautils import utils
from datautils import tgif_qa
from datautils import msrvtt_qa
from datautils import msvd_qa
from datautils import svqa
from transformers import CLIPImageProcessor


def run_batch(cur_batch, model):
    """
    Args:
        cur_batch: treat a video as a batch of images
        model: ResNet model for feature extraction
    Returns:
        ResNet extracted feature.
    """
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)

    image_batch = cur_batch.astype(np.float32)
    image_batch = (image_batch / 255.0 - mean) / std
    image_batch = torch.FloatTensor(image_batch).cuda()
    with torch.no_grad():
        image_batch = torch.autograd.Variable(image_batch)

    feats = model(image_batch)
    feats = feats.data.cpu().clone().numpy()
    return feats


def extract_clips_with_consecutive_frames(path, num_clips, num_frames_per_clip):
    """
    Args:
        path: path of a video
        num_clips: expected numbers of splitted clips
        num_frames_per_clip: number of frames in a single clip, pretrained model only supports 16 frames
    Returns:
        A list of raw features of clips.
    """
    valid = True
    clips = list()
    print(path)
    try:
        cap = cv2.VideoCapture(path)
        video_data = []
        if cap.isOpened():
            rval, frame = cap.read()
            while rval:
                # rval, frame = cap.read()
                b, g, r = cv2.split(frame)
                frame = cv2.merge([r, g, b])
                video_data.append(frame)
                rval, frame = cap.read()
                cv2.waitKey(1)
        cap.release()
    except:
        print('file {} error'.format(path))
        raise ValueError
        valid = False
        if args.model == 'resnext101':
            return list(np.zeros(shape=(num_clips, 3, num_frames_per_clip, 112, 112))), valid
        else:
            return list(np.zeros(shape=(num_clips, num_frames_per_clip, 3, 224, 224))), valid
    total_frames = len(video_data)
    img_size = (args.image_height, args.image_width)
    for i in np.linspace(0, total_frames, num_clips + 2, dtype=np.int32)[1:num_clips + 1]:  # [0 10 20 30 40]
        clip_start = int(i) - int(num_frames_per_clip / 2)  # 2 12 22
        clip_end = int(i) + int(num_frames_per_clip / 2)  # 18 28 38
        if clip_start < 0:
            clip_start = 0
        if clip_end > total_frames:
            clip_end = total_frames - 1
        clip = video_data[clip_start:clip_end]
        if clip_start == 0:  # beginning frames
            shortage = num_frames_per_clip - (clip_end - clip_start)
            added_frames = []
            for _ in range(shortage):
                added_frames.append(np.expand_dims(video_data[clip_start], axis=0))
            if len(added_frames) > 0:
                added_frames = np.concatenate(added_frames, axis=0)
                clip = np.concatenate((added_frames, clip), axis=0)
        if clip_end == (total_frames - 1):
            shortage = num_frames_per_clip - (clip_end - clip_start)
            added_frames = []
            for _ in range(shortage):
                added_frames.append(np.expand_dims(video_data[clip_end], axis=0))
            if len(added_frames) > 0:
                added_frames = np.concatenate(added_frames, axis=0)
                clip = np.concatenate((clip, added_frames), axis=0)
        new_clip = []
        for j in range(num_frames_per_clip):
            frame_data = clip[j]
            img = Image.fromarray(frame_data)
            img = img.resize(img_size, Image.BICUBIC)
            #img = img.transpose(2, 0, 1)[None]
            frame_data = np.array(img)
            frame_data = np.transpose(frame_data, axes=(2, 0, 1))
            new_clip.append(frame_data)
        new_clip = np.asarray(new_clip)  # (num_frames, width, height, channels)
        if args.model in ['resnext101']:
            new_clip = np.squeeze(new_clip)
            new_clip = np.transpose(new_clip, axes=(1, 0, 2, 3))
        clips.append(new_clip)
    return clips, valid


def generate_h5(model, video_ids, num_clips, outfile):  # default-8
    """
    Args:
        model: loaded pretrained model for feature extraction
        video_ids: list of video ids
        num_clips: expected numbers of splitted clips
        outfile: path of output file to be written
    Returns:
        h5 file containing visual features of splitted clips.
    """
    if not os.path.exists('data/{}'.format(args.dataset)):
        os.makedirs('data/{}'.format(args.dataset))

    dataset_size = len(video_ids)  # paths

    with h5py.File(outfile, 'w') as fd:
        feat_dset = None
        video_ids_dset = None
        i0 = 0
        _t = {'misc': utils.Timer()}
        for i, (video_path, video_id) in enumerate(tqdm(video_ids)):
            _t['misc'].tic()
            clips, valid = extract_clips_with_consecutive_frames(video_path, num_clips=num_clips, num_frames_per_clip=16)
            if args.feature_type == 'appearance':
                clip_feat = []
                if valid:
                    for clip_id, clip in enumerate(clips):
                        feats = run_batch(clip, model)
                        feats = feats.squeeze()
                        clip_feat.append(feats)
                else:
                    clip_feat = np.zeros(shape=(num_clips, 16, 2048))
                clip_feat = np.asarray(clip_feat)  # (num_clips, num_frame, 2048)
                if feat_dset is None:
                    print(clip_feat.shape)
                    C, F, D = clip_feat.shape
                    feat_dset = fd.create_dataset('resnet_features', (dataset_size, C, F, D),
                                                  dtype=np.float32)
                    video_ids_dset = fd.create_dataset('video_id', shape=(dataset_size,), dtype=np.int)
            elif args.feature_type == 'motion':
                clip_torch = torch.FloatTensor(np.asarray(clips)).cuda()
                if valid:
                    print('====' * 50)
                    clip_feat = model(clip_torch)  # (8, 2048)
                    print('clip_feat:', clip_feat.shape)
                    clip_feat = clip_feat.squeeze()
                    clip_feat = clip_feat.detach().cpu().numpy()
                else:
                    clip_feat = np.zeros(shape=(num_clips, 2048))
                if feat_dset is None:
                    C, D = clip_feat.shape
                    feat_dset = fd.create_dataset('resnext_features', (dataset_size, C, D),
                                                  dtype=np.float32)
                    video_ids_dset = fd.create_dataset('video_id', shape=(dataset_size,), dtype=np.int)

            i1 = i0 + 1
            feat_dset[i0:i1] = clip_feat
            video_ids_dset[i0:i1] = video_id
            i0 = i1
            _t['misc'].toc()
            if (i % 1000 == 0):
                print('{:d}/{:d} {:.3f}s (projected finish: {:.2f} hours)' \
                      .format(i1, dataset_size, _t['misc'].average_time,
                              _t['misc'].average_time * (dataset_size - i1) / 3600))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=5, help='specify which gpu will be used')
    # dataset info-选定数据集
    parser.add_argument('--dataset', default='msvd-qa', choices=['msvd-qa', 'msrvtt-qa', 'svqa'], type=str)
    parser.add_argument('--question_type', default='none', choices=['none'], type=str)
    # output
    parser.add_argument('--out', dest='outfile',
                        help='output filepath', default="{}_{}_feat_24_{}clip.h5", type=str)
    # image sizes
    parser.add_argument('--num_clips', default=10, type=int)
    parser.add_argument('--image_height', default=224, type=int)
    parser.add_argument('--image_width', default=224, type=int)

    # network params
    parser.add_argument('--model', default='resnet101', choices=['resnet101', 'resnext101'], type=str)
    parser.add_argument('--seed', default='666', type=int, help='random seed')
    args = parser.parse_args()

    if args.model == 'resnet101':
        args.feature_type = 'appearance'
    elif args.model == 'resnext101':
        args.feature_type = 'motion'
    else:
        raise Exception('Feature type not supported!')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # initialize clip processors
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # annotation files
    if args.dataset == 'tgif-qa':
        args.annotation_file = './data/tgif-qa/csv/Total_{}_question.csv'
        args.video_dir = './data/tgif-qa/gifs'
        video_paths = tgif_qa.load_video_paths(args)
        random.shuffle(video_paths)
        # load model
        generate_h5(processor, video_paths, args.num_clips,
                    args.outfile.format(args.dataset, args.feature_type, str(args.num_clips)))


    if args.dataset == 'msrvtt-qa':
        args.annotation_file = './data/MSRVTT-QA/{}_qa.json'
        args.video_dir = './data/MSRVTT-QA/video/'
        video_paths = msrvtt_qa.load_video_paths(args)
        random.shuffle(video_paths)
        # load model
        generate_h5(processor, video_paths, args.num_clips,
                    args.outfile.format(args.dataset, args.feature_type, str(args.num_clips)))

    if args.dataset == 'msvd-qa':
        args.annotation_file = './data/MSVD-QA/{}_qa.json'
        args.video_dir = './data/MSVD-QA/video/'
        args.video_name_mapping = './data/MSVD-QA/youtube_mapping.txt'
        video_paths = msvd_qa.load_video_paths(args)
        random.shuffle(video_paths)
        # load model
        generate_h5(processor, video_paths, args.num_clips,
                    args.outfile.format(args.dataset, args.feature_type, str(args.num_clips)))

    elif args.dataset == 'svqa':
        args.annotation_file = './data/SVQA/questions.json'
        args.video_dir = './data/SVQA/useVideo/'
        video_paths = svqa.load_video_paths(args)
        random.shuffle(video_paths)
        # load model
        generate_h5(processor, video_paths, args.num_clips,
                args.outfile.format(args.dataset, args.feature_type, str(args.num_clips)))

