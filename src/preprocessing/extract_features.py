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
from datautils import utils
from datautils import tgif_qa
from datautils import msrvtt_qa
from datautils import msvd_qa
from datautils import svqa
from transformers import CLIPImageProcessor


def extract_clips_with_consecutive_frames(processor, path, num_frames_per_video=-1, intv=2):
    """
    Args:
        path: path of a video
        num_clips: expected numbers of splitted clips
        num_frames_per_clip: number of frames in a single clip, pretrained model only supports 16 frames
    Returns:
        A list of raw features of clips.
    """
    valid = True
    try:
        cap = cv2.VideoCapture(path)
        video_data = []
        if cap.isOpened():
            rval, frame = cap.read()
            while rval:
                b, g, r = cv2.split(frame)
                frame = cv2.merge([r, g, b])
                video_data.append(frame)
                rval, frame = cap.read()
        cap.release()
    except:
        print('file {} error'.format(path))
        raise ValueError
        valid = False
        return list(np.zeros(shape=(num_frames_per_video, 3, 224, 224))), valid
    
    total_frames = len(video_data)
    if num_frames_per_video > 0: 
        intv = total_frames // num_frames_per_video
    frames = video_data[::intv]
    processed_frames = processor(frames, return_tensors="pt")
    return processed_frames, valid

def generate_h5(processor, video_ids, num_clips, outfile):  # default-8
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
        for i, video_path in enumerate(tqdm(video_ids)):
            video_id = video_path.split('/')[-1].split('.')[0]
            _t['misc'].tic()
            frames, valid = extract_clips_with_consecutive_frames(processor, video_path)
            if args.feature_type == 'video':
                frames_torch = torch.FloatTensor(np.asarray(frames['pixel_values'])).cpu()
                if valid:
                    frames_torch = frames_torch.squeeze()
                else:
                    frames_torch = np.zeros(shape=(num_clips, 2048))
                if feat_dset is None:
                    feat_dset = fd['processed_feats'] = list(range(dataset_size))
                    video_ids_dset = fd['video_id'] = list(range(dataset_size))

            i1 = i0 + 1
            feat_dset[i0:i1] = frames_torch
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
    parser.add_argument('--dataset_root', default='./dataset', type=str)
    parser.add_argument('--question_type', default='none', choices=['none'], type=str)
    # output
    parser.add_argument('--out', dest='outfile',
                        help='output filepath', default="{}_{}_feat_24_{}clip.h5", type=str)
    # image sizes
    parser.add_argument('--num_clips', default=5, type=int)
    parser.add_argument('--image_height', default=224, type=int)
    parser.add_argument('--image_width', default=224, type=int)

    # network params
    # parser.add_argument('--model', default='resnet101', choices=['resnet101', 'resnext101'], type=str)
    parser.add_argument('--seed', default='666', type=int, help='random seed')
    args = parser.parse_args()
    args.feature_type = 'video'

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # initialize clip processors
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    dataset_path = os.path.join(args.dataset_root, args.dataset)

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
        args.annotation_file = os.path.join(dataset_path, 'annotations/qa_{}.json')
        # args.video_dir = './dataset/msvd_qa/video/'
        args.video_dir = os.path.join(dataset_path, 'YouTubeClips')
        video_paths = msvd_qa.load_video_paths(args)
        random.shuffle(video_paths)
        
        outpath = os.path.join(dataset_path, 'processed')
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        outfile = os.path.join(outpath, args.outfile.format(args.dataset, args.feature_type, str(args.num_clips)))
        # load model
        generate_h5(processor, video_paths, args.num_clips,
                    outfile)

    elif args.dataset == 'svqa':
        args.annotation_file = './data/SVQA/questions.json'
        args.video_dir = './data/SVQA/useVideo/'
        video_paths = svqa.load_video_paths(args)
        random.shuffle(video_paths)
        # load model
        generate_h5(processor, video_paths, args.num_clips,
                args.outfile.format(args.dataset, args.feature_type, str(args.num_clips)))

