import time
import torch

def encode(seq_tokens, token_to_idx, allow_unk=False):
    seq_idx = []
    for token in seq_tokens:
        if token not in token_to_idx:
            if allow_unk:
                token = '<UNK>'
            else:
                raise KeyError('Token "%s" not in vocab' % token)
        seq_idx.append(token_to_idx[token])
    return seq_idx


def decode(seq_idx, idx_to_token, delim=None, stop_at_end=True):
    tokens = []
    for idx in seq_idx:
        tokens.append(idx_to_token[idx])
        if stop_at_end and tokens[-1] == '<END>':
            break
    if delim is None:
        return tokens
    else:
        return delim.join(tokens)

def sample_representative_frames(frames, model, args):
    chunk_size = args.chunk_size
    feat_chunks = []
    num_frames = frames.size(0)
    num_chunks = num_frames // chunk_size + 1
    
    for i in range(num_chunks):
        chunk_feats = model(frames[i*chunk_size:(i+1)*chunk_size]).pooler_output
        feat_chunks.append(chunk_feats)

    all_feats = torch.cat(feat_chunks, dim=0) # (N, 768)
    all_sims = all_feats @ all_feats.transpose(0, 1)  # (N, N)       

def sample_frames_uniform(frames, K=32):
    num_frames = len(frames)
    assert num_frames > K
    intv = num_frames // K
    return frames[::intv]

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff