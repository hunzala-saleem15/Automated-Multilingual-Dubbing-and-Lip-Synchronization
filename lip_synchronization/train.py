from os.path import dirname, join, basename
from tqdm import tqdm

from inf_test import parse_filelist
from models.talklip import TalkLip, TalkLip_disc_qual

import torch
import logging
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch import optim
from argparse import Namespace
from torch.utils.data import DataLoader
from python_speech_features import logfbank
from fairseq.data import data_utils
from fairseq import checkpoint_utils, utils, tasks
from fairseq.dataclass.utils import convert_namespace_to_omegaconf, populate_dataclass, merge_with_parent
from scipy.io import wavfile
from utils.data_avhubert import collater_audio, images2avhubert

import os, random, cv2, argparse, subprocess
import multiprocessing

# --- FIXED FOR WINDOWS MULTIPROCESSING ---
# Metrics ko global scope mein handle karne ke liye
try:
    from fairseq import metrics
except ImportError:
    # Agar pip install -e . nahi kiya toh ye manually path add karega
    import sys
    # Aapka av_hubert root yahan automatic add hona chahiye command line se
    pass

def init_logging(level=logging.INFO,
                 log_name='log/sys.log',
                 formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')):
    logger = logging.getLogger()
    logger.setLevel(level=level)
    handler = logging.FileHandler(log_name)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def build_encoder(hubert_root, cfg):
    import sys
    sys.path.append(hubert_root)
    from avhubert.hubert_asr import HubertEncoderWrapper, AVHubertSeq2SeqConfig

    cfg = merge_with_parent(AVHubertSeq2SeqConfig(), cfg)
    arg_overrides = {
        "dropout": cfg.dropout,
        "activation_dropout": cfg.activation_dropout,
        "dropout_input": cfg.dropout_input,
        "attention_dropout": cfg.attention_dropout,
        "mask_length": cfg.mask_length,
        "mask_prob": cfg.mask_prob,
        "mask_selection": cfg.mask_selection,
        "mask_other": cfg.mask_other,
        "no_mask_overlap": cfg.no_mask_overlap,
        "mask_channel_length": cfg.mask_channel_length,
        "mask_channel_prob": cfg.mask_channel_prob,
        "mask_channel_selection": cfg.mask_channel_selection,
        "mask_channel_other": cfg.mask_channel_other,
        "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
        "encoder_layerdrop": cfg.layerdrop,
        "feature_grad_mult": cfg.feature_grad_mult,
    }
    if cfg.w2v_args is None:
        state = checkpoint_utils.load_checkpoint_to_cpu(
            cfg.w2v_path, arg_overrides
        )
        w2v_args = state.get("cfg", None)
        if w2v_args is None:
            w2v_args = convert_namespace_to_omegaconf(state["args"])
        cfg.w2v_args = w2v_args
    else:
        state = None
        w2v_args = cfg.w2v_args
        if isinstance(w2v_args, Namespace):
            cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(
                w2v_args
            )

    w2v_args.task.data = cfg.data
    task_pretrain = tasks.setup_task(w2v_args.task)
    if state is not None:
        task_pretrain.load_state_dict(state['task_state'])
    # task_pretrain.state = task.state

    encoder_ = task_pretrain.build_model(w2v_args.model)
    encoder = HubertEncoderWrapper(encoder_)
    if state is not None and not cfg.no_pretrained_weights:
        # set strict=False because we omit some modules
        del state['model']['mask_emb']
        encoder.w2v_model.load_state_dict(state["model"], strict=False)

    encoder.w2v_model.remove_pretraining_modules()
    return encoder


def get_avhubert(hubert_root, ckptpath):

    import sys
    sys.path.append(hubert_root)
    from avhubert.hubert_pretraining import LabelEncoderS2SToken
    from fairseq.dataclass.utils import DictConfig

    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckptpath])
    criterion = task.build_criterion(saved_cfg.criterion)
    criterion.report_accuracy = True

    dictionaries = [task.target_dictionary]
    bpe_tokenizer = task.s2s_tokenizer
    procs = [LabelEncoderS2SToken(dictionary, bpe_tokenizer) for dictionary in dictionaries]
    extra_gen_cls_kwargs = {
        "lm_model": None,
        "lm_weight": 0.0,
    }
    arg_gen = DictConfig({'_name': None, 'beam': 50, 'nbest': 1, 'max_len_a': 1.0, 'max_len_b': 0, 'min_len': 1,
                          'match_source_len': False, 'unnormalized': False, 'no_early_stop': False,
                          'no_beamable_mm': False, 'lenpen': 1.0, 'unkpen': 0.0, 'replace_unk': None,
                          'sacrebleu': False, 'score_reference': False, 'prefix_size': 0, 'no_repeat_ngram_size': 0,
                          'sampling': False, 'sampling_topk': -1, 'sampling_topp': -1.0, 'constraints': None,
                          'temperature': 1.0, 'diverse_beam_groups': -1, 'diverse_beam_strength': 0.5,
                          'diversity_rate': -1.0, 'print_alignment': None, 'print_step': False, 'lm_path': None,
                          'lm_weight': 0.0, 'iter_decode_eos_penalty': 0.0, 'iter_decode_max_iter': 10,
                          'iter_decode_force_max_iter': False, 'iter_decode_with_beam': 1,
                          'iter_decode_with_external_reranker': False, 'retain_iter_history': False,
                          'retain_dropout': False, 'retain_dropout_modules': None, 'decoding_format': None,
                          'no_seed_provided': False})
    generator = task.build_generator(
        models, arg_gen, extra_gen_cls_kwargs=extra_gen_cls_kwargs
    )
    encoder = build_encoder(hubert_root, saved_cfg.model)
    model_dict_avhubert = models[0].state_dict()
    model_dict_encoder = encoder.state_dict()
    for key in model_dict_encoder.keys():
        model_dict_encoder[key] = model_dict_avhubert['encoder.'+key]
    encoder.load_state_dict(model_dict_encoder)
    return models[0], procs[0], generator, criterion, encoder


def retrieve_avhubert(hubert_root, hubert_ckpt, device):
    avhubert, label_proc, generator, criterion, encoder = get_avhubert(hubert_root, hubert_ckpt)
    """Base configuration"""
    ftlayers = list(range(9, 12))

    ftlayers_full = ['w2v_model.encoder.layers.'+str(layer) for layer in ftlayers]
    for name, p in encoder.named_parameters():
        ft_ind = False
        for layer in ftlayers_full:
            if layer in name:
                ft_ind = True
                break
        if ft_ind:
            p.requires_grad = True
        else:
            p.requires_grad = False

    for p in avhubert.parameters():
        p.requires_grad = False
    avhubert = avhubert.to(device)
    avhubert.eval()
    return avhubert, label_proc, generator, criterion, encoder


class Talklipdata(object):
    def __init__(self, split, args, label_proc):
        self.data_root = args.video_root
        self.bbx_root = args.bbx_root
        self.audio_root = args.audio_root
        self.text_root = args.word_root
        self.label_proc = label_proc
        self.datalists = parse_filelist('{}/{}.txt'.format(args.file_dir, split), None, False)
        self.stack_order_audio = 4
        self.train = (split == 'train')
        self.args = args
        self.crop_size = 96
        self.prob = 0.08
        self.length = 5

    def readtext(self, path):
        with open(path, "r") as f:
            trgt = f.readline()[7:]
        trgt = self.label_proc(trgt)
        return trgt

    def im_preprocess(self, ims):
        # T x 3 x H x W
        x = ims / 255.
        x = x.permute((0, 3, 1, 2))

        return x

    def filter_start_id(self, idlist):
        idlist = sorted(idlist)
        filtered = [idlist[0]]
        for item in idlist:
            if item - filtered[-1] > 4:
                filtered.append(item)
        return filtered

    def croppatch(self, images, bbxs):
        patch = np.zeros((images.shape[0], 96, 96, 3), dtype=np.float32)
        width = images.shape[1]
        for i, bbx in enumerate(bbxs):
            bbx[2] = min(bbx[2], width)
            bbx[3] = min(bbx[3], width)
            patch[i] = cv2.resize(images[i, bbx[1]:bbx[3], bbx[0]:bbx[2], :], (self.crop_size, self.crop_size))
        return patch

    def audio_visual_align(self, audio_feats, video_feats):
        diff = len(audio_feats) - len(video_feats)
        if diff < 0:
            audio_feats = torch.cat(
                [audio_feats, torch.zeros([-diff, audio_feats.shape[-1]], dtype=audio_feats.dtype)])
        elif diff > 0:
            audio_feats = audio_feats[:-diff]
        return audio_feats

    def fre_audio(self, wav_data, sample_rate):
        def stacker(feats, stack_order):
            """
            Concatenating consecutive audio frames, 4 frames of tf forms a new frame of tf
            Args:
            feats - numpy.ndarray of shape [T, F]
            stack_order - int (number of neighboring frames to concatenate
            Returns:
            feats - numpy.ndarray of shape [T', F']
            """
            feat_dim = feats.shape[1]
            if len(feats) % stack_order != 0:
                res = stack_order - len(feats) % stack_order
                res = np.zeros([res, feat_dim]).astype(feats.dtype)
                feats = np.concatenate([feats, res], axis=0)
            feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order*feat_dim)
            return feats

        audio_feats = logfbank(wav_data, samplerate=sample_rate).astype(np.float32)  # [T, F]
        audio_feats = stacker(audio_feats, self.stack_order_audio)  # [T/stack_order_audio, F*stack_order_audio]
        return audio_feats

    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        imgs = []
        while True:
            ret, frame = cap.read()
            if ret:
                imgs.append(frame)
            else:
                break
        cap.release()
        return imgs

    def __len__(self):
        return len(self.datalists)

    def __getitem__(self, idx):
        while True:
            sample = self.datalists[idx]

            video_path = '{}/{}.mp4'.format(self.data_root, sample)
            bbx_path = '{}/{}.npy'.format(self.bbx_root, sample)
            wav_path = '{}/{}.wav'.format(self.audio_root, sample)
            word_path = '{}/{}.txt'.format(self.text_root, sample)

            try:
                # Agar koi file corrupt ya missing hui toh ye lines error dengi
                bbxs = np.load(bbx_path)
                imgs = np.array(self.load_video(video_path)) # T*96*96*3
                volume = len(imgs)
                
                if volume == 0:
                    raise FileNotFoundError("Video empty hai")

                sampRate, wav = wavfile.read(wav_path)
                trgt = self.readtext(word_path)
                
                # Agar sab successfully load ho gaya toh loop se bahar nikal jao
                break
                
            except Exception as e:
                # Error aane par automatic doosra random index select ho jayega (Auto-skip)
                idx = random.randint(0, len(self.datalists) - 1)
                continue

        # --- Baqi saara processing isi valid sample par hoga ---
        spectrogram = self.fre_audio(wav, sampRate)
        spectrogram = torch.from_numpy(spectrogram)  # T'* F, T'*104
        with torch.no_grad():
            spectrogram = F.layer_norm(spectrogram, spectrogram.shape[1:])

        if self.train:
            pid_start = random.sample(list(range(1, volume-4)), max(int(volume * self.prob), 1))
        else:
            pid_start = list(range(0, volume-4, int(volume * 0.12)))

        pid_start = self.filter_start_id(pid_start)
        pid_start = np.array(pid_start)

        poseidx, ididx = [], []

        for i, index in enumerate(pid_start):
            poseidx += list(range(index, index+self.length))
            wrongindex = random.choice(list(range(volume-4)))
            while wrongindex == index:
                wrongindex = random.choice(list(range(volume-4)))
            ididx += list(range(wrongindex, wrongindex+self.length))

        if not self.train:
            ididx = np.zeros(len(poseidx), dtype=np.int32)

        # (N*5)
        pickedimg = poseidx

        poseImg = self.croppatch(imgs[poseidx], bbxs[poseidx])
        idImg = self.croppatch(imgs[ididx], bbxs[ididx])

        poseImg = torch.from_numpy(poseImg)
        idImg = torch.from_numpy(idImg)

        spectrogram = self.audio_visual_align(spectrogram, imgs)

        poseImg = self.im_preprocess(poseImg)
        gtImg = poseImg.clone()
        # mask off the bottom half
        poseImg[:, :, poseImg.shape[2] // 2:] = 0.

        idImg = self.im_preprocess(idImg)
        inpImg = torch.cat([poseImg, idImg], axis=1)

        pickedimg = torch.tensor(pickedimg)

        return inpImg, spectrogram, gtImg, trgt, volume, pickedimg, torch.from_numpy(imgs), torch.from_numpy(bbxs)


def collater_seq_label_s2s(targets):
    lengths = torch.LongTensor([len(t) for t in targets])
    ntokens = lengths.sum().item()
    pad, eos = 1, 2
    targets_ = data_utils.collate_tokens(targets, pad_idx=pad, eos_idx=eos, left_pad=False)
    prev_output_tokens = data_utils.collate_tokens(targets, pad_idx=pad, eos_idx=eos, left_pad=False, move_eos_to_beginning=True)
    return (targets_, prev_output_tokens), lengths, ntokens


def collater_label(targets_by_label):
    targets_list, lengths_list, ntokens_list = [], [], []
    itr = zip(targets_by_label, [-1], [1])
    for targets, label_rate, pad in itr:
        if label_rate == -1:
            targets, lengths, ntokens = collater_seq_label_s2s(targets)
        targets_list.append(targets)
        lengths_list.append(lengths)
        ntokens_list.append(ntokens)
    return targets_list[0], lengths_list[0], ntokens_list[0]


def collate_fn(dataBatch):
    """
    Args:
        dataBatch:

    Returns:
        inpBatch: input T_sum*6*96*96, concatenation of all video chips in the time dimension
        gtBatch: output T_sum*3*96*96
        inputLenBatch: bs
        audioBatch: bs*104*T'
        audio_idx: T_sum
        targetBatch: words for lip-reading expert
        padding_mask: bs*T'
        pickedimg: a list of bs elements, each contain some picked indices
        videoBatch: a list of bs elements, each cotain a video
        bbxs: a list of bs elements
    """

    inpBatch = torch.cat([data[0] for data in dataBatch], dim=0)
    gtBatch = torch.cat([data[2] for data in dataBatch], dim=0)
    inputLenBatch = [data[4] for data in dataBatch]

    audioBatch, padding_mask = collater_audio([data[1] for data in dataBatch], max(inputLenBatch))
    audio_idx = torch.cat([data[5] + audioBatch.shape[2] * i for i, data in enumerate(dataBatch)], dim=0)

    targetBatch = collater_label([[data[3] for data in dataBatch]])

    bbxs = [data[7] for data in dataBatch]
    pickedimg = [data[5] for data in dataBatch]
    videoBatch = [data[6] for data in dataBatch]

    return inpBatch, audioBatch, audio_idx, gtBatch, targetBatch, padding_mask, pickedimg, videoBatch, bbxs


def save_sample_images(x, g, gt, global_step, checkpoint_dir):
    x = (x.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.).astype(np.uint8)
    g = (g.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.).astype(np.uint8)
    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.).astype(np.uint8)

    refs, inps = x[..., 3:], x[..., :3]
    folder = join(checkpoint_dir, "samples_step{:09d}".format(global_step))
    if not os.path.exists(folder): os.mkdir(folder)
    collage = np.concatenate((refs, inps, g, gt), axis=-2)
    for batch_idx, c in enumerate(collage):
            cv2.imwrite('{}/{}.jpg'.format(folder, batch_idx), c)


def get_gpu_memory_map():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def local_sync_loss(pickid, enc_audio, enc_video):
    pickedAud = enc_audio.permute(1, 0, 2).reshape(-1, enc_audio.shape[2])[pickid]
    pickedVid = enc_video.permute(1, 0, 2).reshape(-1, enc_video.shape[2])[pickid]
    return pickedVid, pickedAud


class status_manager(object):
    def __init__(self, patience=5, status=0):
        self.min = 100.
        self.waited_itr = 0
        self.status = status
        self.patience = patience

    def update(self, performance):
        if performance < self.min:
            self.min = performance
            self.waited_itr = 0
        else:
            self.waited_itr += 1

    def check_status(self):
        if self.waited_itr > self.patience:
            self.status += 1
            self.waited_itr = 0
            return self.status, True
        else:
            return self.status, False

def train(device, model, avhubert, criterion, data_loader, optimizer, args, global_step, logger):
    # RTX 4080 optimization: Enables the built-in cudnn auto-tuner
    torch.backends.cudnn.benchmark = True
    
    # Mixed Precision Scaler for RTX 4080 Tensor Cores
    scaler = torch.cuda.amp.GradScaler()
    
    print('Starting Step: {}'.format(global_step))

    lip_train = False
    model['gen'].ft = False
    status = status_manager(5)
    recon_loss = nn.L1Loss()

    for epoch in range(args.n_epoch):
        losses = {'lip': 0, 'local_sync': 0, 'l1': 0, 'prec_g': 0, 'disc_real_g': 0, 'disc_fake_g': 0}
        prog_bar = tqdm(enumerate(data_loader['train']), total=len(data_loader['train']))
        oom_count = 0
        
        for step, (inpim, spectrogram, audio_idx, gtim, ((trgt, prev_trg), tlen, ntoken), padding_mask, vidx, videos, bbxs) in prog_bar:
            for key in model.keys():
                model[key].train()

            criterion.report_accuracy = False

            # Transfer to GPU
            inpim, gtim = inpim.to(device, non_blocking=True), gtim.to(device, non_blocking=True)
            trgt, prev_trg = trgt.to(device, non_blocking=True), prev_trg.to(device, non_blocking=True)
            spectrogram, padding_mask = spectrogram.to(device, non_blocking=True), padding_mask.to(device, non_blocking=True)

            for key in optimizer.keys():
                optimizer[key].zero_grad(set_to_none=True) # Optimized zero_grad

            net_input = {'source': {'audio': spectrogram, 'video': None}, 'padding_mask': padding_mask, 'prev_output_tokens': prev_trg}
            sample = {'net_input': net_input, 'target_lengths': tlen, 'ntokens': ntoken, 'target': trgt}

            try:
                # --- GENERATOR TRAINING WITH AMP ---
                with torch.cuda.amp.autocast():
                    syntim, enc_audio = model['gen'](sample, inpim, audio_idx, spectrogram.shape[0])

                    if lip_train:
                        processed_img = images2avhubert(vidx, videos, bbxs, syntim, spectrogram.shape[2], device)
                        sample['net_input']['source']['video'] = processed_img
                        sample['net_input']['source']['audio'] = None
                        lip_loss, sample_size, logs, enc_out = criterion(avhubert, sample)
                        
                        if args.cont_w > 0:
                            pickedVid, pickedAud = local_sync_loss(audio_idx, enc_audio, enc_out['encoder_out'])
                            local_sync = model['gen'].sync_net(pickedVid, pickedAud)
                        else:
                            local_sync = torch.tensor(0., device=device)
                    else:
                        lip_loss, local_sync = torch.tensor(0., device=device), torch.tensor(0., device=device)

                    if args.perp_w > 0.:
                        perceptual_loss = model['disc'].perceptual_forward(syntim)
                    else:
                        perceptual_loss = torch.tensor(0., device=device)

                    l1loss = recon_loss(syntim, gtim)
                    
                    # Total Loss Calculation
                    loss = args.lip_w * lip_loss + args.perp_w * perceptual_loss + \
                           (1. - args.lip_w - args.perp_w) * l1loss + args.cont_w * local_sync

                # Scaled Backward Pass
                scaler.scale(loss).backward()
                scaler.step(optimizer['gen'])

                # --- DISCRIMINATOR TRAINING ---
                optimizer['gen'].zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=False):

                    gt_float = gtim.float()
                    syn_float = syntim.detach().float()

                    pred_real = model['disc'](gt_float)
                    disc_real_loss = F.binary_cross_entropy(pred_real, torch.ones((len(pred_real), 1), device=device))
                    
                    pred_fake = model['disc'](syn_float)
                    disc_fake_loss = F.binary_cross_entropy(pred_fake, torch.zeros((len(pred_fake), 1), device=device))
                    
                    disc_loss = disc_real_loss + disc_fake_loss

                scaler.scale(disc_loss).backward()
                scaler.step(optimizer['disc'])
                
                # Update Scaler for next iteration
                scaler.update()

                # Logging losses
                losses['lip'] += lip_loss.item()
                losses['local_sync'] += local_sync.item()
                losses['l1'] += l1loss.item()
                losses['prec_g'] += perceptual_loss.item()
                losses['disc_real_g'] += disc_real_loss.item()
                losses['disc_fake_g'] += disc_fake_loss.item()

                if global_step % args.ckpt_interval == 0:
                    save_sample_images(inpim, syntim, gtim, global_step, args.checkpoint_dir)

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"| Step {global_step} | CUDA OOM. Reducing cache...")
                    torch.cuda.empty_cache()
                    oom_count += 1
                    continue
                else:
                    raise e

            global_step += 1

            if global_step == 1 or global_step % args.ckpt_interval == 0:
                save_checkpoint(model['gen'], optimizer['gen'], global_step, args.checkpoint_dir, epoch)
                save_checkpoint(model['disc'], optimizer['disc'], global_step, args.checkpoint_dir, epoch, prefix='disc_')

            # Performance Monitor in TQDM
            if step % 5 == 0:
                vram_used = torch.cuda.memory_reserved(device) / 1e9
                prog_bar.set_description(f"Epoch {epoch} | L1: {l1loss.item():.4f} | VRAM: {vram_used:.1f}GB")

            if global_step % args.ckpt_interval == 0:
                with torch.no_grad():
                    average_sync_loss, valid_log = eval_model(data_loader['test'], avhubert, criterion, global_step, device, model['gen'], model['disc'], args.cont_w, recon_loss)
                    
                    logger.info(f"Step {global_step} Train Log: {losses}")
                    logger.info(valid_log)

                    status.update(average_sync_loss)
                    stage, changed = status.check_status()

                    if changed:
                        if stage == 1:
                            model['gen'].ft = True
                            logger.info('>>> Audio encoder start to finetune')
                        elif stage == 2:
                            lip_train = True
                            logger.info('>>> Lip reading start to work')
                        elif stage == 3:
                            logger.info('>>> Training Completed Successfully')
                            return 

    logger.info('Total OOM cases: {}'.format(oom_count))


def eval_model(test_data_loader, avhubert, criterion, global_step, device, model, disc, cont_w, recon_loss):
    print('Evaluating after training of {} steps'.format(global_step))
    n_correct, n_total = 0, 0
    losses = {'lip': 0, 'local_sync': 0, 'l1': 0, 'prec_g': 0, 'disc_real_g': 0, 'disc_fake_g': 0}

    for step, (x, spectrogram, audio_idx, gt, ((trgt, prev_trg), tlen, ntoken), padding_mask, vidx, videos, bbxs) in enumerate((test_data_loader)):
        model.eval()
        disc.eval()
        criterion.report_accuracy = True

        x = x.to(device)
        spectrogram = spectrogram.to(device)
        gt = gt.to(device)
        trgt, prev_trg = trgt.to(device), prev_trg.to(device)
        padding_mask = padding_mask.to(device)

        sample = {'net_input': {'source': {'audio': spectrogram, 'video': None}, 'padding_mask': padding_mask, 'prev_output_tokens': prev_trg},
                  'target_lengths': tlen, 'ntokens': ntoken, 'target': trgt}

        g, enc_audio = model(sample, x, audio_idx, spectrogram.shape[0])

        pred = disc(gt)
        disc_real_loss = F.binary_cross_entropy(pred, torch.ones((len(pred), 1)).to(device))
        losses['disc_real_g'] += disc_real_loss.item()
        pred = disc(g)
        disc_fake_loss = F.binary_cross_entropy(pred, torch.zeros((len(pred), 1)).to(device))
        losses['disc_fake_g'] += disc_fake_loss.item()

        processed_img = images2avhubert(vidx, videos, bbxs, g, spectrogram.shape[2], device)
        sample['net_input']['source']['video'] = processed_img
        sample['net_input']['source']['audio'] = None

        lip_loss, sample_size, logs, enc_out = criterion(avhubert, sample)
        losses['lip'] += lip_loss.item()

        if cont_w > 0:
            pickedVid, pickedAud = local_sync_loss(audio_idx, enc_audio, enc_out['encoder_out'])
            local_sync = model.sync_net(pickedVid, pickedAud)
            losses['local_sync'] += local_sync.item()

        n_correct += logs['n_correct']
        n_total += logs['total']
        if args.perp_w > 0.:
            perceptual_loss = disc.perceptual_forward(g)
            losses['prec_g'] += perceptual_loss.item()

        l1loss = recon_loss(g, gt)
        losses['l1'] += l1loss.item()

    if n_total > 0:
        avewer = 1 - n_correct / n_total
    else:
        avewer = 1.0

    valid_log = 'Valid step: {} '.format(global_step)
    for key, value in losses.items():
        valid_log += '{}: {:.4f} '.format(key, value / (step + 1))
    valid_log += '| wer: {}'.format(avewer)
    print(valid_log)
    return losses['l1'], valid_log


def save_checkpoint(model, optimizer, global_step, checkpoint_dir, epoch, prefix=''):
    checkpoint_path = join(
        checkpoint_dir, "{}checkpoint_step{:09d}.pth".format(prefix, global_step))
    optimizer_state = optimizer.state_dict() if optimizer is not None else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": global_step,
        "global_epoch": epoch,
    }, checkpoint_path)

    print("Saved checkpoint:", checkpoint_path)


def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint(path, model, optimizer, logger, reset_optimizer=False, overwrite_global_states=True):

    print("Load checkpoint from: {}".format(path))
    logger.info("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = model.state_dict()
    for k, v in s.items():
        new_s[k] = v
    model.load_state_dict(new_s)

    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            logger.info("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    if overwrite_global_states:
        global_step = checkpoint["global_step"]
    else:
        global_step = 0

    return global_step


if __name__ == "__main__":
    # Windows par multiprocessing spawn error se bachne ke liye zaroori hai
    import multiprocessing
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(
        description='TalkLip Training - RTX 4080 Optimized')

    # --- Dataset Path Arguments ---
    parser.add_argument("--video_root", help="Root folder of video", required=True, type=str)
    parser.add_argument("--audio_root", help="Root folder of audio", required=True, type=str)
    parser.add_argument("--word_root", help="Root folder of audio", required=True, type=str)
    parser.add_argument('--bbx_root', help="Root folder of bounding boxes of faces", required=True, type=str)
    parser.add_argument("--file_dir", help="Root folder of filelists", required=True, type=str)
    
    # RTX 4080 (16GB VRAM) optimized defaults
    parser.add_argument('--batch_size', help='batch size of training', default=32, type=int)
    parser.add_argument('--num_worker', help='number of worker', default=0, type=int)

    # --- Checkpoint Arguments ---
    parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)
    parser.add_argument('--gen_checkpoint_path', help='Resume generator from this checkpoint', default=None, type=str)
    parser.add_argument('--disc_checkpoint_path', help='Resume discriminator from this checkpoint', default=None, type=str)
    parser.add_argument('--avhubert_path', help='Resume avhubert from this checkpoint', default=None, type=str)
    parser.add_argument('--avhubert_root', help='Path of av_hubert root', required=True, type=str)

    # --- Optimizer & Loss Arguments ---
    parser.add_argument('--lr', help='learning rate', default=1e-4, type=float)
    parser.add_argument('--lip_w', help='weight of lip-reading expert', default=1e-5, type=float)
    parser.add_argument('--cont_w', help='weight of contrastive learning', default=1e-3, type=float)
    parser.add_argument('--perp_w', help='weight of perceptual loss', default=0.07, type=float)

    # --- Training Config ---
    parser.add_argument('--gpu', help='index of gpu used', default=0, type=int)
    parser.add_argument('--n_epoch', help='number of epoch', default=100, type=int)
    parser.add_argument('--log_name', help='name of a log file', default='talklip', type=str)
    parser.add_argument('--ckpt_interval', help='The interval of saving a checkpoint', default=3000, type=int)

    args = parser.parse_args()

    # --- Device Setup ---
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")
    print(f"Status: Using {torch.cuda.get_device_name(args.gpu)} | VRAM: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")

    # --- AVHubert Loading ---
    avhubert, label_proc, generator, criterion, encoder = retrieve_avhubert(args.avhubert_root, args.avhubert_path, device)

    # --- Dataset & DataLoader (64GB RAM Optimized) ---
    train_dataset = Talklipdata('train', args, label_proc)
    test_dataset = Talklipdata('valid', args, label_proc)

    # pin_memory=True data loading ko GPU ke liye fast bana deta hai
    # train.py ke andar DataLoader wala hissa aise hona chahiye:
    train_data_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=args.num_worker,
        pin_memory=True, # RTX 4080 ke liye lazmi hai
        persistent_workers=True if args.num_worker > 0 else False # Speed boost
    )

    test_data_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        collate_fn=collate_fn, 
        num_workers=args.num_worker,
        pin_memory=True if use_cuda else False
    )

    # --- Model Initialization ---
    # TalkLip model 768-dim encoder ke sath
    imGen = TalkLip(encoder, 768).to(device)
    imDisc = TalkLip_disc_qual().to(device)

from os.path import dirname, join, basename
from tqdm import tqdm

from inf_test import parse_filelist
from models.talklip import TalkLip, TalkLip_disc_qual

import torch
import logging
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch import optim
from argparse import Namespace
from torch.utils.data import DataLoader
from python_speech_features import logfbank
from fairseq.data import data_utils
from fairseq import checkpoint_utils, utils, tasks
from fairseq.dataclass.utils import convert_namespace_to_omegaconf, populate_dataclass, merge_with_parent
from scipy.io import wavfile
from utils.data_avhubert import collater_audio, images2avhubert

import os, random, cv2, argparse, subprocess
import multiprocessing

# --- FIXED FOR WINDOWS MULTIPROCESSING ---
# Metrics ko global scope mein handle karne ke liye
try:
    from fairseq import metrics
except ImportError:
    # Agar pip install -e . nahi kiya toh ye manually path add karega
    import sys
    # Aapka av_hubert root yahan automatic add hona chahiye command line se
    pass

def init_logging(level=logging.INFO,
                 log_name='log/sys.log',
                 formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')):
    logger = logging.getLogger()
    logger.setLevel(level=level)
    handler = logging.FileHandler(log_name)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def build_encoder(hubert_root, cfg):
    import sys
    sys.path.append(hubert_root)
    from avhubert.hubert_asr import HubertEncoderWrapper, AVHubertSeq2SeqConfig

    cfg = merge_with_parent(AVHubertSeq2SeqConfig(), cfg)
    arg_overrides = {
        "dropout": cfg.dropout,
        "activation_dropout": cfg.activation_dropout,
        "dropout_input": cfg.dropout_input,
        "attention_dropout": cfg.attention_dropout,
        "mask_length": cfg.mask_length,
        "mask_prob": cfg.mask_prob,
        "mask_selection": cfg.mask_selection,
        "mask_other": cfg.mask_other,
        "no_mask_overlap": cfg.no_mask_overlap,
        "mask_channel_length": cfg.mask_channel_length,
        "mask_channel_prob": cfg.mask_channel_prob,
        "mask_channel_selection": cfg.mask_channel_selection,
        "mask_channel_other": cfg.mask_channel_other,
        "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
        "encoder_layerdrop": cfg.layerdrop,
        "feature_grad_mult": cfg.feature_grad_mult,
    }
    if cfg.w2v_args is None:
        state = checkpoint_utils.load_checkpoint_to_cpu(
            cfg.w2v_path, arg_overrides
        )
        w2v_args = state.get("cfg", None)
        if w2v_args is None:
            w2v_args = convert_namespace_to_omegaconf(state["args"])
        cfg.w2v_args = w2v_args
    else:
        state = None
        w2v_args = cfg.w2v_args
        if isinstance(w2v_args, Namespace):
            cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(
                w2v_args
            )

    w2v_args.task.data = cfg.data
    task_pretrain = tasks.setup_task(w2v_args.task)
    if state is not None:
        task_pretrain.load_state_dict(state['task_state'])
    # task_pretrain.state = task.state

    encoder_ = task_pretrain.build_model(w2v_args.model)
    encoder = HubertEncoderWrapper(encoder_)
    if state is not None and not cfg.no_pretrained_weights:
        # set strict=False because we omit some modules
        del state['model']['mask_emb']
        encoder.w2v_model.load_state_dict(state["model"], strict=False)

    encoder.w2v_model.remove_pretraining_modules()
    return encoder


def get_avhubert(hubert_root, ckptpath):

    import sys
    sys.path.append(hubert_root)
    from avhubert.hubert_pretraining import LabelEncoderS2SToken
    from fairseq.dataclass.utils import DictConfig

    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckptpath])
    criterion = task.build_criterion(saved_cfg.criterion)
    criterion.report_accuracy = True

    dictionaries = [task.target_dictionary]
    bpe_tokenizer = task.s2s_tokenizer
    procs = [LabelEncoderS2SToken(dictionary, bpe_tokenizer) for dictionary in dictionaries]
    extra_gen_cls_kwargs = {
        "lm_model": None,
        "lm_weight": 0.0,
    }
    arg_gen = DictConfig({'_name': None, 'beam': 50, 'nbest': 1, 'max_len_a': 1.0, 'max_len_b': 0, 'min_len': 1,
                          'match_source_len': False, 'unnormalized': False, 'no_early_stop': False,
                          'no_beamable_mm': False, 'lenpen': 1.0, 'unkpen': 0.0, 'replace_unk': None,
                          'sacrebleu': False, 'score_reference': False, 'prefix_size': 0, 'no_repeat_ngram_size': 0,
                          'sampling': False, 'sampling_topk': -1, 'sampling_topp': -1.0, 'constraints': None,
                          'temperature': 1.0, 'diverse_beam_groups': -1, 'diverse_beam_strength': 0.5,
                          'diversity_rate': -1.0, 'print_alignment': None, 'print_step': False, 'lm_path': None,
                          'lm_weight': 0.0, 'iter_decode_eos_penalty': 0.0, 'iter_decode_max_iter': 10,
                          'iter_decode_force_max_iter': False, 'iter_decode_with_beam': 1,
                          'iter_decode_with_external_reranker': False, 'retain_iter_history': False,
                          'retain_dropout': False, 'retain_dropout_modules': None, 'decoding_format': None,
                          'no_seed_provided': False})
    generator = task.build_generator(
        models, arg_gen, extra_gen_cls_kwargs=extra_gen_cls_kwargs
    )
    encoder = build_encoder(hubert_root, saved_cfg.model)
    model_dict_avhubert = models[0].state_dict()
    model_dict_encoder = encoder.state_dict()
    for key in model_dict_encoder.keys():
        model_dict_encoder[key] = model_dict_avhubert['encoder.'+key]
    encoder.load_state_dict(model_dict_encoder)
    return models[0], procs[0], generator, criterion, encoder


def retrieve_avhubert(hubert_root, hubert_ckpt, device):
    avhubert, label_proc, generator, criterion, encoder = get_avhubert(hubert_root, hubert_ckpt)
    """Base configuration"""
    ftlayers = list(range(9, 12))

    ftlayers_full = ['w2v_model.encoder.layers.'+str(layer) for layer in ftlayers]
    for name, p in encoder.named_parameters():
        ft_ind = False
        for layer in ftlayers_full:
            if layer in name:
                ft_ind = True
                break
        if ft_ind:
            p.requires_grad = True
        else:
            p.requires_grad = False

    for p in avhubert.parameters():
        p.requires_grad = False
    avhubert = avhubert.to(device)
    avhubert.eval()
    return avhubert, label_proc, generator, criterion, encoder


class Talklipdata(object):
    def __init__(self, split, args, label_proc):
        self.data_root = args.video_root
        self.bbx_root = args.bbx_root
        self.audio_root = args.audio_root
        self.text_root = args.word_root
        self.label_proc = label_proc
        self.datalists = parse_filelist('{}/{}.txt'.format(args.file_dir, split), None, False)
        self.stack_order_audio = 4
        self.train = (split == 'train')
        self.args = args
        self.crop_size = 96
        self.prob = 0.08
        self.length = 5

    def readtext(self, path):
        with open(path, "r") as f:
            trgt = f.readline()[7:]
        trgt = self.label_proc(trgt)
        return trgt

    def im_preprocess(self, ims):
        # T x 3 x H x W
        x = ims / 255.
        x = x.permute((0, 3, 1, 2))

        return x

    def filter_start_id(self, idlist):
        idlist = sorted(idlist)
        filtered = [idlist[0]]
        for item in idlist:
            if item - filtered[-1] > 4:
                filtered.append(item)
        return filtered

    def croppatch(self, images, bbxs):
        patch = np.zeros((images.shape[0], 96, 96, 3), dtype=np.float32)
        width = images.shape[1]
        for i, bbx in enumerate(bbxs):
            bbx[2] = min(bbx[2], width)
            bbx[3] = min(bbx[3], width)
            patch[i] = cv2.resize(images[i, bbx[1]:bbx[3], bbx[0]:bbx[2], :], (self.crop_size, self.crop_size))
        return patch

    def audio_visual_align(self, audio_feats, video_feats):
        diff = len(audio_feats) - len(video_feats)
        if diff < 0:
            audio_feats = torch.cat(
                [audio_feats, torch.zeros([-diff, audio_feats.shape[-1]], dtype=audio_feats.dtype)])
        elif diff > 0:
            audio_feats = audio_feats[:-diff]
        return audio_feats

    def fre_audio(self, wav_data, sample_rate):
        def stacker(feats, stack_order):
            """
            Concatenating consecutive audio frames, 4 frames of tf forms a new frame of tf
            Args:
            feats - numpy.ndarray of shape [T, F]
            stack_order - int (number of neighboring frames to concatenate
            Returns:
            feats - numpy.ndarray of shape [T', F']
            """
            feat_dim = feats.shape[1]
            if len(feats) % stack_order != 0:
                res = stack_order - len(feats) % stack_order
                res = np.zeros([res, feat_dim]).astype(feats.dtype)
                feats = np.concatenate([feats, res], axis=0)
            feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order*feat_dim)
            return feats

        audio_feats = logfbank(wav_data, samplerate=sample_rate).astype(np.float32)  # [T, F]
        audio_feats = stacker(audio_feats, self.stack_order_audio)  # [T/stack_order_audio, F*stack_order_audio]
        return audio_feats

    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        imgs = []
        while True:
            ret, frame = cap.read()
            if ret:
                imgs.append(frame)
            else:
                break
        cap.release()
        return imgs

    def __len__(self):
        return len(self.datalists)

    def __getitem__(self, idx):
        """

        Args:
            idx: index of a sample in dataset

        Returns:
            inpImg: N*6*96*96
            gtImg: N*3*96*96
            spectrogram: T*104
            trgt: L
            volume: 1, which indicates T
            pickedimg: N
            imgs: T*160*160*3
            bbxs: T*4
        """
        sample = self.datalists[idx]

        video_path = '{}/{}.mp4'.format(self.data_root, sample)
        bbx_path = '{}/{}.npy'.format(self.bbx_root, sample)
        wav_path = '{}/{}.wav'.format(self.audio_root, sample)
        word_path = '{}/{}.txt'.format(self.text_root, sample)

        bbxs = np.load(bbx_path)
        imgs = np.array(self.load_video(video_path)) # T*96*96*3
        volume = len(imgs)

        sampRate, wav = wavfile.read(wav_path)
        spectrogram = self.fre_audio(wav, sampRate)
        spectrogram = torch.from_numpy(spectrogram)  # T'* F, T'*104
        with torch.no_grad():
            spectrogram = F.layer_norm(spectrogram, spectrogram.shape[1:])

        if self.train:
            pid_start = random.sample(list(range(1, volume-4)), max(int(volume * self.prob), 1))
        else:
            pid_start = list(range(0, volume-4, int(volume * 0.12)))

        pid_start = self.filter_start_id(pid_start)
        pid_start = np.array(pid_start)

        poseidx, ididx = [], []

        for i, index in enumerate(pid_start):
            poseidx += list(range(index, index+self.length))
            wrongindex = random.choice(list(range(volume-4)))
            while wrongindex == index:
                wrongindex = random.choice(list(range(volume-4)))
            ididx += list(range(wrongindex, wrongindex+self.length))

        if not self.train:
            ididx = np.zeros(len(poseidx), dtype=np.int32)

        # (N*5)
        pickedimg = poseidx

        poseImg = self.croppatch(imgs[poseidx], bbxs[poseidx])
        idImg = self.croppatch(imgs[ididx], bbxs[ididx])

        poseImg = torch.from_numpy(poseImg)
        idImg = torch.from_numpy(idImg)

        trgt = self.readtext(word_path)

        spectrogram = self.audio_visual_align(spectrogram, imgs)

        poseImg = self.im_preprocess(poseImg)
        gtImg = poseImg.clone()
        # mask off the bottom half
        poseImg[:, :, poseImg.shape[2] // 2:] = 0.

        idImg = self.im_preprocess(idImg)
        inpImg = torch.cat([poseImg, idImg], axis=1)

        pickedimg = torch.tensor(pickedimg)

        return inpImg, spectrogram, gtImg, trgt, volume, pickedimg, torch.from_numpy(imgs), torch.from_numpy(bbxs)


def collater_seq_label_s2s(targets):
    lengths = torch.LongTensor([len(t) for t in targets])
    ntokens = lengths.sum().item()
    pad, eos = 1, 2
    targets_ = data_utils.collate_tokens(targets, pad_idx=pad, eos_idx=eos, left_pad=False)
    prev_output_tokens = data_utils.collate_tokens(targets, pad_idx=pad, eos_idx=eos, left_pad=False, move_eos_to_beginning=True)
    return (targets_, prev_output_tokens), lengths, ntokens


def collater_label(targets_by_label):
    targets_list, lengths_list, ntokens_list = [], [], []
    itr = zip(targets_by_label, [-1], [1])
    for targets, label_rate, pad in itr:
        if label_rate == -1:
            targets, lengths, ntokens = collater_seq_label_s2s(targets)
        targets_list.append(targets)
        lengths_list.append(lengths)
        ntokens_list.append(ntokens)
    return targets_list[0], lengths_list[0], ntokens_list[0]


def collate_fn(dataBatch):
    """
    Args:
        dataBatch:

    Returns:
        inpBatch: input T_sum*6*96*96, concatenation of all video chips in the time dimension
        gtBatch: output T_sum*3*96*96
        inputLenBatch: bs
        audioBatch: bs*104*T'
        audio_idx: T_sum
        targetBatch: words for lip-reading expert
        padding_mask: bs*T'
        pickedimg: a list of bs elements, each contain some picked indices
        videoBatch: a list of bs elements, each cotain a video
        bbxs: a list of bs elements
    """

    inpBatch = torch.cat([data[0] for data in dataBatch], dim=0)
    gtBatch = torch.cat([data[2] for data in dataBatch], dim=0)
    inputLenBatch = [data[4] for data in dataBatch]

    audioBatch, padding_mask = collater_audio([data[1] for data in dataBatch], max(inputLenBatch))
    audio_idx = torch.cat([data[5] + audioBatch.shape[2] * i for i, data in enumerate(dataBatch)], dim=0)

    targetBatch = collater_label([[data[3] for data in dataBatch]])

    bbxs = [data[7] for data in dataBatch]
    pickedimg = [data[5] for data in dataBatch]
    videoBatch = [data[6] for data in dataBatch]

    return inpBatch, audioBatch, audio_idx, gtBatch, targetBatch, padding_mask, pickedimg, videoBatch, bbxs


def save_sample_images(x, g, gt, global_step, checkpoint_dir):
    x = (x.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.).astype(np.uint8)
    g = (g.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.).astype(np.uint8)
    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.).astype(np.uint8)

    refs, inps = x[..., 3:], x[..., :3]
    folder = join(checkpoint_dir, "samples_step{:09d}".format(global_step))
    if not os.path.exists(folder): os.mkdir(folder)
    collage = np.concatenate((refs, inps, g, gt), axis=-2)
    for batch_idx, c in enumerate(collage):
            cv2.imwrite('{}/{}.jpg'.format(folder, batch_idx), c)


def get_gpu_memory_map():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def local_sync_loss(pickid, enc_audio, enc_video):
    pickedAud = enc_audio.permute(1, 0, 2).reshape(-1, enc_audio.shape[2])[pickid]
    pickedVid = enc_video.permute(1, 0, 2).reshape(-1, enc_video.shape[2])[pickid]
    return pickedVid, pickedAud


class status_manager(object):
    def __init__(self, patience=5, status=0):
        self.min = 100.
        self.waited_itr = 0
        self.status = status
        self.patience = patience

    def update(self, performance):
        if performance < self.min:
            self.min = performance
            self.waited_itr = 0
        else:
            self.waited_itr += 1

    def check_status(self):
        if self.waited_itr > self.patience:
            self.status += 1
            self.waited_itr = 0
            return self.status, True
        else:
            return self.status, False

def train(device, model, avhubert, criterion, data_loader, optimizer, args, global_step, logger):
    # RTX 4080 optimization: Enables the built-in cudnn auto-tuner
    torch.backends.cudnn.benchmark = True
    
    # Mixed Precision Scaler for RTX 4080 Tensor Cores
    scaler = torch.cuda.amp.GradScaler()
    
    print('Starting Step: {}'.format(global_step))

    lip_train = False
    model['gen'].ft = False
    status = status_manager(5)
    recon_loss = nn.L1Loss()

    for epoch in range(args.n_epoch):
        losses = {'lip': 0, 'local_sync': 0, 'l1': 0, 'prec_g': 0, 'disc_real_g': 0, 'disc_fake_g': 0}
        prog_bar = tqdm(enumerate(data_loader['train']), total=len(data_loader['train']))
        oom_count = 0
        
        for step, (inpim, spectrogram, audio_idx, gtim, ((trgt, prev_trg), tlen, ntoken), padding_mask, vidx, videos, bbxs) in prog_bar:
            for key in model.keys():
                model[key].train()

            criterion.report_accuracy = False

            # Transfer to GPU
            inpim, gtim = inpim.to(device, non_blocking=True), gtim.to(device, non_blocking=True)
            trgt, prev_trg = trgt.to(device, non_blocking=True), prev_trg.to(device, non_blocking=True)
            spectrogram, padding_mask = spectrogram.to(device, non_blocking=True), padding_mask.to(device, non_blocking=True)

            for key in optimizer.keys():
                optimizer[key].zero_grad(set_to_none=True) # Optimized zero_grad

            net_input = {'source': {'audio': spectrogram, 'video': None}, 'padding_mask': padding_mask, 'prev_output_tokens': prev_trg}
            sample = {'net_input': net_input, 'target_lengths': tlen, 'ntokens': ntoken, 'target': trgt}

            try:
                # --- GENERATOR TRAINING WITH AMP ---
                with torch.cuda.amp.autocast():
                    syntim, enc_audio = model['gen'](sample, inpim, audio_idx, spectrogram.shape[0])

                    if lip_train:
                        processed_img = images2avhubert(vidx, videos, bbxs, syntim, spectrogram.shape[2], device)
                        sample['net_input']['source']['video'] = processed_img
                        sample['net_input']['source']['audio'] = None
                        lip_loss, sample_size, logs, enc_out = criterion(avhubert, sample)
                        
                        if args.cont_w > 0:
                            pickedVid, pickedAud = local_sync_loss(audio_idx, enc_audio, enc_out['encoder_out'])
                            local_sync = model['gen'].sync_net(pickedVid, pickedAud)
                        else:
                            local_sync = torch.tensor(0., device=device)
                    else:
                        lip_loss, local_sync = torch.tensor(0., device=device), torch.tensor(0., device=device)

                    if args.perp_w > 0.:
                        perceptual_loss = model['disc'].perceptual_forward(syntim)
                    else:
                        perceptual_loss = torch.tensor(0., device=device)

                    l1loss = recon_loss(syntim, gtim)
                    
                    # Total Loss Calculation
                    loss = args.lip_w * lip_loss + args.perp_w * perceptual_loss + \
                           (1. - args.lip_w - args.perp_w) * l1loss + args.cont_w * local_sync

                # Scaled Backward Pass
                scaler.scale(loss).backward()
                scaler.step(optimizer['gen'])

                # --- DISCRIMINATOR TRAINING ---
                optimizer['gen'].zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=False):

                    gt_float = gtim.float()
                    syn_float = syntim.detach().float()

                    pred_real = model['disc'](gt_float)
                    disc_real_loss = F.binary_cross_entropy(pred_real, torch.ones((len(pred_real), 1), device=device))
                    
                    pred_fake = model['disc'](syn_float)
                    disc_fake_loss = F.binary_cross_entropy(pred_fake, torch.zeros((len(pred_fake), 1), device=device))
                    
                    disc_loss = disc_real_loss + disc_fake_loss

                scaler.scale(disc_loss).backward()
                scaler.step(optimizer['disc'])
                
                # Update Scaler for next iteration
                scaler.update()

                # Logging losses
                losses['lip'] += lip_loss.item()
                losses['local_sync'] += local_sync.item()
                losses['l1'] += l1loss.item()
                losses['prec_g'] += perceptual_loss.item()
                losses['disc_real_g'] += disc_real_loss.item()
                losses['disc_fake_g'] += disc_fake_loss.item()

                if global_step % args.ckpt_interval == 0:
                    save_sample_images(inpim, syntim, gtim, global_step, args.checkpoint_dir)

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"| Step {global_step} | CUDA OOM. Reducing cache...")
                    torch.cuda.empty_cache()
                    oom_count += 1
                    continue
                else:
                    raise e

            global_step += 1

            if global_step == 1 or global_step % args.ckpt_interval == 0:
                save_checkpoint(model['gen'], optimizer['gen'], global_step, args.checkpoint_dir, epoch)
                save_checkpoint(model['disc'], optimizer['disc'], global_step, args.checkpoint_dir, epoch, prefix='disc_')

            # Performance Monitor in TQDM
            if step % 5 == 0:
                vram_used = torch.cuda.memory_reserved(device) / 1e9
                prog_bar.set_description(f"Epoch {epoch} | L1: {l1loss.item():.4f} | VRAM: {vram_used:.1f}GB")

            if global_step % args.ckpt_interval == 0:
                with torch.no_grad():
                    average_sync_loss, valid_log = eval_model(data_loader['test'], avhubert, criterion, global_step, device, model['gen'], model['disc'], args.cont_w, recon_loss)
                    
                    logger.info(f"Step {global_step} Train Log: {losses}")
                    logger.info(valid_log)

                    status.update(average_sync_loss)
                    stage, changed = status.check_status()

                    if changed:
                        if stage == 1:
                            model['gen'].ft = True
                            logger.info('>>> Audio encoder start to finetune')
                        elif stage == 2:
                            lip_train = True
                            logger.info('>>> Lip reading start to work')
                        elif stage == 3:
                            logger.info('>>> Training Completed Successfully')
                            return 

    logger.info('Total OOM cases: {}'.format(oom_count))


def eval_model(test_data_loader, avhubert, criterion, global_step, device, model, disc, cont_w, recon_loss):
    print('Evaluating after training of {} steps'.format(global_step))
    n_correct, n_total = 0, 0
    losses = {'lip': 0, 'local_sync': 0, 'l1': 0, 'prec_g': 0, 'disc_real_g': 0, 'disc_fake_g': 0}

    for step, (x, spectrogram, audio_idx, gt, ((trgt, prev_trg), tlen, ntoken), padding_mask, vidx, videos, bbxs) in enumerate((test_data_loader)):
        model.eval()
        disc.eval()
        criterion.report_accuracy = True

        x = x.to(device)
        spectrogram = spectrogram.to(device)
        gt = gt.to(device)
        trgt, prev_trg = trgt.to(device), prev_trg.to(device)
        padding_mask = padding_mask.to(device)

        sample = {'net_input': {'source': {'audio': spectrogram, 'video': None}, 'padding_mask': padding_mask, 'prev_output_tokens': prev_trg},
                  'target_lengths': tlen, 'ntokens': ntoken, 'target': trgt}

        g, enc_audio = model(sample, x, audio_idx, spectrogram.shape[0])

        pred = disc(gt)
        disc_real_loss = F.binary_cross_entropy(pred, torch.ones((len(pred), 1)).to(device))
        losses['disc_real_g'] += disc_real_loss.item()
        pred = disc(g)
        disc_fake_loss = F.binary_cross_entropy(pred, torch.zeros((len(pred), 1)).to(device))
        losses['disc_fake_g'] += disc_fake_loss.item()

        processed_img = images2avhubert(vidx, videos, bbxs, g, spectrogram.shape[2], device)
        sample['net_input']['source']['video'] = processed_img
        sample['net_input']['source']['audio'] = None

        lip_loss, sample_size, logs, enc_out = criterion(avhubert, sample)
        losses['lip'] += lip_loss.item()

        if cont_w > 0:
            pickedVid, pickedAud = local_sync_loss(audio_idx, enc_audio, enc_out['encoder_out'])
            local_sync = model.sync_net(pickedVid, pickedAud)
            losses['local_sync'] += local_sync.item()

        n_correct += logs['n_correct']
        n_total += logs['total']
        if args.perp_w > 0.:
            perceptual_loss = disc.perceptual_forward(g)
            losses['prec_g'] += perceptual_loss.item()

        l1loss = recon_loss(g, gt)
        losses['l1'] += l1loss.item()

    avewer = 1 - n_correct / n_total

    valid_log = 'Valid step: {} '.format(global_step)
    for key, value in losses.items():
        valid_log += '{}: {:.4f} '.format(key, value / (step + 1))
    valid_log += '| wer: {}'.format(avewer)
    print(valid_log)
    return losses['l1'], valid_log


def save_checkpoint(model, optimizer, global_step, checkpoint_dir, epoch, prefix=''):
    checkpoint_path = join(
        checkpoint_dir, "{}checkpoint_step{:09d}.pth".format(prefix, global_step))
    optimizer_state = optimizer.state_dict() if optimizer is not None else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": global_step,
        "global_epoch": epoch,
    }, checkpoint_path)

    print("Saved checkpoint:", checkpoint_path)


def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint(path, model, optimizer, logger, reset_optimizer=False, overwrite_global_states=True):

    print("Load checkpoint from: {}".format(path))
    logger.info("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = model.state_dict()
    for k, v in s.items():
        new_s[k] = v
    model.load_state_dict(new_s)

    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            logger.info("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    if overwrite_global_states:
        global_step = checkpoint["global_step"]
    else:
        global_step = 0

    return global_step


if __name__ == "__main__":
    # Windows par multiprocessing spawn error se bachne ke liye zaroori hai
    import multiprocessing
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(
        description='TalkLip Training - RTX 4080 Optimized')

    # --- Dataset Path Arguments ---
    parser.add_argument("--video_root", help="Root folder of video", required=True, type=str)
    parser.add_argument("--audio_root", help="Root folder of audio", required=True, type=str)
    parser.add_argument("--word_root", help="Root folder of audio", required=True, type=str)
    parser.add_argument('--bbx_root', help="Root folder of bounding boxes of faces", required=True, type=str)
    parser.add_argument("--file_dir", help="Root folder of filelists", required=True, type=str)
    
    # RTX 4080 (16GB VRAM) optimized defaults
    parser.add_argument('--batch_size', help='batch size of training', default=32, type=int)
    parser.add_argument('--num_worker', help='number of worker', default=0, type=int)

    # --- Checkpoint Arguments ---
    parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)
    parser.add_argument('--gen_checkpoint_path', help='Resume generator from this checkpoint', default=None, type=str)
    parser.add_argument('--disc_checkpoint_path', help='Resume discriminator from this checkpoint', default=None, type=str)
    parser.add_argument('--avhubert_path', help='Resume avhubert from this checkpoint', default=None, type=str)
    parser.add_argument('--avhubert_root', help='Path of av_hubert root', required=True, type=str)

    # --- Optimizer & Loss Arguments ---
    parser.add_argument('--lr', help='learning rate', default=1e-4, type=float)
    parser.add_argument('--lip_w', help='weight of lip-reading expert', default=1e-5, type=float)
    parser.add_argument('--cont_w', help='weight of contrastive learning', default=1e-3, type=float)
    parser.add_argument('--perp_w', help='weight of perceptual loss', default=0.07, type=float)

    # --- Training Config ---
    parser.add_argument('--gpu', help='index of gpu used', default=0, type=int)
    parser.add_argument('--n_epoch', help='number of epoch', default=100, type=int)
    parser.add_argument('--log_name', help='name of a log file', default='talklip', type=str)
    parser.add_argument('--ckpt_interval', help='The interval of saving a checkpoint', default=3000, type=int)

    args = parser.parse_args()

    # --- Device Setup ---
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")
    print(f"Status: Using {torch.cuda.get_device_name(args.gpu)} | VRAM: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")

    # --- AVHubert Loading ---
    avhubert, label_proc, generator, criterion, encoder = retrieve_avhubert(args.avhubert_root, args.avhubert_path, device)

    # --- Dataset & DataLoader (64GB RAM Optimized) ---
    train_dataset = Talklipdata('train', args, label_proc)
    test_dataset = Talklipdata('valid', args, label_proc)

    # pin_memory=True data loading ko GPU ke liye fast bana deta hai
    # train.py ke andar DataLoader wala hissa aise hona chahiye:
    train_data_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=args.num_worker,
        pin_memory=True, # RTX 4080 ke liye lazmi hai
        persistent_workers=True if args.num_worker > 0 else False # Speed boost
    )

    test_data_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        collate_fn=collate_fn, 
        num_workers=args.num_worker,
        pin_memory=True if use_cuda else False
    )

    # --- Model Initialization ---
    # TalkLip model 768-dim encoder ke sath
    imGen = TalkLip(encoder, 768).to(device)
    imDisc = TalkLip_disc_qual().to(device)

    # --- Optimizers (RTX Optimized Betas) ---
    optimizer = optim.Adam([p for p in imGen.parameters() if p.requires_grad],
                           lr=args.lr, betas=(0.5, 0.999))
    
    disc_optimizer = optim.Adam([p for p in imDisc.parameters() if p.requires_grad],
                                lr=args.lr, betas=(0.5, 0.999))

    # --- Logging & Checkpoints ---
    os.makedirs('log/', exist_ok=True)
    logger = init_logging(log_name='log/{}.log'.format(args.log_name))

    global_step = 0
    if args.gen_checkpoint_path is not None:
        global_step = load_checkpoint(args.gen_checkpoint_path, imGen, optimizer, logger)

    if args.disc_checkpoint_path is not None:
        load_checkpoint(args.disc_checkpoint_path, imDisc, disc_optimizer, logger,
                        reset_optimizer=False, overwrite_global_states=False)

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    # --- Start Training ---
    train(device, 
          {'gen': imGen, 'disc': imDisc}, 
          avhubert, 
          criterion, 
          {'train': train_data_loader, 'test': test_data_loader},
          {'gen': optimizer, 'disc': disc_optimizer}, 
          args, 
          global_step, 
          logger)