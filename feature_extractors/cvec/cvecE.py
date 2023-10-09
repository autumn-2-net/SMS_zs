

import fairseq
import numpy as np
import torch
import torchaudio

from utils.data_orgmelE import wav2spec

model=None
def set_cvec(paths,cudas=True):
    global model
    ckpt_path = paths
    models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])

    if cudas:
        model =models[0].to('cuda').eval()
    else:
        model = models[0].eval()




    return model


@torch.no_grad()
def cvecE(waveform,config,cudas=True):

    # waveform, _ = torchaudio.load(lll[0])

    if cudas:
        waveform=waveform.to('cuda')
        waveform=torchaudio.transforms.Resample(orig_freq=config['audio_sample_rate'],new_freq=16000).to('cuda')(waveform)
    else:
        waveform = torchaudio.transforms.Resample(orig_freq=config['audio_sample_rate'], new_freq=16000)(waveform)
    emissions, _ = model.extract_features(waveform.unsqueeze(0),
                                         # padding_mask=torch.BoolTensor(waveform.shape).fill_(False).to('cuda'),
                                          output_layer=12)
    emissions = emissions[0].cpu().detach().transpose(1, 0)
    emissions = emissions.unsqueeze(0)

    return emissions
if __name__=='__main__':
    import glob
    import torchaudio
    from tqdm import tqdm

    set_cvec(r'D:\propj\sum_a\content-vec-best-legacy-500.pt')
    # from concurrent.futures import ProcessPoolExecutor
    # import random

    # import re
    # from torch.multiprocessing import Manager, Process, current_process, get_context
    #
    # is_main_process = not bool(re.match(r'((.*Process)|(SyncManager)|(.*PoolWorker))-\d+', current_process().name))
    torch.set_num_threads(4)

    lll = glob.glob(r'D:\propj\Disa\data\opencpop\raw\wavs/**.wav')
    torch.set_num_threads(1)
    lll=lll
    ooo=[]
    oos=r'D:\propj\sum_a\datass'

    for i in tqdm(lll):
        # print(i)
        nmm=i.split('\\')[-1]
        ooo.append(fr'{i},D:\propj\sum_a\datass/{nmm}.npy')
        audio, sr = torchaudio.load(i)
        audio = torch.clamp(audio[0], -1.0, 1.0)
        config={"audio_sample_rate":44100}
        fff=cvecE(audio,config=config)

        glic = {'audio_sample_rate': 44100, 'audio_num_mel_bins': 128, 'hop_size': 512, 'fft_size': 2048,
                'win_size': 2048,
                'fmin': 40, 'fmax': 16000}
        wav,sp=wav2spec(i, config=glic, device='cpu',
                 )

        xxxxx = \
        torch.nn.functional.interpolate(fff, size=len(sp), scale_factor=None, mode='nearest', align_corners=None)[0]
        np.save(f'D:\propj\sum_a\datass/{nmm}.npy', xxxxx.detach().cpu().numpy())

    for i in ooo:
        print(i)
