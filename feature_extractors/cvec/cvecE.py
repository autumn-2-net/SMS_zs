

import fairseq
import torch
import torchaudio

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
    waveform=torchaudio.transforms.Resample(orig_freq=config['audio_sample_rate'],new_freq=16000)(waveform)
    emissions, _ = model.extract_features(waveform,
                                         # padding_mask=torch.BoolTensor(waveform.shape).fill_(False).to('cuda'),
                                          output_layer=12)
    emissions = emissions[0].cpu().detach().transpose(1, 0)
    emissions = emissions.unsqueeze(0)

    return emissions