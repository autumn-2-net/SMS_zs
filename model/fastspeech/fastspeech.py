import torch
import torch.nn as nn
import torch.nn.functional as F
# from fairseq.modules import RelPositionalEncoding

from model.attention.attention_lay import attn_lay
from model.fastspeech.PE import RelPositionalEncoding
from utils.datapre_ph import mel2ph_to_dur


class fs_encoder(nn.Module):
    def __init__(self,dim,lays, heads=4, dim_head=64,hideen_dim=None,kernel_size=9):
        super().__init__()
        self.attn_lay=nn.ModuleList([attn_lay(dim=dim, heads=heads, dim_head=dim_head,hideen_dim=hideen_dim,kernel_size=kernel_size) for _ in range(lays)])

    def forward(self,x,mask=None):
        for i in self.attn_lay:
            x=i(x,mask)
        return x


class fastspeech2(nn.Module): #B T C
    def __init__(self,vocab_size,config:dict):
        super().__init__()

        self.txt_embed = nn.Embedding(vocab_size, config['fs2_hidden_size'], padding_idx=0)
        self.dur_embed = nn.Linear(1, config['fs2_hidden_size'])

        self.encoder = fs_encoder(dim=config['fs2_hidden_size'],lays=config['fs2_lays'], heads=config['fs2_heads'], dim_head=config['fs2_dim_head'],hideen_dim=config.get('fs2_latent_dim'),kernel_size=config['fs2_kernel_size'])


        self.variance_embed_list = []
        self.use_energy_embed = config.get('use_energy_embed', False)
        self.use_breathiness_embed = config.get('use_breathiness_embed', False)
        if self.use_energy_embed:
            self.variance_embed_list.append('energy')
        if self.use_breathiness_embed:
            self.variance_embed_list.append('breathiness')

        self.use_variance_embeds = len(self.variance_embed_list) > 0
        if self.use_variance_embeds:
            self.variance_embeds = nn.ModuleDict({
                v_name: nn.Linear(1, config['fs2_hidden_size'])
                for v_name in self.variance_embed_list
            })



        self.use_key_shift_embed = config.get('use_key_shift_embed', False)
        if self.use_key_shift_embed:
            self.key_shift_embed = nn.Linear(1, config['fs2_hidden_size'])

        self.use_speed_embed = config.get('use_speed_embed', False)
        if self.use_speed_embed:
            self.speed_embed = nn.Linear(1, config['fs2_hidden_size'])

        self.embed_positions = RelPositionalEncoding(config['fs2_hidden_size'], dropout_rate=0.0)


    def forward_variance_embedding(self, condition, key_shift=None, speed=None, **variances):
        if self.use_variance_embeds:
            variance_embeds = torch.stack([
                self.variance_embeds[v_name](variances[v_name][:, :, None])
                for v_name in self.variance_embed_list
            ], dim=-1).sum(-1)
            condition += variance_embeds

        if self.use_key_shift_embed:
            key_shift_embed = self.key_shift_embed(key_shift[:, :, None])
            condition += key_shift_embed

        if self.use_speed_embed:
            speed_embed = self.speed_embed(speed[:, :, None])
            condition += speed_embed

        return condition

    def femb(self,x,dur):
        x=x+dur
        x=self.embed_positions (x)

        return x


    def forward(self, txt_tokens,mel2ph,  key_shift=None, speed=None,
           **kwargs):
        txt_embed = self.txt_embed(txt_tokens)
        dur = mel2ph_to_dur(mel2ph, txt_tokens.shape[1]).float()
        dur_embed = self.dur_embed(dur[:, :, None])
        encoder_out = self.encoder(self.femb(txt_embed, dur_embed), txt_tokens != 0)


        encoder_out = F.pad(encoder_out, [0, 0, 1, 0])
        mel2ph_ = mel2ph[..., None].repeat([1, 1, encoder_out.shape[-1]])
        condition = torch.gather(encoder_out, 1, mel2ph_)






        condition = self.forward_variance_embedding(
            condition, key_shift=key_shift, speed=speed, **kwargs
        )
        return condition

