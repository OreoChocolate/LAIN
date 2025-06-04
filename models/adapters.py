import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import math

from models.transformers import TransformerDecoderLayer


import copy
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class LAINadapter(nn.Module):
    def __init__(self,
                 args=None,
                 adapt_dim=64,
                 dropout=0.1,
                 init_option="lora",
                 adapter_scalar="learnable_scalar",
                 adapter_num_layers=1,
                 ):
        super().__init__()
        self.up_dim = 768
        self.down_dim = adapt_dim

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(768) * 1e-9)
            self.scale2 = nn.Parameter(torch.ones(768) * 1e-9)
        else:
            self.scale = float(adapter_scalar)

        self.alpha = args.adapter_alpha

        self.down_proj = nn.Linear(768, self.down_dim)
        self.non_linear_func = nn.LeakyReLU()

        self.down_proj2 = nn.Linear(768, self.down_dim)

        self.up_proj = nn.Linear(self.down_dim * 2, self.up_dim)
        self.up_proj2 = nn.Linear(self.down_dim, self.up_dim)

        self.down_proj3 = nn.Linear(768, self.down_dim)
        self.down_proj6 = nn.Linear(self.down_dim, self.down_dim)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.down_proj3.weight, a=math.sqrt(5))
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.down_proj3.bias)

        self.adapter_num_layers = adapter_num_layers

        instance_decoder_layer = TransformerDecoderLayer(self.down_dim, 2, self.down_dim * 2,
                                                         self.dropout, 'relu', False)
        self.mhsa_layers = _get_clones(instance_decoder_layer, adapter_num_layers)

        instance_decoder_layer = TransformerDecoderLayer(self.down_dim, 2, self.down_dim * 2,
                                                         self.dropout, 'relu', False)
        self.mhsa_layers2 = _get_clones(instance_decoder_layer, adapter_num_layers)

        instance_decoder_layer = TransformerDecoderLayer(self.down_dim, 2, self.down_dim * 2,
                                                         self.dropout, 'relu', False)
        self.mhsa_layers3 = _get_clones(instance_decoder_layer, adapter_num_layers)

        self.ln = nn.LayerNorm(self.down_dim)
        self.ln2 = nn.LayerNorm(self.down_dim)
        self.ln3 = nn.LayerNorm(self.down_dim)

        self.prompt = nn.Embedding(10, self.down_dim)

        self.down_proj4 = nn.Conv2d(self.down_dim, self.down_dim, 3, 1, 1)
        self.down_proj5 = nn.Conv2d(self.down_dim, self.down_dim, 5, 1, 2)

    def forward(self, x, prior=None, la_masks=None):
        ho_tokens, im_tokens = x[:-196], x[-196:]

        ho_tokens_down = self.down_proj(ho_tokens)
        ho_tokens_down = self.non_linear_func(ho_tokens_down)

        im_tokens_down = self.down_proj3(im_tokens)
        im_tokens_down = F.relu(im_tokens_down)

        #locality adapter
        im_tokens_down = self.alpha * im_tokens_down.view(14, 14, -1,self.down_dim).permute(2, 3, 0, 1) + prior.unsqueeze(0).permute(0, 3, 1, 2)
        im_tokens_down = self.down_proj6(self.ln3(im_tokens_down.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)

        im_tokens_map = self.ln2(
            self.non_linear_func(self.down_proj4(im_tokens_down) + self.down_proj5(im_tokens_down)).permute(0, 2, 3, 1))
        im_tokens_up = self.up_proj2(im_tokens_map).flatten(1, 2).permute(1, 0, 2)
        im_tokens_output = im_tokens_up * self.scale2

        #interaction adapter
        la_im_tokens = im_tokens + im_tokens_output


        object_boxes, human_pid, object_pid = la_masks
        object_boxes = torch.cat([object_boxes, torch.tensor([[0, 0, 224, 244]]).cuda()], dim=0)  # human
        object_feats = torchvision.ops.roi_align(la_im_tokens.permute(1, 2, 0).view(-1, 768, 14, 14), [object_boxes],
                                                 output_size=(7, 7),
                                                 spatial_scale=14 / 224, aligned=True)
        object_feats = object_feats.flatten(-2).permute(0, 2, 1)  # BND
        object_feats = self.down_proj2(object_feats)
        object_feats = self.ln(object_feats)


        object_pid = object_pid.tolist()
        object_pid.append(-1)
        human_pid = human_pid.tolist()
        human_pid.append(-1)

        context_tokens = self.prompt.weight.unsqueeze(1).repeat(1, object_feats.size(0), 1)
        for z, layer in enumerate(self.mhsa_layers):
            context_tokens = layer(context_tokens, object_feats.permute(1, 0, 2), tgt_mask=None,
                                   tgt_key_padding_mask=None,
                                   memory_key_padding_mask=None,
                                   pos=None, query_pos=None)

        for z, layer in enumerate(self.mhsa_layers3):
            ip_feats = layer(context_tokens[:, human_pid + object_pid],
                             context_tokens[:, object_pid + human_pid],
                             tgt_mask=None,
                             tgt_key_padding_mask=None,
                             memory_key_padding_mask=None,
                             pos=None, query_pos=None)

        for z, layer in enumerate(self.mhsa_layers2):
            updated_object_feats = layer(ho_tokens_down.permute(1, 0, 2).repeat(1, 2, 1),
                                         ip_feats,
                                         tgt_mask=None,
                                         tgt_key_padding_mask=None,
                                         memory_key_padding_mask=None,
                                         pos=None, query_pos=None)

        Hfeat, Ofeat = updated_object_feats.chunk(2, dim=1)
        ho_tokens_output = self.up_proj(torch.cat([Hfeat, Ofeat], dim=-1).permute(1, 0, 2))
        ho_tokens_output = ho_tokens_output * self.scale

        return torch.cat([ho_tokens_output, im_tokens_output], dim=0)



