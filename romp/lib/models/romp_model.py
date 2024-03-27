from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import sys, os
root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from models.base import Base
from models.CoordConv import get_coord_maps
from models.basic_modules import BasicBlock,Bottleneck

import config
from config import args
from loss_funcs import Loss
from maps_utils.result_parser import ResultParser

import numpy as np

from torchvision.utils import save_image

BN_MOMENTUM = 0.1

class TransBlock(nn.Module):
    def __init__(self, input_channels, num_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        out = self.relu(x)
        return out

class CoNormBlock(nn.Module):
    def __init__(self, input_channels, hidden_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.phi_latent = nn.Sequential(nn.Conv2d(input_channels, hidden_channels, kernel_size, stride, padding),
                                        nn.ReLU())
        self.phi_scale = nn.Conv2d(hidden_channels, out_channels, kernel_size, stride, padding)
        self.phi_bias = nn.Conv2d(hidden_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x, joint_heatmap):
        latent = self.phi_latent(joint_heatmap).to(x.device)
        scale = self.phi_scale(latent).to(x.device)
        bias = self.phi_bias(latent).to(x.device)
        out = x * scale + bias
        return out


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
        
class ChannelCrossAttn(nn.Module):
    def __init__(self, in_dim):
        super(ChannelCrossAttn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        self.init_weights()

    def init_weights(self):
        kaiming_init(self.query_conv)
        kaiming_init(self.key_conv)
        kaiming_init(self.value_conv)

    def forward(self, q, kv, mask):
        m_batchsize, C, height, width = kv.size()    # B, C, H, W
        proj_query = self.query_conv(q).view(m_batchsize, C, -1)    # B, C, HxW
        proj_key = self.key_conv(kv).view(m_batchsize, C, -1).permute(0, 2, 1) # B, HxW, C
        energy = torch.bmm(proj_query, proj_key)*mask    # B, C, C     # 여기서 masking
        # energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy     # ?
        attention = self.softmax(energy)

        proj_value = self.value_conv(kv).view(m_batchsize, C, -1)    # B, C, HxW

        out = torch.bmm(attention, proj_value)  # B{(C, C) @ (C, HxW)} = B, C, HxW
        out = out.view(m_batchsize, C, height, width)   # B, C, H, W
        # out = self.gamma * out + x
        out = self.gamma*out + kv
        # return out, self.softmax(torch.bmm(proj_query, proj_key)), attention
        return out, torch.bmm(proj_query, proj_key), energy


class ROMP(Base):
    def __init__(self, backbone=None,**kwargs):
        super(ROMP, self).__init__()
        print('Using ROMP v1')
        self.backbone = backbone
        self._result_parser = ResultParser()
        self._build_head()
        if args().model_return_loss:
            self._calc_loss = Loss()
        if not args().fine_tune and not args().eval:
            self.init_weights()
            self.backbone.load_pretrain_params()

        self.coef_trans = TransBlock(self.backbone.backbone_channels+2, 64, 3, 2, 1)
        self.coef_resblock1 = BasicBlock(64, 64)
        self.coef_resblock2 = BasicBlock(64, 64)
        self.coef_convblock = TransBlock(64, 21, 3, 1, 1)
        self.attn = ChannelCrossAttn(21)
        coef_layers = []
        for i in range(63):
            coef_layers.append(nn.Conv2d(in_channels=85, out_channels=1, kernel_size=1, stride=1, padding=0))

        self.coef_layers = nn.ModuleList(coef_layers)

        self.masks = torch.from_numpy(np.load('/home/dcvl/MK/ROMP_Exp07_2/eigenpose_weight_mask.npy'))

    def head_forward(self,x):
        x = torch.cat((x, self.coordmaps.to(x.device).repeat(x.shape[0],1,1,1)), 1)

        params_maps = self.final_layers[1](x)
        center_maps = self.final_layers[2](x)
        if args().merge_smpl_camera_head:
            cam_maps, params_maps = params_maps[:,:3], params_maps[:,3:]
        else:
            cam_maps = self.final_layers[3](x)
        cam_maps[:, 0] = torch.pow(1.1,cam_maps[:, 0])
        params_maps = torch.cat([cam_maps, params_maps], 1)

        joint_maps = self.final_layers[4](x)
        x_ = self.coef_trans(x)
        # x_ = x_.to(x.device)
        x_ = self.coef_resblock1(x_)
        x_64 = self.coef_resblock2(x_)
        x_ = self.coef_convblock(x_64)

        for i in range(63):
            mask = self.masks[i].to(x.device).float()
            x__, attn_map, weighted_attn_map = self.attn(joint_maps, x_, mask)

            x__ = torch.cat((x_64, x__), 1)
            self.coef_layers[i] = self.coef_layers[i].to(x.device)
            if i == 0:
                coef_maps = self.coef_layers[i](x__).to(x.device)
            else:
                coef_maps = torch.cat((coef_maps, self.coef_layers[i](x__).to(x.device)), 1)

        output = {'params_maps':params_maps.float(), 'center_map':center_maps.float(), 'joint_maps':joint_maps.float(), 'coef_maps':coef_maps.float()}
        return output

    def _build_head(self):
        self.outmap_size = args().centermap_size
        params_num, cam_dim = self._result_parser.params_map_parser.params_num, 3
        self.head_cfg = {'NUM_HEADS': 1, 'NUM_CHANNELS': 64, 'NUM_BASIC_BLOCKS': args().head_block_num}
        self.output_cfg = {'NUM_PARAMS_MAP':params_num-cam_dim, 'NUM_CENTER_MAP':1, 'NUM_CAM_MAP':cam_dim}

        self.final_layers = self._make_final_layers(self.backbone.backbone_channels)
        self.coordmaps = get_coord_maps(128)

    def _make_final_layers(self, input_channels):
        final_layers = []
        final_layers.append(None)

        input_channels += 2
        if args().merge_smpl_camera_head:
            final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_PARAMS_MAP']+self.output_cfg['NUM_CAM_MAP']))
            final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_CENTER_MAP']))
        else:
            final_layers.append(self._make_head_layers(input_channels, 16)) # 10 for shape param, 6 for global orientation(pelvis)
            final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_CENTER_MAP']))
            final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_CAM_MAP']))
            final_layers.append(self._make_head_layers(input_channels, 21)) # joint heatmap

        return nn.ModuleList(final_layers)
    
    def _make_head_layers(self, input_channels, output_channels):
        head_layers = []
        num_channels = self.head_cfg['NUM_CHANNELS']

        kernel_sizes, strides, paddings = self._get_trans_cfg()
        for kernel_size, padding, stride in zip(kernel_sizes, paddings, strides):
            head_layers.append(nn.Sequential(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding),
                    nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)))
        
        for i in range(self.head_cfg['NUM_HEADS']):
            layers = []
            for _ in range(self.head_cfg['NUM_BASIC_BLOCKS']):
                layers.append(nn.Sequential(BasicBlock(num_channels, num_channels)))
            head_layers.append(nn.Sequential(*layers))

        head_layers.append(nn.Conv2d(in_channels=num_channels,out_channels=output_channels,\
            kernel_size=1,stride=1,padding=0))

        return nn.Sequential(*head_layers)



    def _get_trans_cfg(self):
        if self.outmap_size == 32:
            kernel_sizes = [3,3]
            paddings = [1,1]
            strides = [2,2]
        elif self.outmap_size == 64:
            kernel_sizes = [3]
            paddings = [1]
            strides = [2]
        elif self.outmap_size == 128:
            kernel_sizes = [3]
            paddings = [1]
            strides = [1]

        return kernel_sizes, strides, paddings

if __name__ == '__main__':
    args().configs_yml = 'configs/v1.yml'
    args().model_version=1
    from models.build import build_model
    model = build_model().cuda()
    outputs=model.feed_forward({'image':torch.rand(4,512,512,3).cuda()})
    for key, value in outputs.items():
        if isinstance(value,tuple):
            print(key, value)
        elif isinstance(value,list):
            print(key, value)
        else:
            print(key, value.shape)