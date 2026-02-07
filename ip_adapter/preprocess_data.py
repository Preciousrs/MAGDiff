import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import os
import re
import random
import argparse
from pathlib import Path
import json
import itertools
import time
from datetime import datetime
import shutil
import torch
import torch.nn.functional as F
import numpy as np
import math
import cv2
from torchvision import transforms
from PIL import Image
import PIL
from transformers import CLIPImageProcessor
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, ControlNetModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection, AutoProcessor, SiglipVisionModel, SiglipImageProcessor, AutoModel, AutoImageProcessor
#from deepspeed.runtime.engine import DeepSpeedEngine


from einops import rearrange

import cv2
import numpy as np
from PIL import Image
import pickle
from diffusers.utils import load_image


from transformers import (
    AutoTokenizer,
    CLIPTextModel,
    T5EncoderModel,
)

import logging



# def resize_img(input_image, max_side=1280, min_side=1024, size=None,
#                pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):

#     w, h = input_image.size
#     if size is not None:
#         w_resize_new, h_resize_new = size
#     else:
#         ratio = min_side / min(h, w)
#         w, h = round(ratio*w), round(ratio*h)
#         ratio = max_side / max(h, w)
#         input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
#         w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
#         h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
#     input_image = input_image.resize([w_resize_new, h_resize_new], mode)

#     if pad_to_max_side:
#         res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
#         offset_x = (max_side - w_resize_new) // 2
#         offset_y = (max_side - h_resize_new) // 2
#         res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
#         input_image = Image.fromarray(res)
#     return input_image




# def encode_siglip_image_emb(image_encoder, siglip_image, device, dtype):
   
#     # siglip_image = siglip_image.to(device, dtype=dtype)
#     res = image_encoder(siglip_image, output_hidden_states=True)

#     siglip_image_embeds = res.last_hidden_state

#     siglip_image_shallow_embeds = torch.cat([res.hidden_states[i] for i in [7, 13, 26]], dim=1)


#     return siglip_image_embeds, siglip_image_shallow_embeds


# def encode_dinov2_image_emb(dino_image_encoder, dinov2_image, device, dtype):
    
#     # dinov2_image = dinov2_image.to(device, dtype=dtype)
#     res = dino_image_encoder(dinov2_image, output_hidden_states=True)

#     dinov2_image_embeds = res.last_hidden_state[:, 1:]

#     dinov2_image_shallow_embeds = torch.cat([res.hidden_states[i][:, 1:] for i in [9, 19, 29]], dim=1)

#     return dinov2_image_embeds, dinov2_image_shallow_embeds

# class encode_image_emb:
#     def __init__(self, sigclip_image_encoder_path,  dinov2_image_encoder_path, device,dtype):
#         self.device = device
#         self.dtype = dtype
#         self.dino_image_encoder_path = dinov2_image_encoder_path
#         self.dino_image_encoder = AutoModel.from_pretrained(self.dino_image_encoder_path).to(self.device, dtype=torch.float16)
#         self.dino_image_processor = AutoImageProcessor.from_pretrained(self.dino_image_encoder_path)
#         self.dino_image_processor.crop_size = dict(height=384, width=384)
#         self.dino_image_encoder.size = dict(shortest_edge=384)
#         self.siglip_image_encoder_path = sigclip_image_encoder_path
#         self.image_encoder = SiglipVisionModel.from_pretrained(self.siglip_image_encoder_path).to(self.device, dtype=torch.float16)
#         self.siglip_image_processor = SiglipImageProcessor.from_pretrained(self.siglip_image_encoder_path)
    

    # def get_image_emb(self, siglip_image=None):
    #     object_image_pil = siglip_image
    #     object_image_pil_low_res = [object_image_pil.resize((384, 384))]
    #     object_image_pil_high_res = object_image_pil.resize((768, 768))
    #     object_image_pil_high_res = [
    #         object_image_pil_high_res.crop((0, 0, 384, 384)),
    #         object_image_pil_high_res.crop((384, 0, 768, 384)),
    #         object_image_pil_high_res.crop((0, 384, 384, 768)),
    #         object_image_pil_high_res.crop((384, 384, 768, 768)),
    #     ]
    #     nb_split_image = len(object_image_pil_high_res)

    #     siglip_image_embeds = encode_siglip_image_emb(
    #         self.image_encoder,
    #         self.siglip_image_processor(images=object_image_pil_low_res, return_tensors="pt").pixel_values, 
    #         self.device,
    #         self.dtype
    #     )
    #     dinov2_image_embeds = encode_dinov2_image_emb(
    #         self.dino_image_encoder,
    #         self.dino_image_processor(images=object_image_pil_low_res, return_tensors="pt").pixel_values, 
    #         self.device,
    #         self.dtype
    #     )

    #     image_embeds_low_res_deep = torch.cat([siglip_image_embeds[0], dinov2_image_embeds[0]], dim=2)
    #     image_embeds_low_res_shallow = torch.cat([siglip_image_embeds[1], dinov2_image_embeds[1]], dim=2)

    #     siglip_image_high_res = self.siglip_image_processor(images=object_image_pil_high_res, return_tensors="pt").pixel_values
    #     siglip_image_high_res = siglip_image_high_res[None]
    #     siglip_image_high_res = rearrange(siglip_image_high_res, 'b n c h w -> (b n) c h w')
    #     siglip_image_high_res_embeds = encode_siglip_image_emb(self.image_encoder, siglip_image_high_res, self.device, self.dtype)
    #     siglip_image_high_res_deep = rearrange(siglip_image_high_res_embeds[0], '(b n) l c -> b (n l) c', n=nb_split_image)
    #     dinov2_image_high_res = self.dino_image_processor(images=object_image_pil_high_res, return_tensors="pt").pixel_values
    #     dinov2_image_high_res = dinov2_image_high_res[None]
    #     dinov2_image_high_res = rearrange(dinov2_image_high_res, 'b n c h w -> (b n) c h w')
    #     dinov2_image_high_res_embeds = encode_dinov2_image_emb(self.dino_image_encoder, dinov2_image_high_res, self.device, self.dtype)
    #     dinov2_image_high_res_deep = rearrange(dinov2_image_high_res_embeds[0], '(b n) l c -> b (n l) c', n=nb_split_image)
    #     image_embeds_high_res_deep = torch.cat([siglip_image_high_res_deep, dinov2_image_high_res_deep], dim=2)

    #     image_embeds_dict = dict(
    #         image_embeds_low_res_shallow=image_embeds_low_res_shallow,
    #         image_embeds_low_res_deep=image_embeds_low_res_deep,
    #         image_embeds_high_res_deep=image_embeds_high_res_deep,
    #     )

    #     return image_embeds_dict


import torch
from einops import rearrange

class EncodeImageEmb:
    def __init__(self, sigclip_path, dinov2_path, device, dtype):
        self.device = device
        self.dtype = dtype

        self.dino_image_encoder = AutoModel.from_pretrained(dinov2_path)
        # self.dino_image_encoder = AutoModel.from_pretrained(dinov2_path)
        # self.dino_image_processor = AutoImageProcessor.from_pretrained(dinov2_path)
        # self.dino_image_processor.crop_size = dict(height=384, width=384)
        self.dino_image_encoder.size = dict(shortest_edge=224)

        self.siglip_image_encoder = SiglipVisionModel.from_pretrained(sigclip_path)
        # self.siglip_image_encoder = SiglipVisionModel.from_pretrained(sigclip_path)

        # self.siglip_image_processor = SiglipImageProcessor.from_pretrained(sigclip_path)

    def encode_siglip_image_emb(self, siglip_image):
        self.siglip_image_encoder.to(self.device, dtype=self.dtype)
        res = self.siglip_image_encoder(siglip_image, output_hidden_states=True)
        siglip_image_embeds = res.last_hidden_state
        # print(len(res.hidden_states))# 224 长度为13
        # siglip_image_shallow_embeds = torch.cat([res.hidden_states[i] for i in [7, 13, 26]], dim=1)#384
        siglip_image_shallow_embeds = torch.cat([res.hidden_states[i] for i in [3,9, 12]], dim=1)#siglip 224
        # for i in range(len(res.hidden_states)):
        #     print(f'{i} : {res.hidden_states[i].shape}')
        self.siglip_image_encoder.to("cpu")
        return siglip_image_embeds, siglip_image_shallow_embeds

    def encode_dinov2_image_emb(self, dinov2_image):
        self.dino_image_encoder.to(self.device, dtype=self.dtype)
        res = self.dino_image_encoder(dinov2_image, output_hidden_states=True)

        dinov2_image_embeds = res.last_hidden_state[:, 1:]
        dinov2_image_shallow_embeds = torch.cat([res.hidden_states[i][:, 1:] for i in [3, 9, 12]], dim=1)
        self.dino_image_encoder.to("cpu")
        return dinov2_image_embeds, dinov2_image_shallow_embeds

    def get_image_emb(self, siglip_low ,dinov2_low, siglip_image_high_res, dinov2_image_high_res):
        # 处理低分辨率图像

        siglip_embeds = self.encode_siglip_image_emb(siglip_low)
        dinov2_embeds = self.encode_dinov2_image_emb(dinov2_low)
        # print(siglip_embeds[0].shape, dinov2_embeds[0].shape)
        image_embeds_low_res_deep = torch.cat([siglip_embeds[0], dinov2_embeds[0]], dim=2)
        # print(siglip_embeds[1].shape, dinov2_embeds[1].shape)

        image_embeds_low_res_shallow = torch.cat([siglip_embeds[1], dinov2_embeds[1]], dim=2)

        # 高分辨率处理
       
        nb_split = 1
        
        siglip_high_embeds = self.encode_siglip_image_emb(siglip_image_high_res)
        siglip_high_deep = rearrange(siglip_high_embeds[0], '(b n) l c -> b (n l) c', n=nb_split)

    
        dinov2_high_embeds = self.encode_dinov2_image_emb(dinov2_image_high_res)
        dinov2_high_deep = rearrange(dinov2_high_embeds[0], '(b n) l c -> b (n l) c', n=nb_split)

        image_embeds_high_res_deep = torch.cat([siglip_high_deep, dinov2_high_deep], dim=2)
        
        return dict(
            image_embeds_low_res_shallow=image_embeds_low_res_shallow,
            image_embeds_low_res_deep=image_embeds_low_res_deep,
            image_embeds_high_res_deep=image_embeds_high_res_deep,
        )
    

    # def get_image_emb(self, image_tensor):
    #     # 确保输入是 (N, C, H, W) 格式
    #     if image_tensor.dim() != 4 or image_tensor.size(0) != 2:
    #         raise ValueError("Expected input tensor shape (2, 3, H, W)")

    #     # 处理低分辨率图像
    #     low_res = torch.nn.functional.interpolate(image_tensor, size=(384, 384), mode='bilinear', align_corners=False)
    #     siglip_low = self.siglip_image_processor(images=low_res, return_tensors="pt").pixel_values.squeeze(0)  # [C,H,W]
    #     dinov2_low = self.dino_image_processor(images=low_res, return_tensors="pt").pixel_values.squeeze(0)

    #     siglip_embeds = self.encode_siglip_image_emb(siglip_low.unsqueeze(0))
    #     dinov2_embeds = self.encode_dinov2_image_emb(dinov2_low.unsqueeze(0))

    #     image_embeds_low_res_deep = torch.cat([siglip_embeds[0], dinov2_embeds[0]], dim=2)
    #     image_embeds_low_res_shallow = torch.cat([siglip_embeds[1], dinov2_embeds[1]], dim=2)

    #     # 高分辨率处理
    #     high_res_img = torch.nn.functional.interpolate(image_tensor, size=(768, 768), mode='bilinear', align_corners=False)
        
    #     crops = [
    #         high_res_img[:, :, 0:384, 0:384],
    #         high_res_img[:, :, 384:768, 0:384],
    #         high_res_img[:, :, 0:384, 384:768],
    #         high_res_img[:, :, 384:768, 384:768],
    #     ]
    #     nb_split = len(crops)

    #     siglip_high = self.siglip_image_processor(images=crops, return_tensors="pt").pixel_values
    #     siglip_high = rearrange(siglip_high, 'b n c h w -> (b n) c h w')
    #     siglip_high_embeds = self.encode_siglip_image_emb(siglip_high)
    #     siglip_high_deep = rearrange(siglip_high_embeds[0], '(b n) l c -> b (n l) c', n=nb_split)

    #     dinov2_high = self.dino_image_processor(images=crops, return_tensors="pt").pixel_values
    #     dinov2_high = rearrange(dinov2_high, 'b n c h w -> (b n) c h w')
    #     dinov2_high_embeds = self.encode_dinov2_image_emb(dinov2_high)
    #     dinov2_high_deep = rearrange(dinov2_high_embeds[0], '(b n) l c -> b (n l) c', n=nb_split)

    #     image_embeds_high_res_deep = torch.cat([siglip_high_deep, dinov2_high_deep], dim=2)

    #     return dict(
    #         image_embeds_low_res_shallow=image_embeds_low_res_shallow,
    #         image_embeds_low_res_deep=image_embeds_low_res_deep,
    #         image_embeds_high_res_deep=image_embeds_high_res_deep,
    #     )
