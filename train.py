#基于reploy 数据 使用草图、轮廓和风格裁剪的图片进行生成  
#controlnet的encoder hidden 为text 输出之后再和ipadapter的特征cat

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
from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from accelerate import DistributedDataParallelKwargs

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, ControlNetModel, StableDiffusionControlNetPipeline, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection

from ip_adapter.pam import Resampler_new
from ip_adapter.utils import is_torch2_available
from ip_adapter.downsample_256 import DownsampleNetwork

if is_torch2_available():
    from ip_adapter.attention_processor import  CDAAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor

import torch.utils.checkpoint
import matplotlib.pyplot as plt


def random_crop_with_outline(image_path, outline_path, crop_size=64):
    # 读取图像和轮廓图
    image = Image.open(image_path).convert("RGB")
    outline = Image.open(outline_path).convert("RGB")
    
    # 定义变换
    size = 256
    transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(size)
    ])

    # 应用变换
    image = transform(image)
    outline = transform(outline)

    # 将图像和轮廓图转换回 NumPy 数组
    image_np = np.array(image)
    outline_np = np.array(outline)

    # 将轮廓图转换为灰度图
    gray_outline = cv2.cvtColor(outline_np, cv2.COLOR_BGR2GRAY)

    # 应用二值化以获取轮廓区域
    _, binary_outline = cv2.threshold(gray_outline, 240, 255, cv2.THRESH_BINARY_INV)

    # 找到轮廓
    contours, _ = cv2.findContours(binary_outline, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找到所有轮廓的边界框
    bboxes = [cv2.boundingRect(contour) for contour in contours]

    valid_crops = []

    for (x, y, w, h) in bboxes:
        # 计算可以裁剪的区域，确保裁剪区域在轮廓内
        for i in range(max(0, x), min(x + w - crop_size + 1, image_np.shape[1] - crop_size + 1)):
            for j in range(max(0, y), min(y + h - crop_size + 1, image_np.shape[0] - crop_size + 1)):
                valid_crops.append((i, j))

    if valid_crops:
        # 随机选择一个有效的裁剪位置
        idx = random.randint(0, len(valid_crops) - 1)
        x, y = valid_crops[idx]

        # 裁剪图像
        cropped_image = image_np[y:y + crop_size, x:x + crop_size]



        return cropped_image, (x, y, x + crop_size, y + crop_size)

    else:
        # 进行中心裁剪
        center_x = image_np.shape[1] // 2
        center_y = image_np.shape[0] // 2

        # 计算裁剪区域的起始点
        x = max(center_x - crop_size // 2, 0)
        y = max(center_y - crop_size // 2, 0)

        # 确保裁剪区域不超出图像边界
        x = min(x, image_np.shape[1] - crop_size)
        y = min(y, image_np.shape[0] - crop_size)

        # 裁剪图像
        cropped_image = image_np[y:y + crop_size, x:x + crop_size]

        # 在原图上标记裁剪区域
        # cv2.rectangle(image_np, (x, y), (x + crop_size, y + crop_size), color=(0, 255, 0), thickness=2)

        # 保存标记后的原图
        # cv2.imwrite("crop_1.png", image_np)

        return cropped_image , (x, y, x + crop_size, y + crop_size)
    


# Process the dataset by loading info from a JSON file, which includes image files, image labels, feature files, keypoint coordinates.
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, json_file, tokenizer,  size=256, center_crop=True,
                 t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05, image_root_path=""):
        super().__init__()

        self.tokenizer = tokenizer
        self.size = size
        self.center_crop = center_crop
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path

        self.data = []
        with open(json_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.design_transforms = transforms.Compose(
            [
                
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.conditioning_image_transforms = transforms.Compose(
            [
                transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.size),
                transforms.ToTensor(),
            ]
        )

        self.clip_image_processor = CLIPImageProcessor()
    

    def __getitem__(self, idx):
        
        item = self.data[idx]
        image_file = item["target"]
        if "prompt_db" in item:

            text = item["prompt_db"]
        else:
            text = item["prompt"]
        '''
        #对文本进行切片 分成六类：概念、“服装长度”>“袖长”>“袖型”>“衣领型”>“下摆型”
        # 分割文本
        elements = text.split(",")

        # 获取倒数第5个元素的索引
        split_index = len(elements) - 5

        # 前一类
        first_category = ", ".join(elements[:split_index])

        categories = [first_category]  # 首先加入前一类
        for i in range(split_index, len(elements)):
            current_category = ", ".join(elements[:i + 1])  # 当前类包含前面的所有类
            categories.append(current_category)

        text_0, text_1, text_2, text_3, text_4, text_5 = categories[0], categories[1], categories[2], categories[3], categories[4], categories[5]
        '''
 
        # print(mask_image.shape)
        #条件：sketch+crop
        if "sketch" in item and "colors" not in item :  
            image_path =  os.path.join(self.image_root_path, image_file)
            new_string = "contour"
            contour_path = re.sub(r'/(.*)/([^/]+)$', lambda m: f'{m.group(0).rsplit("/", 2)[0]}/{new_string}/{m.group(2)}',image_file)
            # print(1)
            outline_path = os.path.join(self.image_root_path, contour_path) 
            # print("contour_path:",contour_path)
            control_add , crop_bbox = random_crop_with_outline(image_path, outline_path)
            design_img = control_add
            # print("use crop")
            # control_add = Image.open(os.path.join(self.image_root_path, item["crop"]))
            # texture_file = item["sketch"]
            texture_file = item["sketch"].replace("sketch", "new_sketch")

        
        #条件:sketch + colors:
        elif "sketch" in item and "colors" in item:
            style_img = os.path.join(self.image_root_path,item["colors"])
            control_add = Image.open(style_img)
            design_img = np.array(control_add.resize((64, 64)).convert("RGB"))
            crop_bbox = (96,96,160,160)
            # texture_file = item["sketch"]
            texture_file = item["sketch"].replace("sketch", "new_sketch")


        #条件：contour + crop
        elif "contour" in item:
            image_path =  os.path.join(self.image_root_path, image_file) 
            outline_path = os.path.join(self.image_root_path, item["contour"]) 
            control_add , crop_bbox = random_crop_with_outline(image_path, outline_path)
            # control_add = Image.open(os.path.join(self.image_root_path,item["crop"]))
            # print("use crop")
            design_img = control_add
            
            texture_file = item["contour"]
        
        # print("control_add:" ,control_add)

        try:
             # read image
            raw_image = Image.open(os.path.join(self.image_root_path, image_file))
            
            texture_image = Image.open(os.path.join(self.image_root_path, texture_file))


                # original size
            original_width, original_height = raw_image.size
            original_size = torch.tensor([original_height, original_width])

                # transform raw_image and kps_image
            image = self.image_transforms(raw_image.convert("RGB"))
                
            kps_image = self.conditioning_image_transforms(texture_image.convert("RGB"))
            # print("image", image.shape)
            design_img = self.design_transforms(design_img)

            # print("design_img: ",design_img.shape)
            clip_image = self.clip_image_processor(images=control_add, return_tensors="pt").pixel_values


                # set cfg drop rate
            drop_feature_embed = 0
            drop_text_embed = 0
            rand_num = random.random()
            if rand_num < self.i_drop_rate:
                drop_feature_embed = 1
            elif rand_num < (self.i_drop_rate + self.t_drop_rate):
                drop_text_embed = 1
            elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
                drop_text_embed = 1
                drop_feature_embed = 1

                # CFG process
            if drop_text_embed:
                # text_0=text_1=text_2=text_3=text_4=text_5  = ""
                text = ""
            if drop_feature_embed:
                clip_image = torch.zeros_like(clip_image)

            # get text and tokenize 六类
            '''
            text_input_ids_0 = self.tokenizer(
                        text_0,
                        max_length=self.tokenizer.model_max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
            ).input_ids

            text_input_ids_1 = self.tokenizer(
                        text_1,
                        max_length=self.tokenizer.model_max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
            ).input_ids

            text_input_ids_2 = self.tokenizer(
                        text_2,
                        max_length=self.tokenizer.model_max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
            ).input_ids
            
            text_input_ids_3 = self.tokenizer(
                        text_3,
                        max_length=self.tokenizer.model_max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
            ).input_ids

            text_input_ids_4 = self.tokenizer(
                        text_4,
                        max_length=self.tokenizer.model_max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
            ).input_ids

            text_input_ids_5 = self.tokenizer(
                        text_5,
                        max_length=self.tokenizer.model_max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
            ).input_ids
            '''
            text_input_ids = self.tokenizer(
                        text,
                        max_length=self.tokenizer.model_max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
            ).input_ids

            crop_bbox = torch.tensor(crop_bbox)

            return {
                    "image": image,
                    "design_img":design_img, 
                    "kps_image": kps_image,
                    "text_input_ids": text_input_ids,
                    "crop_bbox": crop_bbox, 

                    # "face_id_embed": face_id_embed,
                    "clip_image": clip_image,
                    "original_size": original_size,
                    # "crop_coords_top_left": crop_coords_top_left,
                    "target_size": torch.tensor([self.size, self.size]),
                    "drop_feature_embed":drop_feature_embed
            }

       
        except Exception as e:
            print(f"Error occurred while processing {image_file}: {e}")
            return None


    def __len__(self):
        return len(self.data)


def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    design_imgs = torch.stack([example["design_img"] for example in data])

    kps_images = torch.stack([example["kps_image"] for example in data])
    '''
    text_input_ids_0 = torch.cat([example["text_input_ids_0"] for example in data], dim=0)
    text_input_ids_1 = torch.cat([example["text_input_ids_1"] for example in data], dim=0)
    text_input_ids_2 = torch.cat([example["text_input_ids_2"] for example in data], dim=0)
    text_input_ids_3 = torch.cat([example["text_input_ids_3"] for example in data], dim=0)
    text_input_ids_4 = torch.cat([example["text_input_ids_4"] for example in data], dim=0)
    text_input_ids_5 = torch.cat([example["text_input_ids_5"] for example in data], dim=0)
    '''
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    
    # face_id_embed = torch.stack([example["face_id_embed"] for example in data])
    clip_images = torch.cat([example["clip_image"] for example in data], dim=0)
    # clip_images = torch.stack([example["clip_image"] for example in data])
    original_size = torch.stack([example["original_size"] for example in data])
    # crop_coords_top_left = torch.stack([example["crop_coords_top_left"] for example in data])
    target_size = torch.stack([example["target_size"] for example in data])
    drop_feature_embeds = [example["drop_feature_embed"] for example in data]
    crop_bboxs = torch.stack([example["crop_bbox"] for example in data])


    return {
        "images": images,
        "design_imgs" : design_imgs,
        "kps_images": kps_images,
        "text_input_ids": text_input_ids,
 
        # "face_id_embed": face_id_embed,
        "clip_images": clip_images,
        "original_size": original_size,
        # "crop_coords_top_left": crop_coords_top_left,
        "target_size": target_size,
        "drop_feature_embeds":drop_feature_embeds,
        "crop_bboxs":crop_bboxs,

    }

# LGF
class LGFBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()

        # 用 concat 后的特征预测 gate
        self.gate = torch.nn.Sequential(
            torch.nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, f_style, f_struct):
        """
        f_style  : [B,C,H,W]  (StyleNet)
        f_struct : [B,C,H,W]  (ControlNet)
        """

        x = torch.cat([f_style, f_struct], dim=1)
        G = self.gate(x)   # [B,C,H,W] ∈ [0,1]

        fused = G * f_style + (1 - G) * f_struct
        return fused


class InstantIDAdapter(torch.nn.Module):
    """InstantIDAdapter"""
    def __init__(self, unet, controlnet, feature_proj_model, id_proj_emded, adapter_modules, lgf,mid_lgf, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        self.feature_proj_model = feature_proj_model
        self.id_proj_emded = id_proj_emded
        self.adapter_modules = adapter_modules
        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)
        self.unet.enable_xformers_memory_efficient_attention()
        self.unet.enable_gradient_checkpointing()
        self.controlnet.enable_xformers_memory_efficient_attention()
        self.controlnet.enable_gradient_checkpointing()
        self.lgf = lgf
        self.mid_lgf = mid_lgf



    def forward(self,noisy_latents, timesteps, encoder_hidden_states,  feature_embeds, img_vae,  controlnet_image):
        #encoder_hidden_states就是个文本数组
        id_embedding = self.feature_proj_model(feature_embeds, img_vae)
        # print("id_embedding:" , id_embedding.shape) #torch.Size([1, 16, 768])
        control_latents = encoder_hidden_states #就是文本

        #id_embde的中间特征 风格特征
        id_down_samples, id_mid_samples = self.id_proj_emded(id_embedding)
        
        # ControlNet conditioning. 结构特征
        down_block_res_samples, mid_block_res_sample = self.controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=control_latents,  # text feature
                controlnet_cond=controlnet_image,  # canny image
                return_dict=False,
        )

        """
        down_block_res_samples: torch.Size([1, 320, 32, 32])
        down_block_res_samples 1: torch.Size([1, 320, 16, 16])
        down_block_res_samples 2: torch.Size([1, 640, 8, 8])
        down_block_res_samples 3: torch.Size([1, 1280, 4, 4])
        mid_block_res_sample: torch.Size([1, 1280, 4, 4])
        """
      
       
        # print('mid_block_res_sample:', mid_block_res_sample.shape)
            # Predict the noise residual.
        '''noise_pred = self.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=[sample for sample in down_block_res_samples],
                mid_block_additional_residual=mid_block_res_sample,
                
        
        ).sample'''

        # for i in id_down_samples:
        #     print(i.shape)

        # 进行分别相加 LGF
        # combined_samples = [id_down_samples[i] + down_block_res_samples[i] for i in range(len(id_down_samples))]

        # lgf
        
        combined_samples = []


        for i in range(len(id_down_samples)):
            # print(
            #     f"[LGF] scale {i}:",
            #     id_down_samples[i].shape,
            #     down_block_res_samples[i].shape,
            # )
            fused = self.lgf[i](
                id_down_samples[i],
                down_block_res_samples[i],
            )

            combined_samples.append(fused)


        # 拼接 id_mid_samples 和 mid_block_res_sample
        # combined_mid_samples = id_mid_samples + mid_block_res_sample # 沿通道维度拼接
        # lgf
        combined_mid_samples = self.mid_lgf(
            id_mid_samples,
            mid_block_res_sample,
        )


        # print('mid_block_res_sample:', combined_mid_samples.shape)

        # print(timesteps)
        # 计算每个阶段的timesteps比例
        
        # stage_timesteps_01 = int(timesteps.item() * 0.1) 
        '''
        # Determine proportions for encoder hidden states
        encoder_proportions = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]
        cumulative_proportions = torch.cumsum(torch.tensor(encoder_proportions), dim=0)

        # Generate weighted encoder hidden states
        weighted_encoder_states = []
        for i, proportion in enumerate(encoder_proportions):
            weighted_state = encoder_hidden_states[i] * proportion
            weighted_encoder_states.append(weighted_state)

        # Combine weighted encoder states
        combined_encoder_states = sum(weighted_encoder_states)

        # Pass through U-Net
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=combined_encoder_states,
            down_block_additional_residuals=[sample for sample in combined_samples],
            mid_block_additional_residual=combined_mid_samples,
        ).sample

        return noise_pred
        '''
        encoder_hidden_states = torch.cat([encoder_hidden_states, id_embedding], dim=1)

        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,  #
            down_block_additional_residuals=[sample for sample in combined_samples],
            mid_block_additional_residual=combined_mid_samples,
                
        
        ).sample
            
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.feature_proj_model.parameters()]))
        orig_id_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.id_proj_emded.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Check if 'latents' exists in both the saved state_dict and the current model's state_dict
        strict_load_feature_proj_model = True
        if "latents" in state_dict["image_proj"] and "latents" in self.feature_proj_model.state_dict():
            # Check if the shapes are mismatched
            if state_dict["image_proj"]["latents"].shape != self.feature_proj_model.state_dict()["latents"].shape:
                print(f"Shapes of 'image_proj.latents' in checkpoint {ckpt_path} and current model do not match.")
                print("Removing 'latents' from checkpoint and loading the rest of the weights.")
                del state_dict["image_proj"]["latents"]
                strict_load_feature_proj_model = False

        # Load state dict for feature_proj_model and adapter_modules
        self.feature_proj_model.load_state_dict(state_dict["image_proj"], strict=strict_load_feature_proj_model)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)
        self.id_proj_emded.load_state_dict(state_dict["id_proj_emded"], strict=True)
        if "lgf" in state_dict:
            self.lgf.load_state_dict(state_dict["lgf"], strict=True)
            self.mid_lgf.load_state_dict(state_dict["mid_lgf"], strict=True)



        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.feature_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))
        new_id_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.id_proj_emded.parameters()]))


        # Verify if the weights have changed
        # assert orig_ip_proj_sum == new_ip_proj_sum, "Weights of feature_proj_model did not change!"
        # assert orig_adapter_sum == new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")




def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default=None,
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model. If not specified weights are initialized from unet.",
    )

    parser.add_argument(
        "--num_tokens",
        type=int,
        default=16,
        help="Number of tokens to query from the CLIP image encoding.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=10,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--data_json_file",
        type=str,
        default=None,
        required=True,
        help="Training data",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="",
        required=True,
        help="Training data root path",
    )
    # parser.add_argument('--clip_proc_mode',
    #                     choices=["seg_align", "seg_crop", "orig_align", "orig_crop", "seg_align_pad",
    #                              "orig_align_pad"],
    #                     default="orig_crop",
    #                     help='The mode to preprocess clip image encoder input.')

    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to CLIP image encoder",
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-ip_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )


    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    parser.add_argument(
        "--save_epoch",
        type=float,
        default=2,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )

    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--noise_offset", type=float, default=None, help="noise offset")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def plot_loss(step_losses,img_dir):
    plt.figure()
    #if len(step_losses) > 10000:
        #step_losses = step_losses[::]
    plt.plot(step_losses,'b',label = 'loss')        
    plt.ylabel('loss') 
    plt.xlabel('per 100 step')
    plt.legend()
    plt.savefig(os.path.join(img_dir,"train_loss_5e-5.jpg")) 


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )

    num_devices = accelerator.num_processes

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
    if args.controlnet_model_name_or_path:
        print("Loading existing controlnet weights")
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    else:
        print("Initializing controlnet weights from unet")
        controlnet = ControlNetModel.from_unet(unet)



    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)

    #controlnet只有下采样没有上采样，下采样之后是0卷积层
    controlnet.requires_grad_(True)
    # controlnet.requires_grad_(False)

    controlnet.train()

    # ip-adapter: insightface feature
    num_tokens = 16

    # feature_proj_model = Resampler(
    #     dim=1280,
    #     depth=4,
    #     dim_head=64,
    #     heads=20,
    #     num_queries=num_tokens,
    #     embedding_dim=512,
    #     output_dim=unet.config.cross_attention_dim,
    #     ff_mult=4,
    # )
    feature_proj_model = Resampler_new(
        # dim=1280,
        dim=unet.config.cross_attention_dim,#768
        depth=4,
        dim_head=64,
        heads=12,
        num_queries=num_tokens,
        # embedding_dim=224,
        embedding_dim=image_encoder.config.hidden_size,
        output_dim=unet.config.cross_attention_dim,
        ff_mult=4
    )
    # 在初始化处修改
    # 假设是 SDXL 的通道配置：
    down_channels = [
    320, 320, 320,   # Scale 1
    320, 640, 640,   # Scale 2 (注意：SD1.5 这里第一个是 320，后两个是 640)
    640, 1280, 1280, # Scale 3
    1280, 1280, 1280 # Scale 4
]


    lgf = torch.nn.ModuleList([
        LGFBlock(ch) for ch in down_channels
    ])
    # 这里的 mid_lgf 对应中间层，通常是 1280
    # self.mid_lgf = LGFBlock(1280)
    # lgf =  torch.nn.ModuleList([
    #         LGFBlock(320),
    #         LGFBlock(320),
    #         LGFBlock(640),
    #         LGFBlock(1280),
    #     ])
    mid_lgf = LGFBlock(1280)

    # print("unet.config.cross_attention_dim:",unet.config.cross_attention_dim)  
    # print("image_encoder.config.hidden_size:",image_encoder.config.hidden_size)

    #id_embed 下采样
    id_proj_emded = DownsampleNetwork()

    # init adapter modules
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]

            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            # print("weights:",weights)
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=num_tokens)
            attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    
    # Instantiate InstantIDAdapter from pretrained model or from scratch.
    ip_adapter = InstantIDAdapter(unet, controlnet, feature_proj_model, id_proj_emded, adapter_modules, lgf, mid_lgf, args.pretrained_ip_adapter_path )
    
    # Register a hook function to process the state of a specific module before saving.
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # find instance of InstantIDAdapter Model.
            for i, model_instance in enumerate(models):
                if isinstance(model_instance, InstantIDAdapter):
                    # When saving a checkpoint, only save the ip-adapter and image_proj, do not save the unet.
                    ip_adapter_state = {
                        'image_proj': model_instance.feature_proj_model.state_dict(),
                        'ip_adapter': model_instance.adapter_modules.state_dict(),
                        "id_proj_emded" : model_instance.id_proj_emded.state_dict(),
                        "lgf" : model_instance.lgf.state_dict(),
                        "mid_lgf" : model_instance.mid_lgf.state_dict(),


                    }
                    torch.save(ip_adapter_state, os.path.join(output_dir, 'pytorch_model.bin'))
                    print(f"IP-Adapter Model weights saved in {os.path.join(output_dir, 'pytorch_model.bin')}")
                    # Save controlnet separately.
                    sub_dir = "controlnet"
                    model_instance.controlnet.save_pretrained(os.path.join(output_dir, sub_dir))
                    print(f"Controlnet weights saved in {os.path.join(output_dir, sub_dir)}")  
                    # Remove the corresponding weights from the weights list because they have been saved separately.
                    # Remember not to delete the corresponding model, otherwise, you will not be able to save the model
                    # starting from the second epoch.
                    weights.pop(i)
                    break

    def load_model_hook(models, input_dir):
        # find instance of InstantIDAdapter Model.
        while len(models) > 0:
            model_instance = models.pop()
            if isinstance(model_instance, InstantIDAdapter):
                ip_adapter_path = os.path.join(input_dir, 'pytorch_model.bin')
                if os.path.exists(ip_adapter_path):
                    ip_adapter_state = torch.load(ip_adapter_path)
                    model_instance.feature_proj_model.load_state_dict(ip_adapter_state['image_proj'])
                    model_instance.adapter_modules.load_state_dict(ip_adapter_state['ip_adapter'])
                    model_instance.id_proj_emded.load_state_dict(ip_adapter_state['id_proj_emded'])

                    model_instance.lgf.load_state_dict(ip_adapter_state['lgf'])
                    model_instance.mid_lgf.load_state_dict(ip_adapter_state['mid_lgf'])


                    sub_dir = "controlnet"
                    model_instance.controlnet.from_pretrained(os.path.join(input_dir, sub_dir))
                    print(f"Model weights loaded from {ip_adapter_path}")
                else:
                    print(f"No saved weights found at {ip_adapter_path}")


    # Register hook functions for saving  and loading.
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    weight_dtype = torch.float32 
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # vae.to(accelerator.device, dtype=weight_dtype)
    # text_encoder.to(accelerator.device, dtype=weight_dtype)
    # image_encoder.to(accelerator.device, dtype=weight_dtype)
    # controlnet.to(accelerator.device, dtype=weight_dtype)  # error
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    
    unet.enable_xformers_memory_efficient_attention()
    unet.enable_gradient_checkpointing()
    controlnet.to(accelerator.device)

    # trainable params
    params_to_opt = itertools.chain(ip_adapter.feature_proj_model.parameters(),
                                    ip_adapter.id_proj_emded.parameters(),
                                    ip_adapter.adapter_modules.parameters(),
                                    ip_adapter.controlnet.parameters(),
                                    ip_adapter.lgf.parameters(),
                                    ip_adapter.mid_lgf.parameters(),

                                    )

    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)

    # dataloader
    train_dataset = MyDataset(args.data_json_file, tokenizer=tokenizer,size=args.resolution,
                              center_crop=args.center_crop, image_root_path=args.data_root_path)
    total_data_size = len(train_dataset)

    print("read traindataset: {}".format(total_data_size))
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Prepare everything with our `accelerator`.
    ip_adapter, optimizer, train_dataloader = accelerator.prepare(ip_adapter, optimizer, train_dataloader)
    
    step_per_epoch = int(len(train_dataset)/args.train_batch_size/num_devices)
    total_step =  int(len(train_dataset) / args.train_batch_size / num_devices)
    save_step = int(step_per_epoch*args.save_epoch)
    print("batch: {} gpu:{} step_per_epoch: {} save epoch:{}, save step:{})".format(args.train_batch_size,num_devices,step_per_epoch,args.save_epoch,save_step))

    # # Restore checkpoints
    # checkpoint_folders = [folder for folder in os.listdir(args.output_dir) if folder.startswith('checkpoint-')]
    # if checkpoint_folders:
    #     # Extract step numbers from all checkpoints and find the maximum step number
    #     global_step = max(int(folder.split('-')[-1]) for folder in checkpoint_folders if folder.split('-')[-1].isdigit())
    #     checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    #     # Load the checkpoint
    #     accelerator.load_state(checkpoint_path)
    # else:
    #     global_step = 0
    #     print("No checkpoint folders found.")
    global_step = 0
    train_noise_loss = []
    # Calculate steps per epoch and the current epoch and its step number
    # steps_per_epoch = total_data_size // (args.train_batch_size * num_devices)
    # current_epoch = global_step // steps_per_epoch
    # current_step_in_epoch = global_step % steps_per_epoch

    # Training loop
    # train_noise_loss = []
    noise_loss_for_draw = []
    noise_loss_for_draw_step = []
    noise_crop_loss_for_draw_step = []

    
    print("start train")
    for epoch in range(0, args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(ip_adapter):
                # ip_adapter = torch.nn.parallel.DistributedDataParallel(ip_adapter, find_unused_parameters=True)
                # Convert images to latent space
                vae.to(accelerator.device)
                with torch.no_grad():
                    # vae of sdxl should use fp32
                    # print("images",batch["images"].shape)  #([16, 3, 256, 256])
                    latents = vae.encode(
                        batch["images"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                    # print("images latents",latents.shape)  #([16, 4, 32, 32])
                    latents = latents * vae.config.scaling_factor
                    latents = latents.to(accelerator.device, dtype=weight_dtype)

                    #disign
                    design_vae = vae.encode(
                        batch["design_imgs"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                    # print("images latents",latents.shape)  #([16, 4, 32, 32])
                    design_vae = design_vae * vae.config.scaling_factor
                    design_vae = design_vae.to(accelerator.device, dtype=weight_dtype)  #torch.Size([16, 4, 8, 8])
                    # print(design_vae.shape)
                # vae.to("cpu")

          
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1)).to(
                        accelerator.device, dtype=weight_dtype)

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(10, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                # print("time:", timesteps)  #随机整数

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process) 
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # get feature embeddings, with cfg
                # image_embeds = batch["clip_images"].to(accelerator.device, dtype=weight_dtype)
                kps_images = batch["kps_images"].to(accelerator.device, dtype=weight_dtype)
                # print('kps_images:', kps_images.shape)

                # for other experiments
                clip_images = []
                for clip_image, drop_image_embed in zip(batch["clip_images"], batch["drop_feature_embeds"]):
                    if drop_image_embed == 1:
                        clip_images.append(torch.zeros_like(clip_image))
                    else:
                        clip_images.append(clip_image)
                clip_images = torch.stack(clip_images, dim=0)
                # print("clip_images:",clip_images.size())
                image_encoder.to(accelerator.device)
                
                
                with torch.no_grad():
                    image_embeds = image_encoder(clip_images.to(accelerator.device, dtype=weight_dtype),
                                                 output_hidden_states=True).hidden_states[-2]
                # print("image_embeds:",image_embeds.size())  #torch.Size([1, 257, 1280])
                image_encoder.to("cpu")
                
                text_encoder.to(accelerator.device)
                text_embeds = []
                with torch.no_grad():
                    text_embeds = text_encoder(batch['text_input_ids'].to(accelerator.device))[0]
                   

                text_encoder.to("cpu")
                    # 添加所有嵌入到 text_embeds 列表中
                # text_embeds.extend([text_embeds_0, text_embeds_1, text_embeds_2, text_embeds_3, text_embeds_4, text_embeds_5])


                # add cond
                # add_time_ids = [
                #     batch["original_size"].to(accelerator.device),
                #     batch["crop_coords_top_left"].to(accelerator.device),
                #     batch["target_size"].to(accelerator.device),
                # ]
                # add_time_ids = torch.cat(add_time_ids, dim=1).to(accelerator.device, dtype=weight_dtype)
                # # unet_added_cond_kwargs = { "time_ids": add_time_ids}
                

                # print("start denoise")
                noise_pred = ip_adapter(noisy_latents, timesteps, text_embeds, image_embeds, design_vae , kps_images)

                # CDA loss
                # ======================================================

                red_loss_total = 0.0
                num_red = 0

                for m in ip_adapter.adapter_modules:
                    if hasattr(m, "latest_red_loss") and m.latest_red_loss is not None:
                        red_loss_total = red_loss_total + m.latest_red_loss
                        num_red += 1

                if num_red > 0:
                    red_loss_total = red_loss_total / num_red
                else:
                    red_loss_total = torch.tensor(
                        0.0,
                        device=noise_pred.device,
                        dtype=noise_pred.dtype,
                    )
                                # print("text_embeds:",text_embeds.size())
                        

                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                """
                Loss
                """
                pred_original_sample = [
                    noise_scheduler.step(noise, t, noisy_latent).pred_original_sample.to(weight_dtype) \
                        for (noise, t, noisy_latent) in zip(noise_pred, timesteps, noisy_latents)
                ]
                pred_original_sample = torch.stack(pred_original_sample)

                # Map the denoised latents into RGB images
                pred_original_sample = 1 / vae.config.scaling_factor * pred_original_sample
                image = vae.decode(pred_original_sample.to(weight_dtype)).sample
                vae.to("cpu")

                # print(image)
                # 假设 image 是一个 PyTorch 张量
                images = (image / 2 + 0.5).clamp(0, 1)  #torch.Size([1, 3, 256, 256])


                masks = batch["crop_bboxs"]  # 假设形状为 (1, x, y, x+crop_size, y+crop_size)
                gt_images = batch["images"]


                def crop_image_loss(image, gt_image, mask):
                    # 确保输入张量范围在 [0, 1]
                    image = (image * 255).byte()  # 转换为 [0, 255] 的 uint8 张量
                    batch_size = image.shape[0]
                    losses = []  # 用于存储每个批次的损失

                    # 遍历批次中的每个 mask
                    for i in range(batch_size):
                        bbox = mask[i]  # 获取当前批次中的 bbox
                        # print(f"Processing batch {i}: {bbox}")

                        # 解包坐标
                        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]  # 解包坐标

                        # 确保裁剪区域在图像范围内
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(x2, image.shape[3])
                        y2 = min(y2, image.shape[2])

                        # 裁剪当前批次的图像和 gt_image
                        cropped_image = image[i, :, y1:y2, x1:x2]  # 从 image 中裁剪
                        gt_cropped_image = gt_image[i, :, y1:y2, x1:x2]  # 从 gt_image 中裁剪

                        # 计算损失
                        loss = F.mse_loss(cropped_image.float() / 255.0, gt_cropped_image.float())  # 转换为 [0, 1] 范围
                        losses.append(loss)  # 存储损失值

                    total_loss = sum(losses) / len(losses)   # 计算平均损失
                    return total_loss  # 返回总损失


                crop_loss = crop_image_loss(images, gt_images, masks)

                loss = loss + crop_loss + red_loss_total

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                crop_loss = accelerator.gather(crop_loss.repeat(args.train_batch_size)).mean().item()

                
                # train_noise_loss.append(avg_loss)
                noise_loss_for_draw_step.append(avg_loss)
                noise_crop_loss_for_draw_step.append(crop_loss)

                # train_noise_loss.append(avg_loss)
                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                now = datetime.now()
                formatted_time = now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                # if accelerator.is_main_process and step % 10 == 0:
                if accelerator.is_main_process :
                    print("[{}]: Epoch {}, global_step {}, step {}, data_time: {}, time: {}, step_loss: {}, crop_loss: {}".format(
                        formatted_time, epoch, global_step, step, load_data_time, time.perf_counter() - begin,
                        avg_loss, crop_loss))
                    if global_step % 100==0:
                        noise_loss_for_draw.append(np.mean(noise_loss_for_draw_step))
                        noise_loss_for_draw_step = []
                        plot_loss(noise_loss_for_draw,args.output_dir)
    

            global_step += 1
            if accelerator.is_main_process and global_step % save_step == 0:
                save_ori = False
                if global_step==args.save_steps:
                    save_ori = True
                # before saving state, check if this save would set us over the `checkpoints_total_limit`
                if args.checkpoints_total_limit is not None:
                    checkpoints = os.listdir(args.output_dir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                    if len(checkpoints) >= args.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]
                        print(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                        print(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)

                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)

                # trainloss, testloss = log_validation(vae,args.pretrained_model_name_or_path,save_path,args,global_step,save_ori)
                # output_file = 'fashion_vali/test_loss.txt'
                
                # with open(output_file, 'a') as f:
                #     f.write("step:{},train_loss:{},test_loss:{}\n".format(global_step, trainloss, testloss))

            begin = time.perf_counter()


if __name__ == "__main__":
    main()
