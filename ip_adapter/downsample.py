
import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

class DownsampleNetwork(nn.Module):
    def __init__(self, in_channels=16, block_out_channels=[320, 640, 1280]):
        super(DownsampleNetwork, self).__init__()

        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)
        # self.proj_out = nn.Linear(768, 1024) #  256分辨率
        self.proj_out = nn.Linear(768, 4096) #512

        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(block_out_channels[0], block_out_channels[0]),  # 320 channels
            EncoderBlock(block_out_channels[0], block_out_channels[1]),  # 640 channels
            EncoderBlock(block_out_channels[1], block_out_channels[2]),  # 1280 channels
        ])

        self.middle_block = nn.Conv2d(block_out_channels[2], block_out_channels[2], kernel_size=3, padding=1)

    def forward(self, x):
        outputs = []

        # 输入变形为 [1, 16, 32, 32]
        x = self.proj_out(x)
        # x = x.view(x.size(0), 16, 32, 32)
        x = x.view(x.size(0), 16, 64, 64)


        # 第一层卷积
        x = self.conv_in(x)  # 输出 [1, 320, 32, 32]
        outputs.append(x)  # 第一组
        outputs.append(x)  # 第二组
        outputs.append(x)  # 第三组

        # 第一次下采样
        x = self.encoder_blocks[0](x)  # 输出 [1, 320, 16, 16]
        outputs.append(x)  # 第四组

        # 第二次下采样
        outputs.append(self.encoder_blocks[1].conv1(x))  # 第五组（640通道，16x16）
        outputs.append(self.encoder_blocks[1].conv1(x))  # 第六组（640通道，16x16）
        x = self.encoder_blocks[1](x)  # 输出 [1, 640, 8, 8]
        outputs.append(x)  # 第七组

        # 第三次下采样
        outputs.append(self.encoder_blocks[2].conv1(x))  # 第八组（1280通道，8x8）
        outputs.append(self.encoder_blocks[2].conv1(x))  # 第九组（1280通道，8x8）
        x = self.encoder_blocks[2](x)  # 输出 [1, 1280, 4, 4]
        outputs.append(x)  # 第十组

        # 第四次下采样
        outputs.append(x)  # 第十一组（1280通道，4x4）
        outputs.append(x)  # 第十二组（1280通道，4x4）

        mid_output = F.relu(self.middle_block(x))  # 使用最后一个编码器块的输出
        return outputs, mid_output




# # 假设输入向量为 [1, 16, 768]
# input_vector = torch.randn(1, 16, 768)

# # 将输入调整为适合卷积的形状
# # 这里我们需要将768的维度转换为合适的空间维度
# # 768可以转换为32x24的形状
# # input_vector = input_vector.view(1, 16, 32, 32)  # 变为 [1, 16, 24, 32]

# # 创建模型并进行前向传播
# model = DownsampleNetwork()
# outputs, mid_output = model(input_vector)

# # 输出每个阶段的形状
# for i, output in enumerate(outputs):
#     print(f"down_block_res_samples {i}: {output.shape}")
# print("mid_block_res_sample:", mid_output.shape)
# """
# down_block_res_samples 0: torch.Size([1, 320, 24, 32])
# down_block_res_samples 1: torch.Size([1, 320, 12, 16])
# down_block_res_samples 2: torch.Size([1, 640, 6, 8])
# down_block_res_samples 3: torch.Size([1, 1280, 3, 4])
# mid_block_res_sample: torch.Size([1, 1280, 3, 4])
# """

