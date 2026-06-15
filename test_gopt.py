# 假设你在 gopt 根目录
import torch
import sys
sys.path.append('src')

from models.gopt import GOPT   # 具体 import 路径以文件结构为准














gopt = GOPT(embed_dim=24, num_heads=1, depth=3, input_dim=84)
gopt = torch.nn.DataParallel(gopt)
sd = torch.load(r'D:\研究生\智能体\gopt\pretrained_models\gopt_librispeech\best_audio_model.pth', map_location='cpu')
gopt.load_state_dict(sd, strict=True)

# 3. 准备你的输入 GOP 特征
# x: [batch, seq_len, feat_dim]，例如 [N, 50, 84]
# phn: [batch, seq_len, phn_num]，可选
x = ...   # 从你自己的 .npy / .pkl / .pt 文件读
phn = ... # 如果你打算用 canonical phone 特征

with torch.no_grad():
    outputs = model(x, phn)   # 输出是 [u1,u2,u3,u4,u5,p,w1,w2,w3]
