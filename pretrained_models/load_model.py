# -*- coding: utf-8 -*-
# @Time    : 11/16/21 5:00 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : load_model.py

# sample code of loading a pretrained GOPT model

import torch
import sys
import os
sys.path.append(os.path.abspath('../src/'))
from models import GOPT

gopt = GOPT(embed_dim=24, num_heads=1, depth=3, input_dim=84)
# GOPT is trained with dataparallel, so it need to be wrapped with dataparallel even you have a single gpu or cpu
gopt = torch.nn.DataParallel(gopt)
model_path = os.path.join(os.path.dirname(__file__), 'gopt_librispeech', 'best_audio_model.pth')
sd = torch.load(model_path, map_location='cpu')
gopt.load_state_dict(sd, strict=True)
