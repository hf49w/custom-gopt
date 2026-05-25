# -*- coding: utf-8 -*-
# 合并版：单条数据从 Kaldi feat.scp + scores.json 直接生成可用 tensor
# 逻辑来源：load_feats.py + gen_seq_data_phn.py 三个版本

import os
import sys
import json,re
import numpy as np
import kaldi_io
import torch
from src.models import GOPT
from collections import OrderedDict


from utils import load_human_scores, load_phone_symbol_table

# =====================[ 1. 路径配置 —— 按需修改 ]=====================

# ===== 手动指定 Kaldi recipe 的根目录，并切过去 =====
BASE_DIR = "/mnt/d/研究生/智能体/kaldi/egs/gop_speechocean762/s5"
os.chdir(BASE_DIR)
print("Current working dir:", os.getcwd())

# 默认的 phone 分数（没有人工打分时）
DEFAULT_PHONE_SCORE = 0.0

# 默认的句子级分数（没有人工打分时）
DEFAULT_UTT_SCORE = {
    "accuracy": 0.0,
    "completeness": 0.0,
    "fluency": 0.0,
    "prosodic": 0.0,
    "total": 0.0,
}

# 默认的词级分数（没有人工打分时）
DEFAULT_WORD_ACC = 0.0
DEFAULT_WORD_STRESS = 0.0
DEFAULT_WORD_TOTAL = 0.0
DEFAULT_WORD_ID = 0
DEFAULT_WORD_TEXT = ""


# TODO: 把下面这几个路径改成你自己的
#FEATURE_SCP = "/mnt/d/研究生/智能体/kaldi/egs/gop_speechocean762/s5/exp/gop_mydata/feat.scp"            # 只包含你这一条数据的 feat.scp
FEATURE_SCP = "exp/gop_mydata/feat.scp"
PHONE_SYMBOL_TABLE = "/mnt/d/研究生/智能体/kaldi/egs/gop_speechocean762/s5/data/lang_nosp/phones-pure.txt"
HUMAN_SCORING_JSON = "data/local/scores.json"      # 含 word/phone 标注的 JSON
UTT_SCORE_JSON = "data/local/scores.json"          # 句子级分数（若和上面是同一文件就共用）

# phone idx 的过滤范围（和原代码一致）
MIN_PHONE_IDX = -1
MAX_PHONE_IDX = 999

# phone 分数下限（floor）
FLOOR = 0.1

# 若你想保持原代码“序列长度固定 50”，把下面改成 50
MAX_SEQ_LEN = 50   # None 表示按当前 utterance 的最大 phone index 动态长度

KALDI_GOP_ROOT = "/mnt/d/研究生/智能体/kaldi/egs/gop_speechocean762/s5"
PART = "mydata"



# =====================[ 2. 一些辅助函数 ]=====================

def build_word_level_maps(human_scoring_json_path):
    """
    从 scores.json 中构造每个 phone key -> 词级信息的映射：
    key: utt_id.phn_id
    value: word_text / word_id / accuracy / stress / total
    对应你第一个 load_feats.py 里的逻辑。
    """
    with open(human_scoring_json_path, "r", encoding="utf-8") as f:
        info = json.loads(f.read())

    word_of = {}      # key -> word 文本
    word_id_of = {}   # key -> word 在该句子中的序号
    acc_of = {}       # key -> word accuracy
    stress_of = {}    # key -> word stress
    total_of = {}     # key -> word total

    for utt in info:
        phone_num = 0
        for word_id, word in enumerate(info[utt]["words"]):
            cur_word_text = word["text"]
            cur_word_accuracy = word["accuracy"]
            cur_word_stress = word["stress"]
            cur_word_total = word["total"]
            assert len(word["phones"]) == len(word["phones-accuracy"])

            for _ in word["phones"]:
                key = f"{utt}.{phone_num}"
                phone_num += 1
                word_of[key] = cur_word_text
                word_id_of[key] = word_id
                acc_of[key] = cur_word_accuracy
                stress_of[key] = cur_word_stress
                total_of[key] = cur_word_total

    return word_of, word_id_of, acc_of, stress_of, total_of


def get_utt_list_and_max_tok_id(keys):
    """
    根据 key 列表（utt_id.phn_id）得到：
      - utt_ids: 按出现顺序的 utterance id 列表
      - utt_cnt: utterance 数量
      - max_tok_id: 最大的 phone index（整数）
    """
    utt_ids = []
    max_tok_id = -1

    for k in keys:
        utt_id, tok_str = k.split(".")
        tok_id = int(tok_str)
        max_tok_id = max(max_tok_id, tok_id)
        if not utt_ids or utt_id != utt_ids[-1]:
            utt_ids.append(utt_id)

    utt_cnt = len(utt_ids)
    return utt_ids, utt_cnt, max_tok_id

# =====================[ 3. 从 Kaldi + JSON 抽一条数据出来 ]=====================
def build_word_mapping_from_generated_files(
    feat_scp_path: str,
    phone_symbol_table: str,
    text_path: str,
    lexicon_path: str,
    strip_stress: bool = True,
):
    """
    根据 Kaldi 生成的文件，构造：
      - word_of[key]: 每个 phone key (utt.ph_idx) 属于哪个单词文本
      - word_id_of[key]: 每个 phone key 属于该句子的第几个单词 (0,1,2,...)

    需要的文件：
      - feat_scp_path: exp/gop_mydata/feat.scp
      - phone_symbol_table: data/lang_nosp/phones-pure.txt
      - text_path: data/mydata/text
      - lexicon_path: data/local/lexicon.txt
    """

    # 1) phone id -> symbol
    _, phone_int2sym = load_phone_symbol_table(phone_symbol_table)

    # 2) 读 feat.scp，得到每个 utt 对应的 phone 序列（GOP 用的顺序）
    #    feat_phones[utt_id] = [(phone_index, key, ph_sym), ...]，phone_index 用 key 里的 .后面的数字
    feat_phones = {}
    
    for key, feat in kaldi_io.read_vec_flt_scp(feat_scp_path):
        utt_id, tok_str = key.split(".")
        tok_id = int(tok_str)
        ph_id = int(feat[0])
        ph_sym = phone_int2sym.get(ph_id, str(ph_id))
        feat_phones.setdefault(utt_id, []).append((tok_id, key, ph_sym))

    # 按 phone_index 排序，确保是 0,1,2,... 的顺序
    #print("feat_phones:",feat_phones)
    for utt in feat_phones:
        feat_phones[utt].sort(key=lambda x: x[0])

    # 3) 读 data/mydata/text -> utt_id -> [word1, word2, ...]
    utt2words = {}
    with open(text_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            utt_id = parts[0]
            words = parts[1:]
            utt2words[utt_id] = words

    # 4) 读 lexicon.txt -> word -> [ph1, ph2, ...]
    #    如果有多条发音，只用第一条
    lexicon = {}
    with open(lexicon_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            word = parts[0]
            phones = parts[1:]
            if strip_stress:
                # 把 AH0 -> AH 这种重音数字去掉，视情况保留或修改
                phones = [re.sub(r"\d$", "", p) for p in phones]
            if word not in lexicon:
                lexicon[word] = phones

    word_of = {}
    word_id_of = {}

    # 5) 对每个 utt 对齐 canonical phone 序列和 GOP 序列
    for utt_id, seq in feat_phones.items():
        if utt_id not in utt2words:
            print(f"Warning: {utt_id} not in {text_path}, skip.")
            continue

        words = utt2words[utt_id]

        # 展开 canonical phone 序列
        canon_phones = []      # [ph1, ph2, ...]
        canon_word_info = []   # [(word_id, word_text), ...] 与 canon_phones 对齐
        for wid, w in enumerate(words):
            if w not in lexicon:
                raise KeyError(f"Word '{w}' not in lexicon {lexicon_path}")
            phs = lexicon[w]
            for ph in phs:
                canon_phones.append(ph)
                canon_word_info.append((wid, w))

        # GOP 序列中的 phone 符号
        gop_phones = [ph_sym for (_, _, ph_sym) in seq]

        if len(canon_phones) != len(gop_phones):
            print(
                f"Warning: utt {utt_id} canonical phones len={len(canon_phones)}, "
                f"GOP phones len={len(gop_phones)}. Will align by min length."
            )

        L = min(len(canon_phones), len(gop_phones))

        for idx in range(L):
            tok_id, key, gop_ph = seq[idx]
            canon_ph = canon_phones[idx]
            if canon_ph != gop_ph:
                # 这里 phone 不匹配可以打印一下，但不强制报错
                print(
                    f"Phone mismatch at utt {utt_id}, pos {idx}: "
                    f"canon={canon_ph}, gop={gop_ph}"
                )
            wid, wtext = canon_word_info[idx]
            word_of[key] = wtext
            word_id_of[key] = wid
    print(word_of)
    print(word_id_of)
    return word_of, word_id_of


def extract_single_from_kaldi(
    feature_scp=FEATURE_SCP,
    phone_symbol_table=PHONE_SYMBOL_TABLE,
    human_scoring_json=HUMAN_SCORING_JSON,
    utt_score_json=UTT_SCORE_JSON,
    min_phone_idx=MIN_PHONE_IDX,
    max_phone_idx=MAX_PHONE_IDX,
    floor=FLOOR,
):
    """
    整合两个 load_feats.py 的逻辑：
      - 读取 Kaldi feat.scp，得到所有 phone 的特征向量
      - 若有 human_scoring_json，则用 load_human_scores 得到 phone-level 分数
      - 若有 JSON，则构造 word 级信息（accuracy / stress / total / word_id / word_text）
      - 若有 utt_score_json，则读入 utterance 级分数；否则用默认值
    返回：
      feat_arr: (N_phone, feat_dim)
      keys_arr: (N_phone,)，字符串数组，形如 utt_id.phn_id
      labels_phn: list of [ph_symbol(str), score(float)]
      labels_word: list of [ph_symbol(str), word_id(int), word_text(str),
                            acc(float), stress(float), total(float)]
      utt2score: dict[utt_id] -> {accuracy, completeness, fluency, prosodic, total}
    """
    # phone int -> symbol
    _, phone_int2sym = load_phone_symbol_table(phone_symbol_table)

    # ================== human_scoring_json 是否存在 ==================
    have_phone_scores = None

    if have_phone_scores:
        # phone-level score（带 floor）
        score_of, phone_of = load_human_scores(human_scoring_json, floor=floor)
        # word-level 映射
        word_of, word_id_of, acc_of, stress_of, total_of = build_word_level_maps(
            human_scoring_json
        )
    else:
        # # 没有人工 phone/word 评分，全部用默认值
        # score_of = None
        # phone_of = None
        # word_of, word_id_of, acc_of, stress_of, total_of = {}, {}, {}, {}, {}
        # print("Info: human_scoring_json 未提供或不存在，使用默认 phone/word 分数。")
        # 根据 Kaldi 生成的文件计算 word_of / word_id_of
        
        feat_scp_path = os.path.join(
            KALDI_GOP_ROOT, "exp", f"gop_{PART}", "feat.scp"
        )
        phone_symbol_table = os.path.join(
            KALDI_GOP_ROOT, "data", "lang_nosp", "phones-pure.txt"
        )
        text_path = os.path.join(
            KALDI_GOP_ROOT, "data", PART, "text"
        )
        lexicon_path = os.path.join(
            KALDI_GOP_ROOT, "data", "local", "lexicon.txt"
        )
        word_of_auto, word_id_of_auto = build_word_mapping_from_generated_files(
            feat_scp_path,
            phone_symbol_table,
            text_path,
            lexicon_path,
        )

    # ================== utt_score_json 是否存在 ==================
    have_utt_scores = None

    if have_utt_scores:
        with open(utt_score_json, "r", encoding="utf-8") as f:
            utt2score = json.loads(f.read())
    else:
        # 先占位，后面根据 keys 再填默认值
        utt2score = None
        print("Info: utt_score_json 未提供或不存在，使用默认句子级分数。")

    features = []
    keys = []
    labels_phn = []
    labels_word = []

    cnt = 0
    for key, feat in kaldi_io.read_vec_flt_scp(feature_scp):
        cnt += 1

        ph = int(feat[0])
        if ph < min_phone_idx or ph > max_phone_idx:
            continue

        if phone_int2sym is not None and ph in phone_int2sym:
            ph_sym = str(phone_int2sym[ph])
        else:
            ph_sym = str(ph)

        # ===== phone-level 分数 =====
        if have_phone_scores:
            if key not in score_of:
                print(f"Warning: no human phone score for {key}, 用默认值 {DEFAULT_PHONE_SCORE}")
                phone_score = DEFAULT_PHONE_SCORE
            else:
                phone_score = float(score_of[key])
        else:
            phone_score = DEFAULT_PHONE_SCORE


        wtext = word_of_auto.get(key, DEFAULT_WORD_TEXT)
        wid = word_id_of_auto.get(key, DEFAULT_WORD_ID)
        labels_word.append(
                [
                    ph_sym,
                    int(wid),
                    wtext,
                    DEFAULT_WORD_ACC,
                    DEFAULT_WORD_STRESS,
                    DEFAULT_WORD_TOTAL,
                ]
            )


        # 特征 + phone 级标签
        features.append(feat)
        keys.append(key)
        labels_phn.append([ph_sym, phone_score])

    if len(features) == 0:
        raise RuntimeError(
            "No valid phones found from scp + scores.json/默认分数，请检查 feature_scp 或过滤条件。"
        )

    feat_arr = np.stack(features, axis=0)              # (N, feat_dim)
    keys_arr = np.array(keys, dtype=str)               # (N,)
    labels_phn = np.array(labels_phn, dtype=object)    # (N, 2)
    labels_word = np.array(labels_word, dtype=object)  # (N, 6)

    # ===== 若没有句子级分数，根据 keys 构造 utt2score，全部用默认值 =====
    if utt2score is None:
        utt2score = {}
        for k in keys_arr:
            utt_id = k.split(".")[0]
            if utt_id not in utt2score:
                # 拷一份默认 dict，避免引用同一个对象
                utt2score[utt_id] = dict(DEFAULT_UTT_SCORE)

    print(f"Loaded {feat_arr.shape[0]} phone frames from {feature_scp}")

    return feat_arr, keys_arr, labels_phn, labels_word, utt2score


# =====================[ 4. 三种序列化：phone / utt / word ]=====================

def process_feat_seq_phn(feat, keys, labels_phn, phn_dict, max_seq_len=None):
    """
    对应第一个 gen_seq_data_phn.py 的 process_feat_seq：
      - 输入：feat(N, D)，keys(N,), labels_phn(N, 2: [ph_symbol, score])
      - 输出：
          seq_feat: [utt_cnt, seq_len, feat_dim]
          seq_label: [utt_cnt, seq_len, 2]，[..., 0] = phone_id，[..., 1] = score
    """
    utt_ids, utt_cnt, max_tok_id = get_utt_list_and_max_tok_id(keys)

    feat_dim = feat.shape[1] - 1  # 第一维是 phone id，本身不算特征

    if max_seq_len is None:
        seq_len = max_tok_id + 1
    else:
        if max_tok_id + 1 > max_seq_len:
            raise ValueError(
                f"当前 utterance 最大 phone index = {max_tok_id}，"
                f"需要的 seq_len = {max_tok_id + 1} > MAX_SEQ_LEN={max_seq_len}，"
                "请调大 MAX_SEQ_LEN。"
            )
        seq_len = max_seq_len

    print(f"In total utterance number: {utt_cnt}, seq_len = {seq_len}")

    seq_feat = np.zeros([utt_cnt, seq_len, feat_dim], dtype=np.float32)
    # -1 表示 padding token
    seq_label = np.zeros([utt_cnt, seq_len, 2], dtype=np.float32) - 1

    prev_utt_id = keys[0].split(".")[0]
    row = 0

    for i in range(feat.shape[0]):
        cur_utt_id, cur_tok_id = keys[i].split(".")[0], int(keys[i].split(".")[1])

        if cur_utt_id != prev_utt_id:
            row += 1
            prev_utt_id = cur_utt_id

        # 特征：去掉 feat[0] 的 phone id
        seq_feat[row, cur_tok_id, :] = feat[i, 1:]

        # label[utt, seq, 0] = phone id（整型索引）
        ph_symbol = labels_phn[i, 0]
        seq_label[row, cur_tok_id, 0] = phn_dict[ph_symbol]
        # label[utt, seq, 1] = phone score
        seq_label[row, cur_tok_id, 1] = float(labels_phn[i, 1])

    return seq_feat, seq_label


def process_feat_seq_utt(keys, utt2score):
    """
    对应第二个 gen_seq_data_phn.py 的 process_feat_seq_utt：
    返回每个 utterance 一行的句子级标签：
      seq_label_utt: [utt_cnt, 5] = [accuracy, completeness, fluency, prosodic, total]
    """
    utt_ids, utt_cnt, _ = get_utt_list_and_max_tok_id(keys)
    print(f"In total utterance number (utt-level): {utt_cnt}")

    seq_label = np.zeros([utt_cnt, 5], dtype=np.float32)

    for row, utt_id in enumerate(utt_ids):
        if utt_id not in utt2score:
            raise KeyError(f"utt2score 中找不到 {utt_id}")
        seq_label[row, 0] = utt2score[utt_id]["accuracy"]
        seq_label[row, 1] = utt2score[utt_id]["completeness"]
        seq_label[row, 2] = utt2score[utt_id]["fluency"]
        seq_label[row, 3] = utt2score[utt_id]["prosodic"]
        seq_label[row, 4] = utt2score[utt_id]["total"]

    return seq_label


def process_feat_seq_word(keys, labels_word, max_seq_len=None):
    """
    对应第三个 gen_seq_data_phn.py 的 process_feat_seq_word：
    labels_word: 每行 [ph_symbol, word_id, word_text, acc, stress, total]
    返回：
      seq_label_word: [utt_cnt, seq_len, 4]
        [..., :, 0:3] = [accuracy, stress, total]
        [..., :, 3]   = word_id
    """
    utt_ids, utt_cnt, max_tok_id = get_utt_list_and_max_tok_id(keys)

    if max_seq_len is None:
        seq_len = max_tok_id + 1
    else:
        if max_tok_id + 1 > max_seq_len:
            raise ValueError(
                f"当前 utterance 最大 phone index = {max_tok_id}，"
                f"需要的 seq_len = {max_tok_id + 1} > MAX_SEQ_LEN={max_seq_len}，"
                "请调大 MAX_SEQ_LEN。"
            )
        seq_len = max_seq_len

    print(f"In total utterance number (word-level): {utt_cnt}, seq_len = {seq_len}")

    # -1 表示 n/a
    seq_label = np.zeros([utt_cnt, seq_len, 4], dtype=np.float32) - 1

    prev_utt_id = keys[0].split(".")[0]
    row = 0

    # 关键修改：循环长度以 keys 为准（或者取最小值）
    n = min(len(keys), labels_word.shape[0])
    if len(keys) != labels_word.shape[0]:
        print(
            f"Warning: len(keys)={len(keys)} != "
            f"labels_word.shape[0]={labels_word.shape[0]}, use n={n}"
        )

    for i in range(n):
        cur_utt_id, cur_tok_id = keys[i].split(".")[0], int(keys[i].split(".")[1])

        if cur_utt_id != prev_utt_id:
            row += 1
            prev_utt_id = cur_utt_id

        # labels_word[i] = [ph_symbol, word_id, word_text, acc, stress, total]
        acc = float(labels_word[i, 3])
        stress = float(labels_word[i, 4])
        total = float(labels_word[i, 5])
        word_id = int(labels_word[i, 1])

        seq_label[row, cur_tok_id, 0:3] = [acc, stress, total]
        seq_label[row, cur_tok_id, 3] = word_id

    return seq_label

# =====================[ 5. 一键跑完、返回 tensor ]=====================

def prepare_single_sample_tensors():
    """
    整个 pipeline:
      Kaldi feat.scp + scores.json
        -> feat_arr, keys_arr, labels_phn, labels_word, utt2score
        -> seq_feat, seq_label_phn, seq_label_utt, seq_label_word
        -> 转成 PyTorch tensor
    """
    # 1) 读 scp + json
    feat_arr, keys_arr, labels_phn, labels_word, utt2score = extract_single_from_kaldi(human_scoring_json=None,
    utt_score_json=None,)

    # 2) 生成 phone 字典
    phn_dict = json.load(open("/mnt/d/研究生/智能体/gopt/phn_dict.json"))

    print("Phone dict (phn -> id):", phn_dict)

    # 3) 序列化：phone 级
    seq_feat_np, seq_label_phn_np = process_feat_seq_phn(
        feat_arr,
        keys_arr,
        labels_phn,
        phn_dict,
        max_seq_len=MAX_SEQ_LEN,
    )

    # 4) 序列化：utt 级
    seq_label_utt_np = process_feat_seq_utt(keys_arr, utt2score)

    # 5) 序列化：word 级
    seq_label_word_np = process_feat_seq_word(
        keys_arr,
        labels_word,
        max_seq_len=MAX_SEQ_LEN,
    )

    # 6) 转成 PyTorch tensor（你可以按需拆成 int / float）
    seq_feat = torch.from_numpy(seq_feat_np).float()          # [1, T, feat_dim]
    seq_label_phn = torch.from_numpy(seq_label_phn_np).float()   # [1, T, 2]
    seq_label_utt = torch.from_numpy(seq_label_utt_np).float()   # [1, 5]
    seq_label_word = torch.from_numpy(seq_label_word_np).float() # [1, T, 4]

    return {
        "seq_feat": seq_feat,
        "seq_label_phn": seq_label_phn,
        "seq_label_utt": seq_label_utt,
        "seq_label_word": seq_label_word,
        "phn_dict": phn_dict,
        "keys": keys_arr,
    }



# if __name__ == "__main__":
#     data = prepare_single_sample_tensors()
#     print("seq_feat shape:", data["seq_feat"].shape)
#     print("seq_label_phn shape:", data["seq_label_phn"].shape)
#     print("seq_label_utt shape:", data["seq_label_utt"].shape)
#     print("seq_label_word shape:", data["seq_label_word"].shape)


#     sys.path.append(os.path.abspath('../src/'))

#     # ==== 1) 准备数据 ====
#     seq_feat        = data["seq_feat"]        # [1, T, 84]
#     seq_label_phn   = data["seq_label_phn"]   # [1, T, 2]
#     seq_label_utt   = data["seq_label_utt"]   # [1, 5]
#     seq_label_word  = data["seq_label_word"]  # [1, T, 4]
#     keys            = data["keys"]

#     audio_input = seq_feat.clone()                    # [1, T, 84]
#     phns        = seq_label_phn[:, :, 0].long()       # [1, T]

#     def norm_valid(feat, norm_mean, norm_std):
#         norm_feat = torch.zeros_like(feat)
#         for i in range(feat.shape[0]):
#             for j in range(feat.shape[1]):
#                 if feat[i, j, 0] != 0:
#                     norm_feat[i, j, :] = (feat[i, j, :] - norm_mean) / norm_std
#                 else:
#                     break
#         return norm_feat
#     # 归一化（librispeech 的配置）
#     NORM_MEAN, NORM_STD = 3.203, 4.045
#     audio_input = norm_valid(audio_input, NORM_MEAN, NORM_STD)

#     # ==== 2) 强制使用 CPU ====
#     device = torch.device("cpu")

#     input_dim = audio_input.shape[-1]  # 84

#     # 不用 DataParallel，直接单卡模型
#     gopt = GOPT(embed_dim=24, num_heads=1, depth=3, input_dim=input_dim)

#     # 读取预训练权重
#     ckpt_path = "/mnt/d/研究生/智能体/gopt/pretrained_models/gopt_librispeech/best_audio_model.pth"
#     sd = torch.load(ckpt_path, map_location="cpu")

#     # 去掉 "module." 前缀（因为训练时用了 DataParallel）
#     new_sd = OrderedDict()
#     for k, v in sd.items():
#         if k.startswith("module."):
#             new_k = k[len("module."):]
#         else:
#             new_k = k
#         new_sd[new_k] = v

#     gopt.load_state_dict(new_sd, strict=True)
#     gopt = gopt.to(device)
#     gopt.eval()

#     # ==== 3) 前向推理 ====
#     print(audio_input)
#     print(phns)
#     with torch.no_grad():
#         audio_input_dev = audio_input.to(device)   # [1, T, 84]
#         phns_dev        = phns.to(device)         # [1, T]

#         u1, u2, u3, u4, u5, p, w1, w2, w3 = gopt(audio_input_dev, phns_dev)

#     #==== 4) 拿结果 ====
#     # phone-level 分数
#     p_pred = p.squeeze(0).squeeze(-1).cpu().numpy()   # [T]
#     print("Phone-level scores:")
#     for k, s in zip(keys, p_pred):
#         print(k, float(s))

#     # utterance-level 分数
#     utt_pred = torch.cat((u1, u2, u3, u4, u5), dim=1).squeeze(0).cpu().numpy()  # [5]
#     print("UTT-level scores (normalized 0~2):", utt_pred)
#     print("UTT-level scores (scaled back x5):", utt_pred * 5.0)

#     # word-level 分数（phone 粒度的 3 维输出）
#     word_pred = torch.cat((w1, w2, w3), dim=2).squeeze(0).cpu().numpy()        # [T, 3]
#     print("Word-level phone outputs shape:", word_pred.shape)
#     print(word_pred * 5.0)


    
if __name__ == "__main__":
    data = prepare_single_sample_tensors()
    print("seq_feat shape:", data["seq_feat"].shape)
    print("seq_label_phn shape:", data["seq_label_phn"].shape)
    print("seq_label_utt shape:", data["seq_label_utt"].shape)
    print("seq_label_word shape:", data["seq_label_word"].shape)

    sys.path.append(os.path.abspath('../src/'))

    # ==== 1) 准备数据 ====
    seq_feat        = data["seq_feat"]        # [1, T, 84]
    seq_label_phn   = data["seq_label_phn"]   # [1, T, 2]
    seq_label_utt   = data["seq_label_utt"]   # [1, 5]
    seq_label_word  = data["seq_label_word"]  # [1, T, 4]
    keys            = data["keys"]
    phn_dict        = data["phn_dict"]

    # phone-id -> phone 符号
    id2phn = {v: k for k, v in phn_dict.items()}

    # 只用 feat 作为模型输入；phns 只用来做 canonical phone embedding
    audio_input = seq_feat.clone()                    # [1, T, 84]
    phns        = seq_label_phn[:, :, 0].long()       # [1, T]

    # ==== norm_valid：和原始 GoPDataset 保持一致，只归一化有效 token ====
    def norm_valid(feat, norm_mean, norm_std):
        norm_feat = torch.zeros_like(feat)
        for i in range(feat.shape[0]):
            for j in range(feat.shape[1]):
                # feat[...,0] == 0 表示 padding
                if feat[i, j, 0] != 0:
                    norm_feat[i, j, :] = (feat[i, j, :] - norm_mean) / norm_std
                else:
                    break
        return norm_feat

    # 归一化（librispeech 的配置）
    NORM_MEAN, NORM_STD = 3.203, 4.045
    audio_input = norm_valid(audio_input, NORM_MEAN, NORM_STD)

    # ==== 2) 使用 CPU ====
    device = torch.device("cpu")
    input_dim = audio_input.shape[-1]  # 84

    # 不用 DataParallel，单卡模型
    gopt = GOPT(embed_dim=24, num_heads=1, depth=3, input_dim=input_dim)

    # 读取预训练权重
    ckpt_path = "/mnt/d/研究生/智能体/gopt/pretrained_models/gopt_librispeech/best_audio_model.pth"
    sd = torch.load(ckpt_path, map_location="cpu")

    # 去掉 "module." 前缀（因为训练时用了 DataParallel）
    from collections import OrderedDict
    new_sd = OrderedDict()
    for k, v in sd.items():
        if k.startswith("module."):
            new_k = k[len("module."):]
        else:
            new_k = k
        new_sd[new_k] = v

    gopt.load_state_dict(new_sd, strict=True)
    gopt = gopt.to(device)
    gopt.eval()

    # ==== 一些评分解释函数（中文） ====
    def describe_phone(score2: float) -> str:
        """音素 0~2 打分的中文解释"""
        if score2 >= 1.7:
            return "发音正确，接近母语水平"
        elif score2 >= 1.1:
            return "发音基本正确，但口音较重"
        elif score2 >= 0.5:
            return "发音偏差较大，需要改进"
        else:
            return "发音错误或几乎听不见"

    def describe_word_accuracy(score10: float) -> str:
        """单词准确度 0~10 的中文解释"""
        if score10 >= 9:
            return "这个词的发音非常完美"
        elif score10 >= 7:
            return "大多数音素发音正确，但带有比较明显的口音"
        elif score10 >= 4:
            return "有不少音素发音错误，但基本能听懂"
        elif score10 >= 2:
            return "超过 30% 的音素发音错误，或者读成了别的词"
        elif score10 > 0:
            return "发音很难分辨"
        else:
            return "几乎没有发音"

    def describe_stress(label10: float) -> str:
        """重音 5/10 的中文解释"""
        if label10 >= 7.5:
            return "重音位置正确"
        else:
            return "重音位置错误"

    def describe_utt_accuracy(score10: float) -> str:
        if score10 >= 9:
            return "整体发音非常出色，几乎没有明显错误"
        elif score10 >= 7:
            return "整体发音良好，仅有少量错误"
        elif score10 >= 5:
            return "整体发音可以理解，但错误和口音比较多"
        elif score10 >= 3:
            return "整体发音很生硬，严重错误较多"
        else:
            return "整体发音极差，只能听出零星词语"

    def describe_fluency(score10: float) -> str:
        if score10 >= 8:
            return "非常流利，没有明显停顿或结巴"
        elif score10 >= 6:
            return "总体流利，只有少量停顿或重复"
        elif score10 >= 4:
            return "流利度一般，停顿和重复较多"
        else:
            return "非常不流利，停顿和结巴很多"

    def describe_prosody(score10: float) -> str:
        if score10 >= 9:
            return "语调自然，节奏感好，接近母语人士"
        elif score10 >= 7:
            return "语调基本自然，节奏较好"
        elif score10 >= 5:
            return "语调和节奏一般，略显生硬"
        else:
            return "语调差、节奏感差，听起来比较吃力"

    # ==== 3) 前向推理 ====
    with torch.no_grad():
        audio_input_dev = audio_input.to(device)   # [1, T, 84]
        phns_dev        = phns.to(device)         # [1, T]

        u1, u2, u3, u4, u5, p, w1, w2, w3 = gopt(audio_input_dev, phns_dev)

    # ====== 4.1 句子文本与词序列 ======
    # 从 key 里拿出 utt_id
    utt_id = keys[0].split(".")[0]
    text_path = os.path.join(BASE_DIR, "data", PART, "text")
    utt_words = []
    with open(text_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == utt_id:
                utt_words = parts[1:]
                break

    print("\n======================")
    print(f"句子 ID: {utt_id}")
    if utt_words:
        print("句子内容:", " ".join(utt_words))
    else:
        print("句子内容: （在 text 文件中未找到对应 utt_id）")

    # ====== 4.2 phone-level 分数 ======
    p_pred = p.squeeze(0).squeeze(-1).cpu().numpy()   # [T]
    phns_np = phns.squeeze(0).cpu().numpy()           # [T]
    word_ids_tok = seq_label_word[0, :, 3].cpu().numpy().astype(int)  # [T]

    # ====== 4.3 utterance-level 分数（按 0~2 → 0~10 / 完整度 0~1 映射） ======
    utt_pred_norm = torch.cat((u1, u2, u3, u4, u5), dim=1).squeeze(0).cpu().numpy()  # [5]
    utt_pred_norm = np.clip(utt_pred_norm, 0.0, 2.0)

    utt_acc_norm, utt_comp_norm, utt_flu_norm, utt_pros_norm, utt_total_norm = utt_pred_norm

    utt_acc_10   = utt_acc_norm   * 5.0
    # 完整度原始范围 0~1，这里用 0~2 线性映射回 0~1
    utt_comp_01  = np.clip(utt_comp_norm / 2.0, 0.0, 1.0)
    utt_flu_10   = utt_flu_norm   * 5.0
    utt_pros_10  = utt_pros_norm  * 5.0
    utt_total_10 = utt_total_norm * 5.0

    print("\n====== 句子级评分 ======")
    print(f"准确度: {utt_acc_10:.1f} / 10  —— {describe_utt_accuracy(utt_acc_10)}")
    print(f"完整度: {utt_comp_01:.2f} （约 {utt_comp_01*100:.1f}% 单词发音较好）")
    print(f"流利度: {utt_flu_10:.1f} / 10 —— {describe_fluency(utt_flu_10)}")
    print(f"韵律:   {utt_pros_10:.1f} / 10 —— {describe_prosody(utt_pros_10)}")
    print(f"总分:   {utt_total_10:.1f} / 10")

    # ====== 4.4 word-level：按单词聚合 token 的 3 维输出 ======
    word_pred_tok = torch.cat((w1, w2, w3), dim=2).squeeze(0).cpu().numpy()  # [T, 3]
    # 对每个 token 先截断到 [0,2]，再 *5 → 0~10
    word_pred_tok = np.clip(word_pred_tok, 0.0, 2.0)

    # 汇总到每个单词：wid -> list of token_scores
    word_scores = {}  # wid -> [ [acc, stress, total], ... ]
    max_wid = -1
    for t in range(word_ids_tok.shape[0]):
        wid = int(word_ids_tok[t])
        if wid < 0:
            break  # 后面都是 padding
        max_wid = max(max_wid, wid)
        word_scores.setdefault(wid, []).append(word_pred_tok[t, :])

    # 平均得到每个单词的 3 维得分（0~2），再 *5
    word_scores_avg = {}  # wid -> (acc10, stress10, total10)
    for wid, lst in word_scores.items():
        avg_norm = np.mean(np.stack(lst, axis=0), axis=0)   # [3] 0~2
        acc10   = float(avg_norm[0] * 5.0)
        stress10 = float(avg_norm[1] * 5.0)
        total10  = float(avg_norm[2] * 5.0)

        # stress 近似离散到 {5,10}
        stress_label = 10.0 if stress10 >= 7.5 else 5.0
        word_scores_avg[wid] = (acc10, stress_label, total10)

    print("\n====== 逐词评分 ======")
    if utt_words and max_wid + 1 != len(utt_words):
        print(f"[警告] 词数不匹配：word_id 最大值={max_wid}，但句子里只有 {len(utt_words)} 个词。")

    for wid in range(max_wid + 1):
        word_text = utt_words[wid] if utt_words and wid < len(utt_words) else f"<word_{wid}>"
        if wid not in word_scores_avg:
            print(f"[第 {wid+1} 个词] {word_text}: 没有预测到对应的音素（可能是对齐问题）")
            continue

        acc10, stress_label, total10 = word_scores_avg[wid]
        print(f"[第 {wid+1} 个词] {word_text}")
        print(f"  准确度: {acc10:.1f} / 10  —— {describe_word_accuracy(acc10)}")
        print(f"  重音:   {stress_label:.0f}  —— {describe_stress(stress_label)}")
        print(f"  综合得分: {total10:.1f} / 10")

    # ====== 4.5 逐音素评分（按单词分组） ======
    print("\n====== 逐音素评分 ======")
    for wid in range(max_wid + 1):
        word_text = utt_words[wid] if utt_words and wid < len(utt_words) else f"<word_{wid}>"
        print(f"\n【第 {wid+1} 个词】{word_text}")
        for t in range(word_ids_tok.shape[0]):
            if int(word_ids_tok[t]) != wid:
                continue
            ph_id = int(phns_np[t])
            if ph_id < 0:
                continue
            ph_sym = id2phn.get(ph_id, str(ph_id))
            ph_score_norm = float(np.clip(p_pred[t], 0.0, 2.0))
            print(f"  音素 {ph_sym:<5s}: {ph_score_norm:.2f} / 2  —— {describe_phone(ph_score_norm)}")

    print("\n===== 评测完成 =====")



