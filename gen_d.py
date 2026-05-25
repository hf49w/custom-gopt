import numpy as np, json

tr_label = np.loadtxt("./tr_labels_phn.csv", delimiter=",", dtype=str)
phn_dict = {}
idx = 0
for i in range(tr_label.shape[0]):
    ph = tr_label[i,0]
    if ph not in phn_dict:
        phn_dict[ph] = idx
        idx += 1

json.dump(phn_dict, open("phn_dict.json","w"), ensure_ascii=False, indent=2)
