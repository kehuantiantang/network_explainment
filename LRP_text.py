# coding=utf-8
import argparse
import pprint

import torch
from LRP1.innvestigator import InnvestigateModel
from grad_cam_text import get_model
from utils.text_utils import read_process_text_input
import pandas as pd
import html

device = torch.device("cpu")

def show_mask_on_text(text, mask):
    weights = mask.cpu().numpy()
    df_coeff = pd.DataFrame(
        {'word': text,
         'num_code': weights
         })
    word_to_coeff_mapping = {}
    for row in df_coeff.iterrows():
        row = row[1]
        word_to_coeff_mapping[row[0]] = (row[1])


    max_alpha = 0.8
    highlighted_text = []
    for word in text.split():
        weight = word_to_coeff_mapping.get(word)

        if weight is not None:
            highlighted_text.append('<span style="background-color:rgba(135,206,250,' + str(weight / max_alpha) + ');">' + html.escape(word) + '</span>')
        else:
            highlighted_text.append(word)
    highlighted_text = ' '.join(highlighted_text)

    with open('./grad_cam_text.html', 'w') as f:
        f.write(highlighted_text)


def text_lrp(model, x, text):
    inn_model = InnvestigateModel(model, lrp_exponent=2,
                                  method="e-rule",
                                  beta=.5)
    model_prediction, heatmap = inn_model.innvestigate(in_tensor=x)

    show_mask_on_text(text, heatmap)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CAM')
    parser.add_argument('--use_cuda', default=True, type = bool)
    parser.add_argument('--target_index', default=None, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--model_path', default='/home/your_dir/code/network_explainment/vector/best_seq.pt', type=str)

    parser.add_argument('--root-path', default='/home/your_dir/code/network_explainment/train_vocab.txt', type=str)
    parser.add_argument('--train-path', default='/home/your_dir/dataset/na_experiment/aclImdb/train', type=str)
    parser.add_argument('--test-path', default='/home/your_dir/dataset/na_experiment/aclImdb/test', type=str)

    parser.add_argument('--max-seq-length', default=120, type=int, help = 'vocabulary or sequence length')
    parser.add_argument('--embedding-dim', default=300, type=int, help = 'word2vector vector size')

    parser.add_argument('--target_docs', default=80, type=int)

    args = parser.parse_args()
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(args))


    model = get_model(args, args.model_path)

    x, y, text = read_process_text_input(args)
