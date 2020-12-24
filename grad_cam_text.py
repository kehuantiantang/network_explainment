# coding=utf-8

import argparse
import pprint
import cv2
import numpy as np
import torch
from torch.autograd import Function
import html
import pandas as pd
from NBOW import NBOW_CONV
from utils.text_utils import read_process_text_input

device = torch.device("cuda")
class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            else:
                x = module(x)

        return target_activations, x


def show_cam_on_text(text, mask):
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



class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):

        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args





def get_model(args, model_path):
    model = NBOW_CONV(max_seq_length = args.max_seq_length, embedding_dim = args.embedding_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path)['model'].state_dict())
    model.to(device)
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CAM')
    parser.add_argument('--use_cuda', default=True, type = bool)
    parser.add_argument('--target_index', default=None, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--model_path', default='/home/khtt/code/network_explainment/vector/best_seq.pt', type=str)

    parser.add_argument('--root-path', default='/home/khtt/code/network_explainment/train_vocab.txt', type=str)
    parser.add_argument('--train-path', default='/home/khtt/dataset/na_experiment/aclImdb/train', type=str)
    parser.add_argument('--test-path', default='/home/khtt/dataset/na_experiment/aclImdb/test', type=str)

    parser.add_argument('--max-seq-length', default=120, type=int, help = 'vocabulary or sequence length')
    parser.add_argument('--embedding-dim', default=300, type=int, help = 'word2vector vector size')

    parser.add_argument('--target_docs', default=80, type=int)

    args = parser.parse_args()
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(args))


    model = get_model(args, args.model_path)

    print(model)
    grad_cam = GradCam(model=model, feature_module=model, \
                       target_layer_names=["conv_3"], use_cuda=args.use_cuda)


    x, y, text = read_process_text_input(args)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = None
    mask = grad_cam(x, args.target_index)
    #
    show_cam_on_text(text, mask)


    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    print(model._modules.items())
    gb = gb_model(x, index=args.target_index)
    gb = gb.transpose((1, 2, 0))
    # cam_mask = cv2.merge([mask, mask, mask])
    # cam_gb = deprocess_image(cam_mask*gb)
    # gb = deprocess_image(gb)
    #
    #
    # cv2.imwrite('gb.jpg', gb)
    # cv2.imwrite('cam_gb.jpg', cam_gb)