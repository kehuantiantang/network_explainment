# coding=utf-8

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import xmltodict
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries
from torchvision import models, transforms


def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


# xml to json
def load_json(xml_path):
    xml_file = open(xml_path, 'r')

    xml_str = xml_file.read()

    json = xmltodict.parse(xml_str)
    return json


# resize and take the center part of image to what our model expects
def get_input_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    return transf


def get_input_tensors(img):
    transf = get_input_transform()
    # unsqeeze converts single image to batch of 1
    return transf(img).unsqueeze(0)


# to take PIL image, resize and crop it
def get_pil_transform():
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)
    ])

    return transf


# take resized, cropped image and apply whitening.
def get_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    return transf


def batch_predict(images):
    model.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)

    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()


# main function

img = get_image('/home/khtt/code/LCFCN/datasets/download/pascal/VOCdevkit/VOC2007/JPEGImages/006223.jpg')
plt.subplot(2, 2, 1)
plt.title('origin image')
plt.imshow(img)

# Load the pretrained model for Resnet50 available in Pytorch.
model = models.vgg16(pretrained=True)
# Load label texts for ImageNet predictions
idx2label, cls2label, cls2idx = [], {}, {}
with open(os.path.abspath('./imagenet_class_index.json'), 'r') as read_file:
    class_idx = json.load(read_file)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}
    cls2idx = {class_idx[str(k)][0]: k for k in range(len(class_idx))}
# Get the predicition for our image.
img_t = get_input_tensors(img)
model.eval()
logits = model(img_t)
# get probabilities and class labels for top 5 predictions.
probs = F.softmax(logits, dim=1)
probs5 = probs.topk(5)
tuple((p, c, idx2label[c]) for p, c in zip(probs5[0][0].detach().numpy(), probs5[1][0].detach().numpy()))

pill_transf = get_pil_transform()
preprocess_transform = get_preprocess_transform()

test_pred = batch_predict([pill_transf(img)])
test_pred.squeeze().argmax()
# create explanation for this prediciton.
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(np.array(pill_transf(img)),
                                         batch_predict,  # classification function
                                         top_labels=5,
                                         hide_color=0,
                                         num_samples=1000)  # number of images that will be sent to classification function
# show the result
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5,
                                            hide_rest=False)
img_boundry1 = mark_boundaries(temp / 255.0, mask)
plt.subplot(2, 2, 2)
plt.title('boundry1')
plt.imshow(img_boundry1)
plt.savefig('boundary1.jpg')

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10,
                                            hide_rest=False)
img_boundry2 = mark_boundaries(temp / 255.0, mask)
plt.subplot(2, 2, 3)
plt.title('boundry2')
plt.imshow(img_boundry2)
plt.show()
plt.savefig('boundary2.jpg')
