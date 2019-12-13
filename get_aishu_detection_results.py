import numpy as np
import torchvision
import time
import os, json
import copy
import pdb
import time
import argparse

import sys
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import model
import skimage.io
import skimage.transform
import skimage.color
import skimage
#from dataloader import collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
#from dataloader import UnNormalizer
class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def resize_image(image, min_side=320, max_side=320):
    rows, cols, cns = image.shape
    smallest_side = min(rows, cols)
    # rescale the image so the smallest side is min_side
    scale = 1.0 * min_side / smallest_side
    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = 1.0 * max_side / largest_side
    # resize the image with the computed scale
    #image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
    image = cv2.resize(image, (int(round(cols*scale)),int(round((rows*scale)))))
    rows, cols, cns = image.shape
    #print(rows, cols)
    pad_w = 32 - rows%32
    pad_h = 32 - cols%32
    new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
    new_image[:rows, :cols, :] = image.astype(np.float32)
    return torch.from_numpy(new_image), scale

def normalize_image(image):
    mean = np.array([[[0.485, 0.456, 0.406]]])
    std = np.array([[[0.229, 0.224, 0.225]]])
    norm_image =  (image.astype(np.float32)-mean)/std
    return norm_image
print(torch.__version__)
#assert torch.__version__.split('.')[1] == '4'
print('CUDA available: {}'.format(torch.cuda.is_available()))
parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
parser.add_argument('--modeldir', help='Path to model (.pt) file.')
parser.add_argument('--rootdir', help='Path to root dir')
parser.add_argument('--savedir', help='Path to save dir')

unnormalize = UnNormalizer()

def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

parser = parser.parse_args()
modeldir = parser.modeldir
rootdir = parser.rootdir
savedir = parser.savedir
namelist = os.listdir(rootdir)
st = time.time()
img_total = 0
model_list = os.listdir(modeldir)
for mid in range(len(model_list)):
    #if mid < 3 :
    #    continue
    model_file = "csv_layer04_retinanet_%d.pt"%mid
    model_path = os.path.join(modeldir, model_file)
    outdir = os.path.join(savedir, "coco_ep%d"%mid)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    retinanet = torch.load(model_path)
    use_gpu = True
    if use_gpu:
        retinanet = retinanet.cuda()
    retinanet.eval()

    for idx, imgdir in enumerate(namelist):
        #if idx <= 4:
        #    continue
        out_json_path = os.path.join(outdir,imgdir+".json")
        if os.path.isfile(out_json_path):
            continue
        imgdir = os.path.join(rootdir, imgdir, 'imgs')
        out_ann = dict()
        #imgdir = os.path.join(rootdir, imgdir)
        imnames = os.listdir(imgdir)
        for imid, imname in enumerate(imnames):
            #if imid >= 10:
            #    break
            all_bbs = []
            impath = os.path.join(imgdir,imname)
            print("process %d/%d %s/%s"%(imid,len(imnames),imgdir, imname))
            ori_img = cv2.imread(impath)
            ori_img = ori_img[:,:,::-1]
            image = ori_img.astype(np.float32)/255.0
            image = normalize_image(image)
            image, scale = resize_image(image)
            h,w,c = image.shape
            #print(h,w,c)
            im_tensor = torch.zeros(1,h,w,3)
            im_tensor[0,:h,:w,:] = image
            #print(im_tensor.shape)
            im_tensor = im_tensor.permute(0,3,1,2)
            with torch.no_grad():
                scores, classification, transformed_anchors = retinanet(im_tensor.cuda().float())
                scores = scores.cpu().numpy()
                #scores, transformed_anchors = retinanet(im_tensor.cuda().float())
                #print('Elapsed time: {}'.format(time.time()-st))
                idxs = np.where(scores>=0.0)
                for j in range(idxs[0].shape[0]):
                    bbox = transformed_anchors[idxs[0][j], :]
                    x1 = int(bbox[0]/scale)
                    score = scores[idxs[0][j]]
                    y1 = int(bbox[1]/scale)
                    x2 = int(bbox[2]/scale)
                    y2 = int(bbox[3]/scale)
                    predict_class = int(classification[idxs[0][j]])
                    score_str = "%.3f"%score
                    if predict_class == 0:
                        all_bbs.append([x1,y1,x2,y2,score_str])
            if len(all_bbs) == 0:
                continue
            out_ann[imname] = all_bbs
        #for imid, imname in enumerate(imnames):
        #    frame_id = int(imname[:-4])
        #    print("%d/%d, %s"%(imid, len(imnames),impath))
        with open(out_json_path,'w') as f:
            json.dump(out_ann, f, indent=2)
