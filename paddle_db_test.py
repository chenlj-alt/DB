# -*- coding: utf-8 -*-
# @Author  : chenlijuan
# @Time    : 2021/2/7 下午4:35
# @Desc    : 测试paddle DB det --> pytorch 模型

import torch
import numpy as np
import os
import sys
import cv2

from concern.config import Configurable, Config


from paddle import fluid


def _load_state(path):
    """
    加载paddlepaddle识别模型参数
    :param path:
    :return: dict, 参数名<->参数值
    """
    if os.path.exists(path + ".pdparams"):
        return fluid.io.load_program_state(path)
    else:
        return None


class Demo:
    def __init__(self, experiment, args):
        self.experiment = experiment
        experiment.load('evaluation', **args)

        self.args = args
        self.structure = experiment.structure
        self.model_path = self.args['model_path']

    def init_torch_tensor(self):
        # Use gpu or not
        torch.set_default_tensor_type('torch.FloatTensor')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')

    def init_model(self):
        model = self.structure.builder.build(self.device)
        return model

    # preprocess: the same way as paddle db
    def resize_image_type0(self, im):  # used by paddle db
        """
        resize image to a size multiple of 32 which is required by the network
        args:
            img(array): array with shape [h, w, c]
        return(tuple):
            img, (ratio_h, ratio_w)
        """
        max_side_len = 2400
        h, w, _ = im.shape

        resize_w = w
        resize_h = h

        # limit the max side
        if max(resize_h, resize_w) > max_side_len:
            if resize_h > resize_w:
                ratio = float(max_side_len) / resize_h
            else:
                ratio = float(max_side_len) / resize_w
        else:
            ratio = 1.
        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)
        if resize_h % 32 == 0:
            resize_h = resize_h
        elif resize_h // 32 <= 1:
            resize_h = 32
        else:
            resize_h = (resize_h // 32) * 32
        if resize_w % 32 == 0:
            resize_w = resize_w
        elif resize_w // 32 <= 1:
            resize_w = 32
        else:
            resize_w = (resize_w // 32) * 32
        try:
            if int(resize_w) <= 0 or int(resize_h) <= 0:
                return None, (None, None)
            im = cv2.resize(im, (int(resize_w), int(resize_h)))
        except:
            print(im.shape, resize_w, resize_h)
            sys.exit(0)
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return im, (ratio_h, ratio_w)

    # preprocess: the same way as paddle db
    def normalize(self, im):
        """
        Normalize image
        :param im: input image
        :return: Normalized image
        """
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        im = im.astype(np.float32, copy=False)
        im = im / 255
        im[:, :, 0] -= img_mean[0]
        im[:, :, 1] -= img_mean[1]
        im[:, :, 2] -= img_mean[2]
        im[:, :, 0] /= img_std[0]
        im[:, :, 1] /= img_std[1]
        im[:, :, 2] /= img_std[2]
        channel_swap = (2, 0, 1)
        im = im.transpose(channel_swap)
        return im

    def inference(self, image_path, visualize=False):
        # ===> model construct
        self.init_torch_tensor()
        model = self.init_model()

        # ===> load paddle db state_dict
        model.load_3rd_state_dict("paddle", _load_state(self.args["paddle_model_path"]))

        # ===> model eval
        model.eval()

        # ===> save pytorch model, can used for train
        # torch.save(model.state_dict(), "/share_sdb/clj/DB2/DB/det_model_transfer/ic15_resnet_vd_18.pth")

        # ===> image preprocess
        img = cv2.imread(image_path)
        img, (ratio_h, ratio_w) = self.resize_image_type0(img)
        img = self.normalize(img)
        img = img[np.newaxis, :]
        tensor = torch.from_numpy(img).float()

        tensor = tensor.to("cuda:0")
        print(tensor.shape)
        batch = dict()
        batch['image'] = tensor

        # ===> model predict
        pred = model.forward(batch, training=False)
        return pred


if __name__ == "__main__":
    config = "/share_sdb/clj/DB2/DB/experiments/seg_detector/ic15_resnet_vd_18.yaml"
    conf = Config()
    experiment_args = conf.compile(conf.load(config))['Experiment']

    image_path = "/share_sdb/clj/paddle/img_10.jpg"

    args = {}
    args["model_path"] = ""
    args["image_short_side"] = ""
    args["result_dir"] = "/share_sdb/clj/DB/demo_results"
    args["polygon"] = False
    args["paddle_model_path"] = "/share_sdb/clj/paddle/models/ch_ppocr_server_v1.1_det_train/best_accuracy"

    experiment_args.update(cmd=args)
    experiment = Configurable.construct_class_from_config(experiment_args)

    pred = Demo(experiment, args).inference(image_path)
    print(pred)
    print(pred.shape)


