import torch
import cv2
import os
import time
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.torch_utils import select_device
from PIL import Image,ImageOps
from torchvision.transforms import transforms
from torchvision import models
import torch.nn as nn
import torch


device_obj = select_device('')
half = device_obj.type != 'cpu'

def return_model(obj_path = "pth\\best_adj.pt", cls_res_path = None,cls_next_path=None):
    '''
    input: 物件偵測模型參數位址, 分類模型參數位址
    return: 物件偵測模型, 分類模型
    resnet : D:\\Python\\TB_Rapid\\Double_color\\cls_after_obj\\best_resnet18_3.pth
    resnext : D:\\Python\\TB_Rapid\\Double_color\\cls_after_obj\\best_resNext50_3.pth
    '''
    weights, imgsz = \
    obj_path, 576
    device_obj = select_device('')
    half = device_obj.type != 'cpu'
    model_obj = attempt_load(weights, map_location=device_obj)
    model_obj.to(device_obj).eval()
    if half:
        model_obj.half()  # to FP16

    device_cls = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #model_cls = models.resnext50_32x4d(pretrained=False) #next
    #model_cls.fc = nn.Linear(2048, 2)
    if cls_res_path is not None:
        model_cls = models.resnet18(pretrained=False)
        model_cls.fc = nn.Sequential(nn.Linear(512, 1),nn.Sigmoid())

        model_cls.to(device_cls)
        model_cls.load_state_dict(torch.load(cls_res_path))
        model_cls.eval()
    elif cls_next_path is not None:
        model_cls = models.resnext50_32x4d(pretrained=False)
        model_cls.fc = nn.Sequential(nn.Linear(2048, 1),nn.Sigmoid())

        model_cls.to(device_cls)
        model_cls.load_state_dict(torch.load(cls_next_path))
        model_cls.eval()

    return model_obj, model_cls

def letterbox(img, new_shape=(576, 576), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

#original one (1.8 s)
def object_detection(imgs, score_threshold = 0.4,model=None):
    '''
    imgs:放入預偵測的圖片
    score_threshold = 0.4
    model (default is None):放入物件偵測模型
    '''
    top, down = 0, 576
    rag = 528
    N = 0
    reshape_ratio = 1
    predict_bbox = []
    #predict_bbox_dict = {}
    for h in range(4):
        left, right = 0, 576
        for w in range(4):
            img_crop=imgs[top:down,left:right]
            #Padded resize
            #img = letterbox(img_crop, new_shape=576)[0]#如果沒有調整大小的問題，可以先不加這個
            #Convert
            img = img_crop[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device_obj)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            model.eval()
            pred = model(img, augment='store_true')[0]
            pred_nms = non_max_suppression(pred, score_threshold, 0.5, agnostic='store_true')#0.4表示要多少信心度以上才選
            if pred_nms != [None]:
                for i in range(len(pred_nms[0])):
                    bb_left = int(left+pred_nms[0][i][0]*reshape_ratio)
                    bb_top = int(top+pred_nms[0][i][1]*reshape_ratio)
                    bb_right = int(left+pred_nms[0][i][2]*reshape_ratio)
                    bb_down = int(top+pred_nms[0][i][3]*reshape_ratio)
                    if i ==0:
                        predict_bbox.append([bb_left, bb_top, bb_right, bb_down, float(pred_nms[0][i][4]),((bb_left+bb_right)/2,(bb_top+bb_down)/2)])
                    elif  (((bb_left+bb_right)/2-predict_bbox[i-1][5][0])**2+((bb_top+bb_down)/2-predict_bbox[i-1][5][1])**2)**(1/2) >5:
                        predict_bbox.append([bb_left, bb_top, bb_right, bb_down, float(pred_nms[0][i][4]),((bb_left+bb_right)/2,(bb_top+bb_down)/2)])
                    else:pass
            else: pass

            left += rag
            right += rag
        top += rag
        down += rag
    return predict_bbox



#faster one (0.8 s)
def object_detection_faster(imgs, score_threshold = 0.4,model=None):
    '''
    imgs:放入預偵測的圖片
    score_threshold = 0.4
    model (default is None):放入物件偵測模型
    '''
    top, down = 0, 576
    rag = 528
    N = 0
    reshape_ratio = 1
    predict_bbox = []
    img_combine = None
    loc_combine = [[0, 0], [528, 0], [1056, 0], [1584, 0], [0, 528], [528, 528], [1056, 528], [1584, 528], [0, 1056], [528, 1056], [1056, 1056], [1584, 1056], [0, 1584], [528, 1584], [1056, 1584], [1584, 1584]]
    #predict_bbox_dict = {}
    for h in range(4):
        left, right = 0, 576
        for w in range(4):
            img_crop=imgs[top:down,left:right]
            #Convert
            img = img_crop[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img)
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            #combine img
            if img_combine != None:
                img_combine = torch.cat([img_combine, img],0)
            else:
                img_combine = img

            #combine loc
            # loc_combine.append([left, top])

            left += rag
            right += rag
        top += rag
        down += rag
    img_combine = img_combine.to(device_obj)
    img_combine = img_combine.half() if half else img_combine.float()  # uint8 to fp16/32
    img_combine /= 255.0  # 0 - 255 to 0.0 - 1.0
    model.eval()
    pred = model(img_combine, augment='store_true')[0]
    pred_nms = non_max_suppression(pred, score_threshold, 0.5, agnostic='store_true')#0.4表示要多少信心度以上才選
    for i, result in enumerate(pred_nms):
        if result != None:
            left,top = loc_combine[i]
            for j in range(len(result)):
                bb_left = int(left+result[j][0])
                bb_top = int(top+result[j][1])
                bb_right = int(left+result[j][2])
                bb_down = int(top+result[j][3])
                predict_bbox.append([bb_left, bb_top, bb_right, bb_down, float(result[j][4]),((bb_left+bb_right)/2,(bb_top+bb_down)/2)])
    output=[]
    for i,info in enumerate(predict_bbox):
        if i ==0:
            output.append(info)
        elif ((info[5][0]-predict_bbox[i-1][5][0])**2+(info[5][1]-predict_bbox[i-1][5][1])**2)**(1/2)>5:
            output.append(info)
    return output


# if i ==0:
#     predict_bbox.append([bb_left, bb_top, bb_right, bb_down, float(pred_nms[0][i][4]),((bb_left+bb_right)/2,(bb_top+bb_down)/2)])
# elif  (((bb_left+bb_right)/2-predict_bbox[i-1][5][0])**2+((bb_top+bb_down)/2-predict_bbox[i-1][5][1])**2)**(1/2) >5:
#     predict_bbox.append([bb_left, bb_top, bb_right, bb_down, float(pred_nms[0][i][4]),((bb_left+bb_right)/2,(bb_top+bb_down)/2)])
# else:pass


def object_detection_slower(imgs, score_threshold = 0.4,model=None):
    '''
    imgs:放入預偵測的圖片
    score_threshold = 0.4
    model (default is None):放入物件偵測模型
    '''
    device_obj = select_device('')
    half = device_obj.type != 'cpu'
    top, down = 0, 576
    rag = 528
    N = 0
    reshape_ratio = 1
    predict_bbox = []
    img_combine = None
    loc_combine = [[0, 0], [528, 0], [1056, 0], [1584, 0], [0, 528], [528, 528], [1056, 528], [1584, 528], [0, 1056], [528, 1056], [1056, 1056], [1584, 1056], [0, 1584], [528, 1584], [1056, 1584], [1584, 1584]]
    #predict_bbox_dict = {}
    for h in range(4):
        left, right = 0, 576
        for w in range(4):
            img_crop=imgs[top:down,left:right]
            #Convert
            img = img_crop[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img)
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            #combine img
            if img_combine != None:
                img_combine = torch.cat([img_combine, img],0)
            else:
                img_combine = img

            #combine loc
            # loc_combine.append([left, top])

            left += rag
            right += rag
        top += rag
        down += rag
    img_combine = img_combine.to(device_obj)
    img_combine = img_combine.half() if half else img_combine.float()  # uint8 to fp16/32
    img_combine /= 255.0  # 0 - 255 to 0.0 - 1.0
    model.eval()
    pred = model(img_combine, augment='store_true')[0]
    pred_nms = non_max_suppression(pred, score_threshold, 0.5, agnostic='store_true')#0.4表示要多少信心度以上才選
    for i, result in enumerate(pred_nms):
        if result != None:
            left,top = loc_combine[i]
            for j in range(len(result)):
                bb_left = int(left+result[j][0])
                bb_top = int(top+result[j][1])
                bb_right = int(left+result[j][2])
                bb_down = int(top+result[j][3])
                predict_bbox.append([bb_left, bb_top, bb_right, bb_down, float(result[j][4]),((bb_left+bb_right)/2,(bb_top+bb_down)/2)])
    return predict_bbox


def adjust_xy(x,y,range=32):
    top = y-range
    down = y+range
    left = x-range
    right = x+range
    if x-range<0:
        left = 0
    elif x+range>2159:
        right = 2159
    elif y-range<0:
        top = 0
    elif y+range>2159:
        down = 2159
    else:pass
    return int(top), int(down), int(left), int(right)


# def classify(output,img_path,device="cpu",model_cls=None,nor_mean=[0.1086, 0.0688, 0.0000],nor_std=[0.2120, 0.1158, 1]):
#     if output != []:
#         transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(nor_mean,nor_std)])
#         img = Image.open(img_path)
#         img_crop_combine = None
#         for i in range(len(output)):
#             x,y = output[i][5]
#             top, down, left, right = adjust_xy(x,y,range=32)
#             img_crop = img.crop((left,top,right,down))
#             img_crop = ImageOps.expand(img_crop,(int((64-(right-left))/2),int((64-(down-top))/2),64-(right-left)-int((64-(right-left))/2),64-(down-top)-int((64-(down-top))/2)),0)
#             img_crop = transform(img_crop)
#             img_crop = img_crop.unsqueeze(0)
#             if img_crop_combine != None:
#                 img_crop_combine = torch.cat([img_crop_combine, img_crop],0)
#             else:
#                 img_crop_combine = img_crop
#         img_crop_combine = img_crop_combine.to(device)
#         output = model_cls(img_crop_combine).max(dim = 1)[1]
#     else:
#         output = [0]
#     return output

def classify(output,img_path,device="cpu",model_cls=None,nor_mean=[0.1037, 0.0704, 0.0000],nor_std=[0.2372, 0.1422, 1]):
    if output != []:
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(nor_mean,nor_std)])
        img = Image.open(img_path)
        img_crop_combine = None
        for i in range(len(output)):
            x,y = output[i][5]
            top, down, left, right = adjust_xy(x,y,range=32)
            img_crop = img.crop((left,top,right,down))
            img_crop = ImageOps.expand(img_crop,(int((64-(right-left))/2),int((64-(down-top))/2),64-(right-left)-int((64-(right-left))/2),64-(down-top)-int((64-(down-top))/2)),0)
            img_crop = transform(img_crop)
            img_crop = img_crop.unsqueeze(0)
            if img_crop_combine != None:
                img_crop_combine = torch.cat([img_crop_combine, img_crop],0)
            else:
                img_crop_combine = img_crop
        img_crop_combine = img_crop_combine.to(device)
        output = model_cls(img_crop_combine).cpu()
        output.squeeze_(0)
        output_cls = cls(output)
        final_output = [list(output_cls.detach().numpy()),list(output.detach().numpy())]
    else:
        final_output = []
    return final_output

def cls(y_pred,threshold=0.5):
  y_pred = torch.where(y_pred>threshold, 1, 0)
  return y_pred

