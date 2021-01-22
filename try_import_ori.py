from try_import_packages import *

model_obj, model_cls = return_model(cls_next_path="D:\\Python\\TB_Rapid\\Double_color\\cls_after_obj\\best_resNext50_3.pth")


neg_img_path = "D:\\Python\\TB_Rapid\\Double_color\\Dataset\\full\\adj_final\\test\\negative\\4_Overlay_A01_s62.tif"
img = cv2.imread(neg_img_path)
pos_img_path = "D:\\Python\\TB_Rapid\Double_color\\Dataset\\full\\adj_final\\test\\positive\\4017532_Overlay_A01_s152.tif"
img = cv2.imread(pos_img_path)


device_obj = select_device('')
half = device_obj.type != 'cpu'
device_cls = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output = object_detection_faster(img, score_threshold = 0.4,model=model_obj)
result = classify(output,pos_img_path,device=device_cls,model_cls=model_cls)

#刪除掉cls判斷不是的
new_output = []
for i,adj in enumerate(list(result)):
    if adj==1:
        new_output.append(output[i])




#統計對於test dataset的效果

#total image precision
pos_image_path = "D:\\Python\\TB_Rapid\\Double_color\\Dataset\\test\\positive"
pos_images = os.listdir(pos_image_path)
total = len(pos_images)
adj_pos = 0
adj_neg = 0
org_pos = 0
org_neg = 0
for img_n in pos_images:
    path = os.path.join(pos_image_path, img_n)
    img = cv2.imread(path)
    output = object_detection_faster(img, score_threshold = 0.4,model=model_obj)
    result = classify(output,path,device=device_cls,model_cls=model_cls)
    if output == []:
        org_neg+=1
    else:
        org_pos+=1

    if 1 in result:
        adj_pos+=1
    else:
        adj_neg+=1
        print(img_n)

print(adj_pos)#6
print(adj_neg)#1 ps 但這個原本就是標錯的
print(org_pos)#7
print(org_neg)#0 ps 但這個原本就是標錯的

neg_image_path = "D:\\Python\\TB_Rapid\\Double_color\\Dataset\\test\\negative"
neg_images = os.listdir(neg_image_path)
total = len(neg_images)
adj_pos = 0
adj_neg = 0
org_pos = 0
org_neg = 0
for img_n in neg_images:
    path = os.path.join(neg_image_path, img_n)
    img = cv2.imread(path)
    output = object_detection_faster(img, score_threshold = 0.4,model=model_obj)
    result = classify(output,path,device=device_cls,model_cls=model_cls)
    if output == []:
        org_neg+=1
    else:
        org_pos+=1

    if 1 in result:
        adj_pos+=1
    else:
        adj_neg+=1

print(adj_pos)#0
print(adj_neg)#13
print(org_pos)#7
print(org_neg)#6


#each object precision
import json
with open("D:\\Python\\TB_Rapid\\Double_color\\Dataset\\newlabel_adjust_loc.json","r") as f:
    loc = json.load(f)


ori_correct_list = []
ori_miss_list = []
ori_more_list = []
cls_correct_list = []
cls_miss_list = []
cls_more_list = []
for t in range(len(pos_images)):
    #loc[pos_images[t][:-4]]
    path = os.path.join(pos_image_path, pos_images[t])
    img = cv2.imread(path)
    output = object_detection_faster(img, score_threshold = 0.4,model=model_obj)
    result = classify(output,path,device=device_cls,model_cls=model_cls)
    new_output = []
    for i,adj in enumerate(list(result)):
        if adj==1:
            new_output.append(output[i])
    #new_output[0][5][0]

    answers_len = len(loc[pos_images[t][:-4]])
    tp=0
    for i in new_output:
        for j,obj_loc in enumerate(loc[pos_images[t][:-4]]):
            if i[5][0] > obj_loc[0] and i[5][0]<obj_loc[2] and i[5][1] >obj_loc[1] and i[5][1]<obj_loc[3]:#檢查
                tp+=1
    correct = tp
    miss = answers_len-correct
    more = len(new_output)-(answers_len-miss)
    cls_correct_list.append(correct)
    cls_miss_list.append(miss)
    cls_more_list.append(more)
    # print(correct)
    # print(miss)
    # print(more)

    tp=0
    for i in output:
        for j,obj_loc in enumerate(loc[pos_images[t][:-4]]):
            if i[5][0] > obj_loc[0] and i[5][0]<obj_loc[2] and i[5][1] >obj_loc[1] and i[5][1]<obj_loc[3]:#檢查
                tp+=1
    correct = tp
    miss = answers_len-correct
    more = len(output)-(answers_len-miss)
    ori_correct_list.append(correct)
    ori_miss_list.append(miss)
    ori_more_list.append(more)
    # print(correct)
    # print(miss)
    # print(more)

#resnext50
print(ori_correct_list)#[1, 0, 1, 2, 1, 1, 3]
print(ori_miss_list)   #[0, 1, 0, 0, 0, 0, 0]
print(ori_more_list)   #[2, 1, 1, 0, 1, 0, 0]
print(cls_correct_list)#[1, 0, 1, 2, 1, 1, 3]
print(cls_miss_list)   #[0, 1, 0, 0, 0, 0, 0]
print(cls_more_list)   #[0, 0, 0, 0, 0, 0, 0]

#resnet18
print(ori_correct_list)#[1, 0, 1, 2, 1, 1, 3]
print(ori_miss_list)#[0, 1, 0, 0, 0, 0, 0]
print(ori_more_list)#[2, 1, 1, 0, 1, 0, 0]
print(cls_correct_list)#[1, 0, 1, 2, 1, 1, 3]
print(cls_miss_list)#[0, 1, 0, 0, 0, 0, 0]
print(cls_more_list)#[0, 0, 0, 0, 0, 0, 0]



#只有obj(0.7)

pos_image_path = "D:\\Python\\TB_Rapid\\Double_color\\Dataset\\test\\positive"
pos_images = os.listdir(pos_image_path)
total = len(pos_images)
adj_pos = 0
adj_neg = 0
org_pos = 0
org_neg = 0
for img_n in pos_images:
    path = os.path.join(pos_image_path, img_n)
    img = cv2.imread(path)
    output = object_detection_faster(img, score_threshold = 0.7,model=model_obj)
    result = classify(output,path,device=device_cls,model_cls=model_cls)
    if output == []:
        org_neg+=1
    else:
        org_pos+=1

    if 1 in result:
        adj_pos+=1
    else:
        adj_neg+=1
        print(img_n)

# print(adj_pos)#6
# print(adj_neg)#1 ps 但這個原本就是標錯的
print(org_pos)#7
print(org_neg)#0 ps 但這個原本就是標錯的

neg_image_path = "D:\\Python\\TB_Rapid\\Double_color\\Dataset\\test\\negative"
neg_images = os.listdir(neg_image_path)
total = len(neg_images)
adj_pos = 0
adj_neg = 0
org_pos = 0
org_neg = 0
for img_n in neg_images:
    path = os.path.join(neg_image_path, img_n)
    img = cv2.imread(path)
    output = object_detection_faster(img, score_threshold = 0.7,model=model_obj)
    result = classify(output,path,device=device_cls,model_cls=model_cls)
    if output == []:
        org_neg+=1
    else:
        org_pos+=1

    if 1 in result:
        adj_pos+=1
    else:
        adj_neg+=1

# print(adj_pos)#0
# print(adj_neg)#13
print(org_pos)#4
print(org_neg)#9


ori_correct_list = []
ori_miss_list = []
ori_more_list = []
cls_correct_list = []
cls_miss_list = []
cls_more_list = []
for t in range(len(pos_images)):
    #loc[pos_images[t][:-4]]
    path = os.path.join(pos_image_path, pos_images[t])
    img = cv2.imread(path)
    output = object_detection_faster(img, score_threshold = 0.7,model=model_obj)
    result = classify(output,path,device=device_cls,model_cls=model_cls)
    new_output = []
    for i,adj in enumerate(list(result)):
        if adj==1:
            new_output.append(output[i])
    #new_output[0][5][0]

    answers_len = len(loc[pos_images[t][:-4]])
    tp=0
    for i in new_output:
        for j,obj_loc in enumerate(loc[pos_images[t][:-4]]):
            if i[5][0] > obj_loc[0] and i[5][0]<obj_loc[2] and i[5][1] >obj_loc[1] and i[5][1]<obj_loc[3]:#檢查
                tp+=1
    correct = tp
    miss = answers_len-correct
    more = len(new_output)-(answers_len-miss)
    cls_correct_list.append(correct)
    cls_miss_list.append(miss)
    cls_more_list.append(more)
    # print(correct)
    # print(miss)
    # print(more)

    tp=0
    for i in output:
        for j,obj_loc in enumerate(loc[pos_images[t][:-4]]):
            if i[5][0] > obj_loc[0] and i[5][0]<obj_loc[2] and i[5][1] >obj_loc[1] and i[5][1]<obj_loc[3]:#檢查
                tp+=1
    correct = tp
    miss = answers_len-correct
    more = len(output)-(answers_len-miss)
    ori_correct_list.append(correct)
    ori_miss_list.append(miss)
    ori_more_list.append(more)
    # print(correct)
    # print(miss)
    # print(more)

print(ori_correct_list)#[1, 0, 1, 2, 1, 1, 3]
print(ori_miss_list)#[0, 1, 0, 0, 0, 0, 0]
print(ori_more_list)#[2, 1, 1, 0, 1, 0, 0]
# print(cls_correct_list)#[1, 0, 1, 2, 1, 1, 3]
# print(cls_miss_list)#[0, 1, 0, 0, 0, 0, 0]
# print(cls_more_list)#[0, 0, 0, 0, 0, 0, 0]




#只有obj(0.9)

pos_image_path = "D:\\Python\\TB_Rapid\\Double_color\\Dataset\\test\\positive"
pos_images = os.listdir(pos_image_path)
total = len(pos_images)
adj_pos = 0
adj_neg = 0
org_pos = 0
org_neg = 0
for img_n in pos_images:
    path = os.path.join(pos_image_path, img_n)
    img = cv2.imread(path)
    output = object_detection_faster(img, score_threshold = 0.9,model=model_obj)
    result = classify(output,path,device=device_cls,model_cls=model_cls)
    if output == []:
        org_neg+=1
    else:
        org_pos+=1

    if 1 in result:
        adj_pos+=1
    else:
        adj_neg+=1
        print(img_n)

# print(adj_pos)#6
# print(adj_neg)#1 ps 但這個原本就是標錯的
print(org_pos)#3
print(org_neg)#4 ps 但這個原本就是標錯的

neg_image_path = "D:\\Python\\TB_Rapid\\Double_color\\Dataset\\test\\negative"
neg_images = os.listdir(neg_image_path)
total = len(neg_images)
adj_pos = 0
adj_neg = 0
org_pos = 0
org_neg = 0
for img_n in neg_images:
    path = os.path.join(neg_image_path, img_n)
    img = cv2.imread(path)
    output = object_detection_faster(img, score_threshold = 0.9,model=model_obj)
    result = classify(output,path,device=device_cls,model_cls=model_cls)
    if output == []:
        org_neg+=1
    else:
        org_pos+=1

    if 1 in result:
        adj_pos+=1
    else:
        adj_neg+=1

# print(adj_pos)#0
# print(adj_neg)#13
print(org_pos)#1
print(org_neg)#12


ori_correct_list = []
ori_miss_list = []
ori_more_list = []
cls_correct_list = []
cls_miss_list = []
cls_more_list = []
for t in range(len(pos_images)):
    #loc[pos_images[t][:-4]]
    path = os.path.join(pos_image_path, pos_images[t])
    img = cv2.imread(path)
    output = object_detection_faster(img, score_threshold = 0.9,model=model_obj)
    result = classify(output,path,device=device_cls,model_cls=model_cls)
    new_output = []
    for i,adj in enumerate(list(result)):
        if adj==1:
            new_output.append(output[i])
    #new_output[0][5][0]

    answers_len = len(loc[pos_images[t][:-4]])
    tp=0
    for i in new_output:
        for j,obj_loc in enumerate(loc[pos_images[t][:-4]]):
            if i[5][0] > obj_loc[0] and i[5][0]<obj_loc[2] and i[5][1] >obj_loc[1] and i[5][1]<obj_loc[3]:#檢查
                tp+=1
    correct = tp
    miss = answers_len-correct
    more = len(new_output)-(answers_len-miss)
    cls_correct_list.append(correct)
    cls_miss_list.append(miss)
    cls_more_list.append(more)
    # print(correct)
    # print(miss)
    # print(more)

    tp=0
    for i in output:
        for j,obj_loc in enumerate(loc[pos_images[t][:-4]]):
            if i[5][0] > obj_loc[0] and i[5][0]<obj_loc[2] and i[5][1] >obj_loc[1] and i[5][1]<obj_loc[3]:#檢查
                tp+=1
    correct = tp
    miss = answers_len-correct
    more = len(output)-(answers_len-miss)
    ori_correct_list.append(correct)
    ori_miss_list.append(miss)
    ori_more_list.append(more)
    # print(correct)
    # print(miss)
    # print(more)

print(ori_correct_list)#[0, 0, 1, 2, 1, 0, 0]
print(ori_miss_list)#[1, 1, 0, 0, 0, 1, 3]
print(ori_more_list)#[0, 0, 0, 0, 0, 0, 0]
# print(cls_correct_list)#[1, 0, 1, 2, 1, 1, 3]
# print(cls_miss_list)#[0, 1, 0, 0, 0, 0, 0]
# print(cls_more_list)#[0, 0, 0, 0, 0, 0, 0]