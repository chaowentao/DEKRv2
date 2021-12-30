from pycocotools.coco import COCO
import json

# annFile = '/home/rookie/cwt/DEKR/data/minicoco/annotations/person_keypoints_train2017_.json'
# tar_img_dir = '/home/rookie/cwt/DEKR/data/minicoco/images/train2017'
# tar_json_path = './person_keypoints_train2017.json'
# coco = COCO(annFile)
# # coco.split_json(tarDir=tar_img_dir, tarFile=tar_json_path)
# ids = list(coco.imgs.keys())
# print(len(ids))

val_dataset = json.load(
    open("/home/rookie/cwt/DEKR_gc/data/crowdpose/json/crowdpose_test.json"))
print(len(val_dataset['images']))
print(val_dataset['images'][2].keys())
canditate_filenames = []
for i in range(len(val_dataset['images'])):
    canditate_filenames.append
