from crowdposetools.coco import COCO

test_filename = '/home/rookie/cwt/DEKR/data/crowdpose/json/crowdpose_test.json'
test_coco = COCO(test_filename)
test_ids = list(test_coco.imgs.keys())
print(len(test_ids))
print(test_ids[0])
trainval_filename = '/home/rookie/cwt/DEKR/data/crowdpose/json/crowdpose_trainval.json'
trainval_coco = COCO(trainval_filename)
trainval_ids = list(trainval_coco.imgs.keys())
print(len(trainval_ids))
for test_id in test_ids:
    if test_id in trainval_ids:
        print(test_id)