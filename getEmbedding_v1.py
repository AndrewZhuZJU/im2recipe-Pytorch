#############################################
# author: Andrew
# date: June 6, 2018
# get recipe embeddings using im2recipe model
#############################################

import  torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from trijoint_v1 import im2recipe
import pickle
from PIL import Image
import lmdb

#single <recipe, image> pair
#imgs are in the base_path directory, value of partition{train or test or val} 
def getSingleEmbedding(base_path, rec_id, img_id, partition):
    model = im2recipe()
    model.visionMLP = torch.nn.DataParallel(model.visionMLP, device_ids=[0,1])
    print('loading pretrained model....')
    checkpoint = torch.load('model_e220_v-4.700.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    print('loading model done.')

    # get input image
    im = Image.open(base_path + img_id)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    im_trans = transforms.Compose([
                transforms.Scale(256), # rescale the image keeping the original aspect ratio
                transforms.CenterCrop(224), # we get only the center of that rescaled
                transforms.ToTensor(),
                normalize,
            ])

    img = im_trans(im)

    # get input instruction and ingredient
    maxInst = 20
    env_val = lmdb.open('data/' + partition + '/' + partition + '_lmdb/')
    txn_val = env_val.begin(write = False)
    serialized_sample = txn_val.get(rec_id)
    sample = pickle.loads(serialized_sample)

    #instrctions
    instrs = sample['intrs']
    itr_ln = len(instrs)
    t_inst = np.zeros((maxInst, np.shape(instrs)[1]), dtype=np.float32)
    t_inst[:itr_ln][:] = instrs
    instrs = torch.FloatTensor(t_inst)

    # ingredients
    ingrs = sample['ingrs'].astype(int)
    ingrs = torch.LongTensor(ingrs)
    igr_ln = max(np.nonzero(sample['ingrs'])[0]) + 1

    output = model(img, instrs, torch.IntTensor([itr_ln]), ingrs, torch.IntTensor([igr_ln]))

    #print(output.detach().tolist())
    #print(type(output.detach().tolist()))
    np.save('recipe_emb.npy', output.detach().numpy())


#multi <recipe, image> pairs
#rec_img file contains information of <rec_id, img_id, partition>
# e.g. <459a173d82 c10e32af4f.jpg train>
def getEmbeddings(base_path, rec_img):
    rec_imgs = []
    with open(base_path + rec_img, 'r') as f:
        while(True):
            pair = f.readline()
            if(not pair):
                break
            pair = pair.strip().split(' ')
            rec_imgs.append(pair)
    

    #recipe_embs
    recipe_embs = []

    model = im2recipe()
    model.visionMLP = torch.nn.DataParallel(model.visionMLP, device_ids=[0,1])
    print('loading pretrained model....')
    checkpoint = torch.load('model_e220_v-4.700.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    print('loading model done.')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    im_trans = transforms.Compose([
                transforms.Scale(256), # rescale the image keeping the original aspect ratio
                transforms.CenterCrop(224), # we get only the center of that rescaled
                transforms.ToTensor(),
                normalize,
            ])

    #open database
    env_val = lmdb.open('data/val/val_lmdb/')
    txn_val = env_val.begin(write = False)
    env_test = lmdb.open('data/test/test_lmdb/')
    txn_test = env_test.begin(write = False)
    env_train = lmdb.open('data/train/train_lmdb/')
    txn_train = env_train.begin(write = False)


    for pair in rec_imgs: 
        print(pair)
        # get input image
        im = Image.open(base_path + pair[1])
        img = im_trans(im)

        # get input instruction and ingredient
        maxInst = 20
        serialized_sample = 0
        if(pair[2] == 'train'):
            serialized_sample = txn_train.get(pair[0])
        elif(pair[2] == 'test'):
            serialized_sample = txn_test.get(pair[0])
        else:
            serialized_sample = txn_val.get(pair[0])
        
        if(not serialized_sample):
            print(pair[0] + ' not exists in database!')
            recipe_embs.append('None')
            continue
        sample = pickle.loads(serialized_sample)

        #instrctions
        instrs = sample['intrs']
        itr_ln = len(instrs)
        t_inst = np.zeros((maxInst, np.shape(instrs)[1]), dtype=np.float32)
        t_inst[:itr_ln][:] = instrs
        instrs = torch.FloatTensor(t_inst)

        # ingredients
        ingrs = sample['ingrs'].astype(int)
        ingrs = torch.LongTensor(ingrs)
        igr_ln = max(np.nonzero(sample['ingrs'])[0]) + 1

        output = model(img, instrs, torch.IntTensor([itr_ln]), ingrs, torch.IntTensor([igr_ln]))
        recipe_embs.append(output.detach().tolist())
    print(len(recipe_embs))
    np.save(base_path + 'recipe_embs.npy', recipe_embs)

#test
#getSingleEmbedding('results/dataset/15/', 'b64eddc1b2', '4d36296cdd.jpg', 'val')
getEmbeddings('results/dataset/3/', 'rec_img.txt')
