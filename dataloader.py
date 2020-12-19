from PIL import Image, ImageOps
from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import random









class Task(object):
      def __init__(self,all_classes,num_classes,num_instances,trainframe):
        self.all_classes=all_classes
        self.num_classes=num_classes
        self.num_instances=num_instances
        self.trainframe=trainframe
        self.train_roots=[]
        self.meta_roots=[]
        #self.train_labels=[]
        #self.meta_labels=[]
        samples_per_class=50
        sampled_classes=random.sample(all_classes,num_classes)
        label=0
        #labels=list(range(len(sampled_classes)))

        for c in sampled_classes:
            dframe=trainframe[trainframe["ID"]==c]
            dframe=dframe.sample(100)
            paths=dframe[["Image_path","Label_path"]]
            sample_idxs=np.random.choice(samples_per_class,samples_per_class,replace=False)
            train_idxs=sample_idxs[:num_instances]
            meta_idxs=sample_idxs[num_instances:(num_instances*2)]
            for idx in train_idxs:
                self.train_roots.append((paths.iloc[idx][0],paths.iloc[idx][1]))
                #self.train_labels.append(paths.iloc[idx][1])
            for idx in meta_idxs:
                self.meta_roots.append((paths.iloc[idx][0],paths.iloc[idx][1]))
                #self.meta_labels.append(paths.iloc[idx][1])
            label+=1


class TestTask(object):
      def __init__(self,all_classes,num_classes,num_instances,num_test_instances,testframe):
        self.all_classes=all_classes
        self.num_classes=num_classes
        self.num_instances=num_instances
        self.testframe=testframe
        self.num_test_instances=num_test_instances
        self.test_roots=[]
        self.train_roots=[]

        samples_per_class=195
        sampled_classes= random.sample(all_classes,num_classes)
        label=0
        #labels=list(range(len(sampled_classes)))

        for c in sampled_classes:
            cframe=testframe[testframe["ID"]==c].sample(195)
            cframe.reset_index(inplace=True,drop=True)
            paths=cframe[["Image_path","Label_path"]]
            sample_idxs=np.random.choice(samples_per_class,samples_per_class,replace=False)
            train_idxs=sample_idxs[:num_instances]
            test_idxs=sample_idxs[num_instances:(num_instances+num_test_instances)]
            for idx in test_idxs:
              self.test_roots.append((paths.iloc[idx][0],paths.iloc[idx][1]))
              #self.test_labels.append(paths.iloc[idx][1])
            # for idx in meta_idxs:
            #   self.meta_roots.append(paths[idx])
            #   self.meta_labels.append(label)
            for idx in train_idxs:
              self.train_roots.append((paths.iloc[idx][0],paths.iloc[idx][1]))
              #self.train_labels.append(paths.iloc[idx][1])
            label+=1


def handle_task(task,transform):
    img_list, msk_list=[],[]
    for each_path in task:
        image=np.array(Image.open(each_path[0]))
        mask=np.array(Image.open(each_path[1]).convert('L'))
        augmented=transform(image=image,mask=mask)
        image=augmented["image"]
        mask=augmented["mask"]
        img_list.append(image)
        msk_list.append(mask)

    img_tensor=torch.stack(img_list)
    msk_tensor=torch.stack(msk_list)
    img_tensor=torch.reshape(img_tensor,(len(task),3,image.shape[1],image.shape[2]))
    msk_tensor=torch.reshape(msk_tensor,(len(task),1,mask.shape[1],mask.shape[2]))
    return img_tensor,msk_tensor



class MiniSet(Dataset):
  def __init__(self,fileroots,transform):
    self.tasks=fileroots
    self.transform=transform

  def __len__(self):
    return len(self.tasks)

  def __getitem__(self,idx):
        task=self.tasks[idx]
        img_tens,msk_tens= handle_task(task,self.transform)

        return [img_tens,msk_tens]
