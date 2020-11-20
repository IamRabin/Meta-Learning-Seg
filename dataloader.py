from PIL import Image, ImageOps
from torch.utils.data import Dataset
from random import random, randint


class Task(object):
      def __init__(self,all_classes,num_classes,num_instances):
        self.all_classes=all_classes
        self.num_classes=num_classes
        self.num_instances=num_instances
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
      def __init__(self,all_classes,num_classes,num_instances,num_test_instances):
        self.all_classes=all_classes
        self.num_classes=num_classes
        self.num_instances=num_instances
        self.num_test_instances=num_test_instances
        self.test_roots=[]
        self.train_roots=[]

        samples_per_class=20
        sampled_classes= random.sample(all_classes,num_classes)
        label=0
        #labels=list(range(len(sampled_classes)))

        for c in sampled_classes:
            cframe=testframe[testframe["ID"]==c].sample(100)
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

def handle_task(task):
    img_list, msk_list=[],[]
    for each_path in task:
        image=Image.open(each_path[0])
        mask=Image.open(each_path[1]).convert('L')
        image=transform(image)
        mask=transform(mask)
        image=np.array(image)
        mask=np.array(mask)
        img_list.append(image)
        msk_list.append(mask)

    img_arr=np.array(img_list)
    msk_arr=np.array(msk_list)
    img_arr=img_arr.reshape(len(task),3,image.shape[1],image.shape[2])
    msk_arr=msk_arr.reshape(len(task),1,mask.shape[1],mask.shape[2])
    return img_arr,msk_arr


class MiniSet(Dataset):
  def __init__(self,fileroots,labels,transform):
    self.fileroots=fileroots
    self.labels=labels
    self.transform=transform

  def __len__(self):
    return len(self.fileroots)

  def __getitem__(self,idx):
    img=Image.open(self.fileroots[idx])
    mask=Image.open(self.labels[idx]).convert('L')
    img=self.transform(img)
    mask=self.transform(mask)
    return [img,mask]
