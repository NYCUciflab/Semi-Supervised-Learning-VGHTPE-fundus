#Load Dataset for Semi-Supervised Learning 

import os
from torch.utils.data import Dataset
from torchvision import transforms
from randaugment import RandAugmentMC
from PIL import Image

class LoadTestData(Dataset):

    def __init__(self, root_dir, color_mode=None, preprocessing=None, mode=None):
        """
        Args:
            root_dir(str): Path to target image folder.
                Classification mode should contain atleast one subdirectory per class.
                Still working on segmentation and object detection mode.
            color_mode(str):
                Color mode of input image.
                RGB, RGBA, BGR,etc.
            preprocessing(callable, optional): Preprocessing that would be applied on image
        """
        self.root_dir = root_dir
        self.preprocessing = preprocessing
        self.color_mode = color_mode
        #self.mode = mode

        self.class_names = [f for f in os.listdir(root_dir) if not f.startswith('.')]
        self.img_names = []
        self.labels = []

        for i in range(len(self.class_names)):
            class_dir = os.path.join(self.root_dir, self.class_names[i])
            img_list = os.listdir(class_dir)
            img_list = [class_dir
                + os.sep
                + f for f in img_list if not f.startswith('.')]
            self.img_names += img_list
            self.labels += [i for _ in range(len(img_list))]

        print(len(self.img_names),
             "images belong to",
             len(self.class_names),
             "class")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(self.img_names[idx])
        img_names = self.img_names
        if self.color_mode is not None:
            img = img.convert(self.color_mode)

        if self.preprocessing is not None:
            img = self.preprocessing(img)

        label = self.labels[idx]

        return img, label
    
class LoadLabeledData(Dataset):

    def __init__(self, root_dir, color_mode):
        """
        Args:
            root_dir(str): Path to target image folder.
                Should contain at least one subdiretory folder
            color_mode(str):
                Color mode of input image.
                RGB, RGBA, BGR, etc.

        """
        self.root_dir = root_dir
        self.color_mode = color_mode


        self.class_names = [f for f in os.listdir(root_dir) if not f.startswith('.')]
        self.img_names = []
        self.labels = []

        for i in range(len(self.class_names)):
            class_dir = os.path.join(self.root_dir, self.class_names[i])
            img_list = sorted(os.listdir(class_dir))
            img_list = [class_dir
                + os.sep
                + f for f in img_list if not f.startswith('.')]
            self.img_names += img_list
            self.labels += [i for _ in range(len(img_list))]

        print(len(self.img_names),
             "images belong to",
             len(self.class_names),
             "class")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        img = Image.open(self.img_names[idx])
        img_names = self.img_names[idx]
        if self.color_mode is not None:
            img = img.convert(self.color_mode)
        preprocessing = transforms.Compose([
        transforms.Resize((224, 224)),
        RandAugmentMC(n=2, m=2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_of_data, std=std_of_data)
        ])
        n_processing = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_of_data, std=std_of_data)
        ])
        rand_seed = random.randint(1, 10)
        if rand_seed >= 5:
            img = preprocessing(img)
        else:
            img = n_processing(img)
        label = self.labels[idx]
        label = np.array(label)


        return img, label


class LoadUnlabeledData(Dataset):
    def __init__(self, root_dir, color_mode, mode):
        """
        Args:
            root_dir(str): Path to target image folder.
                Should contain at least one subdiretory folder
            color_mode(str):
                Color mode of input image.
                RGB, RGBA, BGR, etc.
            mode(callable, optional):
                semi-supervised method that will be applied on image.
        """
        self.root_dir = root_dir
        self.color_mode = color_mode


        self.img_src = [f for f in os.listdir(root_dir) if not f. startswith('.')]
        self.img_names = []
        self.labels=[]
        self.mode = mode

        for i in range(len(self.img_src)):
            img_dir = os.path.join(self.root_dir, self.img_src[i])
            img_list = sorted(os.listdir(img_dir))
            img_list = [img_dir
                       + os.sep
                       + f for f in img_list if not f.startswith('.')]
            self.img_names += img_list
            self.labels += [i for _ in range(len(img_list))]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):

        img = Image.open(self.img_names[idx])
        if self.color_mode is not None:
            img = img.convert(self.color_mode)
        weakaug = transforms.Compose([
            transforms.Resize((224, 224)),
            #transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=224,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            transforms.ToTensor()

        ])

        strongaug1 = transforms.Compose([
            transforms.Resize((224, 224)),
            #transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=224,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=5, m=10),
            transforms.ToTensor()
        ])

        w_img = weakaug(img)
        s1_img = strongaug1(img)
        label = self.labels[idx]

        if self.mode == "CoMatch":
            strongaug2 = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                #transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
            ])
            s2_img = strongaug2(img)
            return (w_img, s1_img, s2_img), label
        elif self.mode == "FixMatch":
            return (w_img, s1_img), label
