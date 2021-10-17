import os
from PIL import Image, ImageOps
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance
import cv2
from skimage import segmentation, color

num_fs = 12
zero_pad_pth = os.getcwd() + '/LFSOD_dataset/black.jpg'

#several data augumentation strategies
def cv_random_flip(img, label,depth, fss):
    flip_flag = random.randint(0, 1)
    # flip_flag2= random.randint(0,1)
    #left right flip
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        for idx in range(num_fs):
            fss[idx] = fss[idx].transpose(Image.FLIP_LEFT_RIGHT)
    #top bottom flip
    # if flip_flag2==1:
    #     img = img.transpose(Image.FLIP_TOP_BOTTOM)
    #     label = label.transpose(Image.FLIP_TOP_BOTTOM)
    #     depth = depth.transpose(Image.FLIP_TOP_BOTTOM)
    return img, label, depth, fss
def randomCrop(image, label,depth, fss):
    border=30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width-border , image_width)
    crop_win_height = np.random.randint(image_height-border , image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    for idx in range(num_fs):
        fss[idx] = fss[idx].crop(random_region)
    return image.crop(random_region), label.crop(random_region), depth.crop(random_region), fss
def randomRotation(image, label, depth, fss):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        depth = depth.rotate(random_angle, mode)
        for idx in range(num_fs):
            fss[idx] = fss[idx].rotate(random_angle, mode)
    return image, label, depth, fss
def colorEnhance(image):
    bright_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity=random.randint(0,20)/10.0
    image=ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity=random.randint(0,30)/10.0
    image=ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image
def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im
    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))
def randomPeper(img):
    img=np.array(img)
    noiseNum=int(0.0015*img.shape[0]*img.shape[1])
    for i in range(noiseNum):
        randX=random.randint(0,img.shape[0]-1)
        randY=random.randint(0,img.shape[1]-1)
        if random.randint(0,1)==0:
            img[randX,randY]=0
        else:
            img[randX,randY]=255 
    return Image.fromarray(img)  

# dataset for training
class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, depth_root, fs_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
                    or f.endswith('.png')]
        self.fss = [fs_root + f for f in os.listdir(fs_root) if f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        self.fss = sorted(self.fss)
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.depths_transform = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)),transforms.ToTensor()])
        self.fs_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        if gt.size != image.size:
            gt = gt.resize(image.size)
        depth = self.binary_loader(self.depths[index])
        depth = depth.resize(image.size)
        fss = self.fs_loader(self.images[index], self.fss, image.size)
        image, gt, depth, fss = cv_random_flip(image, gt, depth, fss)
        image, gt, depth, fss = randomCrop(image, gt, depth, fss)
        image, gt, depth, fss = randomRotation(image, gt, depth, fss)
        image = colorEnhance(image)
        gt = randomPeper(gt)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        depth = self.depths_transform(depth)
        for idx in range(num_fs):
            fss[idx] = self.fs_transform(fss[idx])
        
        return image, gt, depth, fss

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def fs_loader(self, item, fs_list, img_size):
        img_name = item.split('/')[-1]
        fs_pth = item[:-(len(img_name)+5)] + '/FS_rgb/'
        m_list = []
        for fs in fs_list:
            fs_name = fs[len(fs_pth):]
            temp = fs_name.split('_')
            m_list.append(temp[0])
        idxs = [i for i, x in enumerate(m_list) if x == img_name[:-4]]
        trg_fs_list = []
        for idx in range(len(idxs)):
            trg_fs_list.append(fs_list[idxs[idx]])
        #temp_count_1 = int(num_fs / len(trg_fs_list))
        #temp_count_2 = num_fs % len(trg_fs_list)
        #new_trg_fs_list = temp_count_1 * trg_fs_list + trg_fs_list[:temp_count_2]
        #new_trg_fs_list = sorted(new_trg_fs_list)
        num_zero_pad = num_fs - len(trg_fs_list)
        if num_zero_pad == 0:
            new_trg_fs_list = trg_fs_list
        else:
            zero_fs_list = []
            for idx in range(num_zero_pad):
                zero_fs_list.append(zero_pad_pth)
            new_trg_fs_list = trg_fs_list + zero_fs_list

        imgs = []
        for idx in range(num_fs):
            img_pth = new_trg_fs_list[idx]
            with open(img_pth, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
                img = img.resize(img_size)
            imgs.append(img)

        return imgs

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt, depth):
        assert img.size == gt.size and gt.size==depth.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST),depth.resize((w, h), Image.NEAREST)
        else:
            return img, gt, depth

    def __len__(self):
        return self.size

#dataloader for training
def get_loader(image_root, gt_root, depth_root, fs_root, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=True):

    dataset = SalObjDataset(image_root, gt_root, depth_root, fs_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

#test dataset and loader
class test_dataset:
    def __init__(self, image_root, gt_root, depth_root, fs_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.depths=[depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
                    or f.endswith('.png')]
        self.fss = [fs_root + f for f in os.listdir(fs_root) if f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        self.fss = sorted(self.fss)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.fs_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.depths_transform = transforms.Compose([transforms.Resize((self.testsize, self.testsize)),transforms.ToTensor()])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        depth = self.binary_loader(self.depths[self.index])
        depth = self.depths_transform(depth).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        image_for_post = self.rgb_loader(self.images[self.index])
        image_for_post = image_for_post.resize(gt.size)
        fss = self.fs_loader(self.images[self.index], self.fss)
        for idx in range(num_fs):
            fss[idx] = self.fs_transform(fss[idx]).unsqueeze(0)
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return image, gt, depth, fss, name, np.array(image_for_post)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def fs_loader(self, item, fs_list):
        img_name = item.split('/')[-1]
        fs_pth = item[:-(len(img_name) + 5)] + '/FS_rgb/'
        m_list = []
        for fs in fs_list:
            fs_name = fs[len(fs_pth):]
            temp = fs_name.split('_')
            m_list.append(temp[0] + '_' + temp[1])
        idxs = [i for i, x in enumerate(m_list) if x == img_name[:-4]]
        trg_fs_list = []
        for idx in range(len(idxs)):
            trg_fs_list.append(fs_list[idxs[idx]])
        # temp_count_1 = int(num_fs / len(trg_fs_list))
        # temp_count_2 = num_fs % len(trg_fs_list)
        # new_trg_fs_list = temp_count_1 * trg_fs_list + trg_fs_list[:temp_count_2]
        # new_trg_fs_list = sorted(new_trg_fs_list)
        num_zero_pad = num_fs - len(trg_fs_list)
        if num_zero_pad == 0:
                new_trg_fs_list = trg_fs_list
        else:
                zero_fs_list = []
                for idx in range(num_zero_pad):
                    zero_fs_list.append(zero_pad_pth)
                new_trg_fs_list = trg_fs_list + zero_fs_list

        imgs = []
        for idx in range(num_fs):
            img_pth = new_trg_fs_list[idx]
            with open(img_pth, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
            imgs.append(img)

        return imgs

    def __len__(self):
        return self.size

#test dataset and loader (with FS during testing)
class test_dataset_2:
    def __init__(self, image_root, gt_root, depth_root, fs_root, testsize, set_name):
        self.dataset_name = set_name
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
                    or f.endswith('.png')]
        self.fss = [fs_root + f for f in os.listdir(fs_root) if f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        self.fss = sorted(self.fss)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        # self.gt_transform = transforms.Compose([
        #     transforms.Resize((self.trainsize, self.trainsize)),
        #     transforms.ToTensor()])
        self.depths_transform = transforms.Compose([transforms.Resize((self.testsize, self.testsize)),transforms.ToTensor()])
        self.size = len(self.images)
        self.index = 0

    def load_data(self, set_name):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        depth = self.binary_loader(self.depths[self.index])
        depth = self.depths_transform(depth).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        image_for_post = self.rgb_loader(self.images[self.index])
        image_for_post = image_for_post.resize(gt.size)
        fss = self.fs_loader(self.images[self.index], self.fss)
        for idx in range(num_fs):
            fss[idx] = self.transform(fss[idx]).unsqueeze(0)
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return image, gt, depth, fss, name, np.array(image_for_post)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def fs_loader(self, item, fs_list):
        img_name = item.split('/')[-1]
        fs_pth = item[:-(len(img_name) + 5)] + '/FS_rgb/'
        m_list = []
        for fs in fs_list:
            fs_name = fs[len(fs_pth):]
            temp = fs_name.split('_')
            m_list.append(temp[0])
        idxs = [i for i, x in enumerate(m_list) if x == img_name[:-4]]
        trg_fs_list = []
        for idx in range(len(idxs)):
            trg_fs_list.append(fs_list[idxs[idx]])
        # temp_count_1 = int(num_fs / len(trg_fs_list))
        # temp_count_2 = num_fs % len(trg_fs_list)
        # new_trg_fs_list = temp_count_1 * trg_fs_list + trg_fs_list[:temp_count_2]
        # new_trg_fs_list = sorted(new_trg_fs_list)
        num_zero_pad = num_fs - len(trg_fs_list)
        if num_zero_pad == 0:
                new_trg_fs_list = trg_fs_list
        else:
                zero_fs_list = []
                for idx in range(num_zero_pad):
                    zero_fs_list.append(zero_pad_pth)
                new_trg_fs_list = trg_fs_list + zero_fs_list

        imgs = []
        for idx in range(num_fs):
            img_pth = new_trg_fs_list[idx]
            with open(img_pth, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
            imgs.append(img)

        return imgs

    def __len__(self):
        return self.size
