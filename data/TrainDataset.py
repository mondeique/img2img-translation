import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import util


class TrainDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def make_data_bundles(self, base_image_path):
        path_bundles = []
        for base_path in base_image_path:
            components = base_path.split('/')
            # components = [root,images,base,pXXX,cXXX,XXX]
            base_cloth_path = os.path.join(self.dir_clothes, components[-3])
            for color in os.listdir(base_cloth_path):
                if color[:-4] != components[-2]:
                    path_bundles.append({
                        'base_image' : base_path,
                        'base_image_mask' : os.path.join(self.root, 'images', 'mask', components[-3], components[-2], components[-1]),
                        'base_cloth' : os.path.join(self.dir_clothes, 'basic', components[-3], f'{components[-2]}.jpg'),
                        'base_cloth_mask' : os.path.join(self.dir_clothes, 'mask', components[-3], f'{components[-2]}.jpg'),
                        'input_cloth' : os.path.join(self.dir_clothes, 'basic', components[-3], f'{color}'),
                        'input_cloth_mask' : os.path.join(self.dir_clothes, 'mask', components[-3], f'{color}')
                    })

        return path_bundles

    def initialize(self, opt):
        self.opt = opt
        self.batch_size = opt.batch_size
        self.root = opt.dataroot
        self.batch_size = opt.batch_size
        self.dir_clothes = os.path.join(self.root, 'clothes')
        self.dir_images = os.path.join(self.root, 'images')
        self.dir_base_images = os.path.join(self.root, 'images/base')
        self.base_images_path = sorted(make_dataset(self.dir_base_images))

        self.train_data_bundle_paths = self.make_data_bundles(self.base_images_path)

        assert(opt.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):
        train_path = self.train_data_bundle_paths[index]

        base_image = Image.open(train_path('base_image')).convert('RGB')
        base_image_mask = Image.open(train_path('base_image_mask')).convert('L')
        base_cloth = Image.open(train_path('base_cloth')).convert('RGB')
        base_cloth_mask = Image.open(train_path('base_cloth_mask')).convert('L')
        input_cloth = Image.open(train_path('input_cloth')).convert('RGB')
        input_cloth_mask = Image.open(train_path('input_cloth_mask')).convert('L')

        image_list = [base_image, base_image_mask, base_cloth, base_cloth_mask, input_cloth, input_cloth_mask]
        resized_image_list = []
        for image in image_list:
            new_image = util.expand2square(image)
            new_image.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
            new_image = transforms.ToTensor()(new_image)
            resized_image_list.append(new_image)

        

        w, h = AB.size
        assert(self.opt.loadSize >= self.opt.fineSize)
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        B = AB.crop((w2, 0, w, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        A = transforms.ToTensor()(A)
        B = transforms.ToTensor()(B)
        w_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))

        A = A[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        B = B[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]

        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)

        if self.opt.direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        return {'A': A, 'B': B,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'
