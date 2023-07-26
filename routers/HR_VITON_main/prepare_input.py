import json
import time
from argparse import Namespace
from io import BytesIO

import cv2
from loguru import logger

from .cp_dataset_test import (CPDataLoader, CPDatasetTest, Image, np, osp,
                              torch, transforms)
from .test_generator import (ConditionGenerator, F, SPADEGenerator, get_opt,
                             load_checkpoint, load_checkpoint_G, make_grid, nn,
                             os, remove_overlap, tgm)


class DefinedCPDatasetTest(CPDatasetTest):
    def __init__(self, opt: Namespace):
        super(CPDatasetTest, self).__init__()
        # base setting
        self.opt = opt
        self.root = opt.dataroot
        self.datamode = "test"
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.semantic_nc = opt.semantic_nc
        self.data_path = osp.join(opt.dataroot, opt.datamode)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def update_opt(self, im_name, cloth_array, cloth_mask_array):
        im_names = [im_name]
        c_names = [cloth_array, cloth_mask_array]
        self.im_names = im_names
        self.c_names = {
            'paired': im_names,
            'unpaired': c_names
        }

    def array_to_image(self, array: np.ndarray):
        """Convert a numpy array to PIL Image."""
        # logger.debug('array dtype-> {}', array.dtype)
        return Image.fromarray(cv2.cvtColor(
            array.astype(np.uint8), cv2.COLOR_BGR2RGB))

    def prepare_input(self, cloth_array: np.ndarray,
                      cloth_mask_array: np.ndarray):

        cloth_rgb = self.array_to_image(cloth_array)
        if cloth_rgb.mode != 'RGB':
            cloth_rgb = cloth_rgb.convert('RGB')
        cloth_resize = transforms.Resize(
            self.fine_width, interpolation=2)(cloth_rgb)  # type: ignore
        cloth_trans = self.transform(cloth_resize)

        cloth_mask = self.array_to_image(cloth_mask_array).convert('L')
        cloth_mask_resize = transforms.Resize(
            self.fine_width, interpolation=0)(cloth_mask)

        cloth_mask_array_ = np.array(cloth_mask_resize)
        cloth_mask_array_ = (cloth_mask_array_ >= 128).astype(
            np.float32)
        cloth_mask_tensor = torch.from_numpy(cloth_mask_array_)
        cloth_mask_tensor.unsqueeze_(0)
        return cloth_trans, cloth_mask_tensor

    def __getitem__(self, index):
        im_name = self.im_names[index]
        c_name = {}  # im_name & c_name
        c = {}
        cm = {}
        for key in self.c_names:
            if key == 'paired':
                c_name[key] = self.c_names[key][index]

                c[key] = Image.open(
                    osp.join(
                        self.data_path,
                        'cloth',
                        c_name[key])).convert('RGB')
                c[key] = transforms.Resize(
                    self.fine_width,
                    interpolation=2)(
                    c[key])
                cm[key] = Image.open(
                    osp.join(
                        self.data_path,
                        'cloth-mask',
                        c_name[key]))
                cm[key] = transforms.Resize(
                    self.fine_width,
                    interpolation=0)(
                    cm[key])

                c[key] = self.transform(c[key])  # [-1,1]
                cm_array = np.array(cm[key])
                cm_array = (cm_array >= 128).astype(np.float32)
                cm[key] = torch.from_numpy(cm_array)  # [0,1]
                cm[key].unsqueeze_(0)
            else:
                c_name[key] = str(int(time.time())) + '_output'
                c[key], cm[key] = self.prepare_input(*self.c_names[key])

        # person image
        im_pil_big = Image.open(osp.join(self.data_path, 'image', im_name))
        im_pil = transforms.Resize(
            self.fine_width, interpolation=2)(im_pil_big)

        im = self.transform(im_pil)

        # load parsing image
        parse_name = im_name.replace('.jpg', '.png')
        im_parse_pil_big = Image.open(
            osp.join(
                self.data_path,
                'image-parse-v3',
                parse_name))
        im_parse_pil = transforms.Resize(
            self.fine_width, interpolation=0)(im_parse_pil_big)
        parse = torch.from_numpy(np.array(im_parse_pil)[None]).long()
        # im_parse = self.transform(im_parse_pil.convert('RGB'))

        labels = {
            0: ['background', [0, 10]],
            1: ['hair', [1, 2]],
            2: ['face', [4, 13]],
            3: ['upper', [5, 6, 7]],
            4: ['bottom', [9, 12]],
            5: ['left_arm', [14]],
            6: ['right_arm', [15]],
            7: ['left_leg', [16]],
            8: ['right_leg', [17]],
            9: ['left_shoe', [18]],
            10: ['right_shoe', [19]],
            11: ['socks', [8]],
            12: ['noise', [3, 11]]
        }

        parse_map = torch.FloatTensor(
            20, self.fine_height, self.fine_width).zero_()
        parse_map = parse_map.scatter_(0, parse, 1.0)
        new_parse_map = torch.FloatTensor(
            self.semantic_nc,
            self.fine_height,
            self.fine_width).zero_()

        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_map[i] += parse_map[label]

        parse_onehot = torch.FloatTensor(
            1, self.fine_height, self.fine_width).zero_()
        for i in range(len(labels)):
            for label in labels[i][1]:
                parse_onehot[0] += parse_map[label] * i

        # load image-parse-agnostic
        image_parse_agnostic = Image.open(
            osp.join(
                self.data_path,
                'image-parse-agnostic-v3.2',
                parse_name))
        image_parse_agnostic = transforms.Resize(
            self.fine_width, interpolation=0)(image_parse_agnostic)
        parse_agnostic = torch.from_numpy(
            np.array(image_parse_agnostic)[None]).long()
        image_parse_agnostic = self.transform(
            image_parse_agnostic.convert('RGB'))

        parse_agnostic_map = torch.FloatTensor(
            20, self.fine_height, self.fine_width).zero_()
        parse_agnostic_map = parse_agnostic_map.scatter_(
            0, parse_agnostic, 1.0)
        new_parse_agnostic_map = torch.FloatTensor(
            self.semantic_nc, self.fine_height, self.fine_width).zero_()
        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_agnostic_map[i] += parse_agnostic_map[label]

        # parse cloth & parse cloth mask
        # pcm = new_parse_map[3:4]
        # im_c = im * pcm + (1 - pcm)

        # load pose points
        pose_name = im_name.replace('.jpg', '_rendered.png')
        pose_map = Image.open(
            osp.join(
                self.data_path,
                'openpose_img',
                pose_name))
        pose_map = transforms.Resize(
            self.fine_width, interpolation=2)(pose_map)
        pose_map = self.transform(pose_map)  # [-1,1]

        pose_name = im_name.replace('.jpg', '_keypoints.json')
        with open(osp.join(self.data_path, 'openpose_json', pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints_2d']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))[:, :2]

        # load densepose
        densepose_name = im_name.replace('image', 'image-densepose')
        densepose_map = Image.open(
            osp.join(
                self.data_path,
                'image-densepose',
                densepose_name))
        densepose_map = transforms.Resize(
            self.fine_width, interpolation=2)(densepose_map)
        densepose_map = self.transform(densepose_map)  # [-1,1]
        agnostic = self.get_agnostic(im_pil_big, im_parse_pil_big, pose_data)
        agnostic = transforms.Resize(
            self.fine_width, interpolation=2)(agnostic)
        agnostic = self.transform(agnostic)

        result = {
            'c_name': c_name,     # for visualization
            'im_name': im_name,    # for visualization or ground truth
            # intput 1 (clothfloww)
            'cloth': c,          # for input
            'cloth_mask': cm,   # for input
            # intput 2 (segnet)
            'parse_agnostic': new_parse_agnostic_map,
            'densepose': densepose_map,
            'pose': pose_map,       # for conditioning
            # GT
            'parse_onehot': parse_onehot,  # Cross Entropy
            # 'parse': new_parse_map,  # GAN Loss real
            # 'pcm': pcm,             # L1 Loss & vis
            # 'parse_cloth': im_c,    # VGG Loss & vis
            # visualization
            'image': im,         # for visualization
            'agnostic': agnostic
        }

        return result


class Test:

    is_prepared = False

    def __init__(self, dataroot, tocg_checkpoint, gen_checkpoint):
        # #默认参数详见 get_opt
        self.opt = get_opt()
        self.opt.dataroot = dataroot
        self.opt.tocg_checkpoint = tocg_checkpoint
        self.opt.gen_checkpoint = gen_checkpoint

    def prepare(self):
        self.is_prepared = True
        input1_nc = 4  # cloth + cloth-mask
        input2_nc = self.opt.semantic_nc + 3  # parse_agnostic + densepose
        self.tocg = ConditionGenerator(
            self.opt,
            input1_nc=input1_nc,
            input2_nc=input2_nc,
            output_nc=self.opt.output_nc,
            ngf=96,
            norm_layer=nn.BatchNorm2d)

        # generator
        self.opt.semantic_nc = 7
        self.generator = SPADEGenerator(self.opt, 3 + 3 + 3)
        # self.generator.print_network()

        # Load Checkpoint
        load_checkpoint(self.tocg, self.opt.tocg_checkpoint, self.opt)
        load_checkpoint_G(self.generator, self.opt.gen_checkpoint, self.opt)

        self.gauss = tgm.image.GaussianBlur((15, 15), (3, 3))
        if self.opt.cuda:
            self.gauss.cuda()
            self.tocg.cuda()
        self.tocg.eval()
        self.generator.eval()

    def prepare_test_loader(self, im_name, cloth_array, cloth_mask_array):
        if not hasattr(self, 'test_dataset'):
            self.test_dataset = DefinedCPDatasetTest(self.opt)
        self.test_dataset.update_opt(im_name, cloth_array, cloth_mask_array)
        if not hasattr(self, 'test_loader'):
            self.test_loader = CPDataLoader(self.opt, self.test_dataset)
        if not self.is_prepared:
            try:
                self.prepare()
            except Exception as e:
                logger.exception(e)

        # return test_loader

    def test(self):
        gauss = self.gauss
        opt = self.opt
        tocg = self.tocg
        generator = self.generator
        try:
            with torch.no_grad():
                for inputs in self.test_loader.data_loader:
                    # pose_map = inputs['pose']
                    pre_clothes_mask = inputs['cloth_mask'][opt.datasetting]
                    # label = inputs['parse']
                    parse_agnostic = inputs['parse_agnostic']
                    agnostic = inputs['agnostic']
                    clothes = inputs['cloth'][opt.datasetting]  # target cloth
                    densepose = inputs['densepose']
                    # im = inputs['image']
                    # _, input_parse_agnostic = label, parse_agnostic
                    pre_clothes_mask = torch.FloatTensor(
                        (pre_clothes_mask.detach().cpu().numpy() > 0.5).astype(np.float64))

                    # down
                    # pose_map_down = F.interpolate(
                    # pose_map, size=(256, 192), mode='bilinear')
                    pre_clothes_mask_down = F.interpolate(
                        pre_clothes_mask, size=(256, 192), mode='nearest')
                    # input_label_down = F.interpolate(
                    #     input_label, size=(256, 192), mode='bilinear')
                    input_parse_agnostic_down = F.interpolate(
                        parse_agnostic, size=(256, 192), mode='nearest')
                    # agnostic_down = F.interpolate(
                    #     agnostic, size=(256, 192), mode='nearest')
                    clothes_down = F.interpolate(
                        clothes, size=(256, 192), mode='bilinear')
                    densepose_down = F.interpolate(
                        densepose, size=(256, 192), mode='bilinear')

                    # shape = pre_clothes_mask.shape

                    # multi-task inputs
                    input1 = torch.cat(
                        [clothes_down, pre_clothes_mask_down], 1)
                    input2 = torch.cat(
                        [input_parse_agnostic_down, densepose_down], 1)

                    # forward
                    flow_list, fake_segmap, _, warped_clothmask_paired = tocg(
                        opt, input1, input2)

                    # warped cloth mask one hot
                    if opt.cuda:
                        warped_cm_onehot = torch.FloatTensor(
                            (warped_clothmask_paired.detach().cpu().numpy() > 0.5).astype(np.float64)).cuda()
                    else:
                        warped_cm_onehot = torch.FloatTensor(
                            (warped_clothmask_paired.detach().cpu().numpy() > 0.5).astype(np.float64))

                    if opt.clothmask_composition != 'no_composition':
                        if opt.clothmask_composition == 'detach':
                            cloth_mask = torch.ones_like(fake_segmap)
                            cloth_mask[:, 3:4, :, :] = warped_cm_onehot
                            fake_segmap = fake_segmap * cloth_mask

                        if opt.clothmask_composition == 'warp_grad':
                            cloth_mask = torch.ones_like(fake_segmap)
                            cloth_mask[:, 3:4, :, :] = warped_clothmask_paired
                            fake_segmap = fake_segmap * cloth_mask

                    # make generator input parse map
                    fake_parse_gauss = gauss(F.interpolate(fake_segmap, size=(
                        opt.fine_height, opt.fine_width), mode='bilinear'))
                    fake_parse = fake_parse_gauss.argmax(dim=1)[:, None]

                    if opt.cuda:
                        old_parse = torch.FloatTensor(fake_parse.size(
                            0), 13, opt.fine_height, opt.fine_width).zero_().cuda()
                    else:
                        old_parse = torch.FloatTensor(fake_parse.size(
                            0), 13, opt.fine_height, opt.fine_width).zero_()
                    old_parse.scatter_(1, fake_parse, 1.0)

                    labels = {
                        0: ['background', [0]],
                        1: ['paste', [2, 4, 7, 8, 9, 10, 11]],
                        2: ['upper', [3]],
                        3: ['hair', [1]],
                        4: ['left_arm', [5]],
                        5: ['right_arm', [6]],
                        6: ['noise', [12]]
                    }
                    if opt.cuda:
                        parse = torch.FloatTensor(fake_parse.size(
                            0), 7, opt.fine_height, opt.fine_width).zero_().cuda()
                    else:
                        parse = torch.FloatTensor(fake_parse.size(
                            0), 7, opt.fine_height, opt.fine_width).zero_()
                    for i in range(len(labels)):
                        for label in labels[i][1]:
                            parse[:, i] += old_parse[:, label]

                    # warped cloth
                    N, _, iH, iW = clothes.shape
                    flow = F.interpolate(
                        flow_list[-1].permute(0, 3, 1, 2), size=(iH, iW), mode='bilinear').permute(0, 2, 3, 1)
                    flow_norm = torch.cat(
                        [flow[:, :, :, 0:1] / ((96 - 1.0) / 2.0), flow[:, :, :, 1:2] / ((128 - 1.0) / 2.0)], 3)

                    grid = make_grid(N, iH, iW, opt)
                    warped_grid = grid + flow_norm
                    warped_cloth = F.grid_sample(
                        clothes, warped_grid, padding_mode='border')
                    warped_clothmask = F.grid_sample(
                        pre_clothes_mask, warped_grid, padding_mode='border')
                    if opt.occlusion:
                        warped_clothmask = remove_overlap(
                            F.softmax(fake_parse_gauss, dim=1), warped_clothmask)
                        warped_cloth = warped_cloth * warped_clothmask + \
                            torch.ones_like(warped_cloth) * \
                            (1 - warped_clothmask)

                    output = generator(
                        torch.cat((agnostic, densepose, warped_cloth), dim=1), parse)
                    # save output
                return self.save_image(output)
        except Exception as e:
            logger.exception(e)
            return str(e)

    @staticmethod
    def save_image(img_tensors):
        img_tensor = img_tensors[0]
        tensor = (img_tensor.clone() + 1) * 0.5 * 255
        tensor = tensor.cpu().clamp(0, 255)

        try:
            array = tensor.numpy().astype('uint8')
        except BaseException:
            array = tensor.detach().numpy().astype('uint8')

        if array.shape[0] == 1:
            array = array.squeeze(0)
        elif array.shape[0] == 3:
            array = array.swapaxes(0, 1).swapaxes(1, 2)

        im = Image.fromarray(array)
        io_bytes = BytesIO()
        im.save(io_bytes, format='png')
        return io_bytes.getvalue()
