import os.path as osp
import numpy as np
import mmcv

from torch.utils.data import Dataset
from mmcv.parallel.data_container import DataContainer

from cores.pre_processing.image_transform_de import ImageTransform
from cores.pre_processing.augmentations import Augmentation, random_scale
from cores.misc import dental_detection_classes
from utils.image_utils import to_tensor


class DentalClassDataset(Dataset):
    """dental dataset for detection.

    Annotation format:
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            "pigment": int,
            "soft_deposit": int,
            'ann': {
                'bboxes': <np.ndarray> (n, 4),
                'labels': <np.ndarray> (n, ),
            }
        },
        ...
    ]
    The `ann` field is optional for testing.

    detection labels:
        0: negative
        1: Periodontal_disease (includes periodontitis and gingivitis)
        2：abscess
        3: caries
        4: wedge_shaped_defect
        5: calculus
    """
    CLASSES = (
        'pigment', 'soft_deposit',
    )

    def __init__(
            self,
            ann_file,
            img_prefix,
            img_scale,
            img_norm_cfg,
            multiscale_mode='value',
            flip_ratio=0,
            with_label=True,
            extra_aug=None,
            test_mode=False
    ):
        """Dataset class.

        Args:
            ann_file (List(Dir)): format as above.
            img_prefix (str): dir path.
            img_scale (tuple or List(Tuple)): (h, w)
            img_norm_cfg (mean, std, to_rgb, size_divisor, resize_keep_ratio):
            multiscale_mode (str):
                None for not multi-scale.
                'value' for random take one scale from img_scale.
                'range' for taking as a range.
            flip_ratio (float): flip_ratio = 0 means no flip. should be within [0, 1].
            with_label (bool): with bbox label or not.
            extra_aug (None or dict): augmentation except scaling and flipping.
                Including PhotoMetricDistortion, Expand, and RandomCrop.
                Input list is for their parameters.
            test_mode (bool): test mode.

        Returns:

        """

        # prefix for images path
        self.img_prefix = img_prefix

        # load annotations at once during init
        self.img_infos = self.load_annotations(ann_file)

        # in test mode or not
        self.test_mode = test_mode

        # filter images that are too small
        if not self.test_mode:
            valid_inds = self._filter_imgs()
            self.img_infos = [self.img_infos[i] for i in valid_inds]

        # (long_edge, short_edge) or [(long1, short1), (long2, short2), ...]
        self.img_scales = img_scale if isinstance(img_scale, list) else [img_scale]
        assert mmcv.is_list_of(self.img_scales, tuple)

        # normalization configs
        self.img_norm_cfg = img_norm_cfg

        # multi-scale mode (only applicable for multi-scale training)
        self.multiscale_mode = multiscale_mode
        assert multiscale_mode in ['value', 'range']

        # flip ratio
        self.flip_ratio = flip_ratio
        assert 0 <= flip_ratio <= 1

        # with label is False for RPN
        self.with_label = with_label

        # transforms
        self.img_transform = ImageTransform(
            mean=img_norm_cfg['mean'],
            std=img_norm_cfg['std'],
            to_rgb=img_norm_cfg['to_rgb'],
        )

        # if use extra augmentation
        if extra_aug is not None:
            self.extra_aug = Augmentation(**extra_aug)
        else:
            self.extra_aug = None

        # image padding color values
        self.pad_values = img_norm_cfg['pad_values']

        # image rescale if keep ratio
        self.resize_keep_ratio = img_norm_cfg['resize_keep_ratio']

    def __len__(self):
        return len(self.img_infos)

    def _rand_another(self):
        return np.random.choice(list(range(0, len(self))))

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another()
                continue
            return data

    def prepare_test_img(self, idx):
        """Prepare an image for testing"""
        img_info = self.img_infos[idx]

        # load image
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))

        img, img_shape, pad_shape, scale_factor = \
            self.img_transform(
                img=img, scale=self.img_scales[0], flip=False, pad_val=self.pad_values, keep_ratio=self.resize_keep_ratio
            )
        img_meta = dict(
            ori_shape=(img_info['height'], img_info['width'], 3),
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=False
        )

        data = dict(
            img=to_tensor(img),
            img_meta=DataContainer(img_meta, cpu_only=True)
        )
        return data

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]

        # load image
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))

        ann = self.get_ann_info(idx)
        gt_labels = np.zeros([len(self.CLASSES)], dtype=np.float32)
        for index, item in enumerate(self.CLASSES):
            gt_labels[index] = int(ann[item] == 1)

        # aug 1: apply extra augmentation
        if self.extra_aug is not None:
            img, gt_bboxes, gt_labels = \
                self.extra_aug(img, None, gt_labels)

        # aug 2: apply ordinary augmentations: flipping
        flip = True if np.random.rand() < self.flip_ratio else False

        # aug 3: apply ordinary augmentations: scaling
        img_scale = random_scale(self.img_scales, self.multiscale_mode)

        img, img_shape, pad_shape, scale_factor = \
            self.img_transform(
                img=img, scale=img_scale, flip=flip, pad_val=self.pad_values, keep_ratio=self.resize_keep_ratio
            )

        img = img.copy()

        img_meta = dict(
            ori_shape=(img_info['height'], img_info['width'], 3),
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip
        )

        data = dict(
            img=DataContainer(to_tensor(img), stack=True),
            img_meta=DataContainer(data=img_meta, cpu_only=True),
        )
        if self.with_label:
            data['gt_labels'] = DataContainer(to_tensor(gt_labels), stack=False)

        return data

    def load_annotations(self, ann_file):

        annotations = mmcv.load(ann_file)

        # correct label counts
        # full cat: [
        #   'Periodontal_disease', 'abscess', 'caries', 'wedge_shaped_defect', 'calculus'
        # ]
        cat = dental_detection_classes()
        self.cat2label = {}
        start_count = 1
        for i, cat_name in enumerate(cat):
            if cat_name in self.CLASSES:
                self.cat2label[i+1] = start_count
                start_count = start_count + 1
            else:
                self.cat2label[i + 1] = -1
        print('label-category mapping: {}'.format(self.cat2label))

        return annotations

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def get_ann_info(self, idx):
        ann_info = self.img_infos[idx]
        return ann_info
