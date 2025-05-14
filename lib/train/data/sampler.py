import pdb
import random
import torch.utils.data
from lib.utils import TensorDict
import numpy as np


def no_processing(data):
    return data



class TrackingSampler(torch.utils.data.Dataset):
    """ Class responsible for sampling frames from training sequences to form batches. 

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is selected
    from that dataset. A base frame is then sampled randomly from the sequence. Next, a set of 'train frames' and
    'test frames' are sampled from the sequence from the range [base_frame_id - max_gap, base_frame_id]  and
    (base_frame_id, base_frame_id + max_gap] respectively. Only the frames in which the target is visible are sampled.
    If enough visible frames are not found, the 'max_gap' is increased gradually till enough frames are found.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    """

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_search_frames, num_template_frames=1, processing=no_processing, frame_sample_mode='causal',
                 train_cls=False, pos_prob=0.5):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the train frames and the test frames.
            num_search_frames - Number of search frames to sample.
            num_template_frames - Number of template frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
            frame_sample_mode - Either 'causal' or 'interval'. If 'causal', then the test frames are sampled in a causally,
                                otherwise randomly within the interval.
        """
        self.num = 1  # NUM 替换为实例变量
        self.datasets = datasets
        self.train_cls = train_cls  # whether we are training classification
        self.pos_prob = pos_prob  # probability of sampling positive class when making classification

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_search_frames = num_search_frames
        self.num_template_frames = num_template_frames
        self.processing = processing
        self.frame_sample_mode = frame_sample_mode
        # control the selection of the seq
        self.inbatchsize = 1
        self.seq_id = None


    def __len__(self):
        return self.samples_per_epoch

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None,
                            allow_invisible=False, force_invisible=False):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if num_ids == 0:
            return []
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)
        # get valid ids
        if force_invisible:
            valid_ids = [i for i in range(min_id, max_id) if not visible[i]]
        else:
            if allow_invisible:
                valid_ids = [i for i in range(min_id, max_id)]
            else:
                valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    def __getitem__(self, index):
        if self.train_cls:
            return self.getitem_cls()
        else:
            return self.getitem()

    def getitem(self):

        """
        returns:
            TensorDict - dict containing all the data blocks
        """
        valid = False

        while not valid:
            dataset = random.choices(self.datasets, self.p_datasets)[0]
            is_video_dataset = dataset.is_video_sequence()

            # 检查是否需要重新选择 seq_id
            if self.inbatchsize > self.num:
                self.seq_id = None
                self.inbatchsize = 1

            # 根据 seq_id 是否为空选择采样方法
            if self.seq_id is None:
                seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)
                self.seq_id = seq_id
            else:
                seq_id, visible, seq_info_dict = self.sample_seq_from_dataset_fixed(dataset, is_video_dataset, seq_id=self.seq_id)
            # 其他采样逻辑...
            

            if is_video_dataset:
                template_frame_ids = None
                search_frame_ids = None
                gap_increase = 0

                if self.frame_sample_mode == 'causal':
                    # Sample test and train frames in a causal manner, i.e. search_frame_ids > template_frame_ids
                    while search_frame_ids is None:
                        base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_template_frames - 1,
                                                                 max_id=len(visible) - self.num_search_frames)
                        prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_template_frames - 1,
                                                                  min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                                  max_id=base_frame_id[0])
                        if prev_frame_ids is None:
                            gap_increase += 5
                            continue
                        template_frame_ids = base_frame_id + prev_frame_ids
                        search_frame_ids = self._sample_visible_ids(visible, min_id=template_frame_ids[0] + 1,
                                                                  max_id=template_frame_ids[0] + self.max_gap + gap_increase,
                                                                  num_ids=self.num_search_frames)
                        # Increase gap until a frame is found
                        gap_increase += 5

                elif self.frame_sample_mode == "trident" or self.frame_sample_mode == "trident_pro":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_trident(visible)
                elif self.frame_sample_mode == "stark":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_stark(visible, seq_info_dict["valid"])
                else:
                    raise ValueError("Illegal frame sample mode")
            else:
                # In case of image dataset, just repeat the image to generate synthetic video
                template_frame_ids = [1] * self.num_template_frames
                search_frame_ids = [1] * self.num_search_frames
            try:
                template_frames, template_anno, meta_obj_train = dataset.get_frames(seq_id, template_frame_ids, seq_info_dict)
                search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)
                search_full_images = search_frames
                #get full images for training
                blocks, labels,centers = self.process_single_image(search_full_images[0],search_anno['bbox'])

                data = TensorDict({'ID':seq_id,
                                    'template_images': template_frames,
                                    'template_anno': template_anno['bbox'],
                                    'search_images': search_frames,
                                    'search_anno': search_anno['bbox'],
                                    # 'search_full_images':torch.tensor(search_full_images),
                                    'dataset': dataset.get_name(),
                                    'images_blocks':blocks,
                                    'blocks_labels':labels,
                                    't_modal_up':template_anno['modality_up'],
                                    't_modal_down':template_anno['modality_down'],
                                    's_modal_up':search_anno['modality_up'],
                                    's_modal_down':search_anno['modality_down'],
                                    'test_class': meta_obj_test.get('object_class_name')})
                # make data augmentation

                data = self.processing(data)

                # check whether data is valid
                valid = data['valid']
            except:
                valid = False
            # 假设采样成功，增加 inbatchsize
            if valid:
                self.inbatchsize += 1
        return data

    def getitem_cls(self):
        # get data for classification
        """
        args:
            index (int): Index (Ignored since we sample randomly)
            aux (bool): whether the current data is for auxiliary use (e.g. copy-and-paste)

        returns:
            TensorDict - dict containing all the data blocks
        """
        valid = False
        label = None
        while not valid:
            # Select a dataset
            dataset = random.choices(self.datasets, self.p_datasets)[0]

            is_video_dataset = dataset.is_video_sequence()

            # sample a sequence from the given dataset

            seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)

            # sample template and search frame ids
            if is_video_dataset:
                if self.frame_sample_mode in ["trident", "trident_pro"]:
                    template_frame_ids, search_frame_ids = self.get_frame_ids_trident(visible)
                elif self.frame_sample_mode == "stark":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_stark(visible, seq_info_dict["valid"])
                else:
                    raise ValueError("illegal frame sample mode")
            else:
                # In case of image dataset, just repeat the image to generate synthetic video
                template_frame_ids = [1] * self.num_template_frames
                search_frame_ids = [1] * self.num_search_frames
            try:
                # "try" is used to handle trackingnet data failure
                # get images and bounding boxes (for templates)
                template_frames, template_anno, meta_obj_train = dataset.get_frames(seq_id, template_frame_ids,
                                                                                    seq_info_dict)
                H, W, _ = template_frames[0].shape
                template_masks = template_anno['mask'] if 'mask' in template_anno else [torch.zeros(
                    (H, W))] * self.num_template_frames
                # get images and bounding boxes (for searches)
                # positive samples
                if random.random() < self.pos_prob:
                    label = torch.ones(1,)
                    search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)
                    search_masks = search_anno['mask'] if 'mask' in search_anno else [torch.zeros(
                        (H, W))] * self.num_search_frames
                # negative samples
                else:
                    label = torch.zeros(1,)
                    if is_video_dataset:
                        search_frame_ids = self._sample_visible_ids(visible, num_ids=1, force_invisible=True)
                        if search_frame_ids is None:
                            search_frames, search_anno, meta_obj_test = self.get_one_search()
                        else:
                            search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids,
                                                                                           seq_info_dict)
                            search_anno["bbox"] = [self.get_center_box(H, W)]
                    else:
                        search_frames, search_anno, meta_obj_test = self.get_one_search()
                    H, W, _ = search_frames[0].shape
                    search_masks = search_anno['mask'] if 'mask' in search_anno else [torch.zeros(
                        (H, W))] * self.num_search_frames

                data = TensorDict({'template_images': template_frames,
                                   'template_anno': template_anno['bbox'],
                                   'template_masks': template_masks,
                                   'search_images': search_frames,
                                   'search_anno': search_anno['bbox'],
                                   'search_masks': search_masks,
                                   'dataset': dataset.get_name(),
                                   'test_class': meta_obj_test.get('object_class_name')})
                
        
                # make data augmentation
                data = self.processing(data)
                # add classification label
                data["label"] = label
                # check whether data is valid
                valid = data['valid']
            except:
                valid = False

        return data

    def get_center_box(self, H, W, ratio=1/8):
        cx, cy, w, h = W/2, H/2, W * ratio, H * ratio
        return torch.tensor([int(cx-w/2), int(cy-h/2), int(w), int(h)])

    def sample_seq_from_dataset(self, dataset, is_video_dataset):

        # Sample a sequence with enough visible frames
        enough_visible_frames = False
        while not enough_visible_frames:
           # Sample a sequence
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)

            # Sample frames
            seq_info_dict = dataset.get_sequence_info(seq_id)
            visible = seq_info_dict['visible']

            enough_visible_frames = visible.type(torch.int64).sum().item() > 2 * (
                    self.num_search_frames + self.num_template_frames) and len(visible) >= 20

            enough_visible_frames = enough_visible_frames or not is_video_dataset
        return seq_id, visible, seq_info_dict
    
    def sample_seq_from_dataset_fixed(self, dataset, is_video_dataset,seq_id):

        # Sample a sequence with enough visible frames
        enough_visible_frames = False
        while not enough_visible_frames:
            # Sample a sequence
            # seq_id = random.randint(0, dataset.get_num_sequences() - 1)

            # Sample frames
            seq_info_dict = dataset.get_sequence_info(seq_id)
            visible = seq_info_dict['visible']
            enough_visible_frames = visible.type(torch.int64).sum().item() > 2 * (
                    self.num_search_frames + self.num_template_frames) and len(visible) >= 20

            enough_visible_frames = enough_visible_frames or not is_video_dataset
        return seq_id, visible, seq_info_dict


    def get_one_search(self):
        # Select a dataset
        dataset = random.choices(self.datasets, self.p_datasets)[0]

        is_video_dataset = dataset.is_video_sequence()
        # sample a sequence
        seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)
        # sample a frame
        if is_video_dataset:
            if self.frame_sample_mode == "stark":
                search_frame_ids = self._sample_visible_ids(seq_info_dict["valid"], num_ids=1)
            else:
                search_frame_ids = self._sample_visible_ids(visible, num_ids=1, allow_invisible=True)
        else:
            search_frame_ids = [1]
        # get the image, bounding box and other info
        search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)

        return search_frames, search_anno, meta_obj_test
    


    def get_frame_ids_trident(self, visible):
        # get template and search ids in a 'trident' manner
        template_frame_ids_extra = []
        while None in template_frame_ids_extra or len(template_frame_ids_extra) == 0:
            template_frame_ids_extra = []
            # first randomly sample two frames from a video
            template_frame_id1 = self._sample_visible_ids(visible, num_ids=1)  # the initial template id
            search_frame_ids = self._sample_visible_ids(visible, num_ids=1)  # the search region id
            # get the dynamic template id
            for max_gap in self.max_gap:
                if template_frame_id1[0] >= search_frame_ids[0]:
                    min_id, max_id = search_frame_ids[0], search_frame_ids[0] + max_gap
                else:
                    min_id, max_id = search_frame_ids[0] - max_gap, search_frame_ids[0]
                if self.frame_sample_mode == "trident_pro":
                    f_id = self._sample_visible_ids(visible, num_ids=1, min_id=min_id, max_id=max_id,
                                                    allow_invisible=True)
                else:
                    f_id = self._sample_visible_ids(visible, num_ids=1, min_id=min_id, max_id=max_id)
                if f_id is None:
                    template_frame_ids_extra += [None]
                else:
                    template_frame_ids_extra += f_id

        template_frame_ids = template_frame_id1 + template_frame_ids_extra
        return template_frame_ids, search_frame_ids

    def get_frame_ids_stark(self, visible, valid):
        # get template and search ids in a 'stark' manner
        template_frame_ids_extra = []
        while None in template_frame_ids_extra or len(template_frame_ids_extra) == 0:
            template_frame_ids_extra = []
            # first randomly sample two frames from a video
            template_frame_id1 = self._sample_visible_ids(visible, num_ids=1)  # the initial template id
            search_frame_ids = self._sample_visible_ids(visible, num_ids=1)  # the search region id
            # get the dynamic template id
            for max_gap in self.max_gap:
                if template_frame_id1[0] >= search_frame_ids[0]:
                    min_id, max_id = search_frame_ids[0], search_frame_ids[0] + max_gap
                else:
                    min_id, max_id = search_frame_ids[0] - max_gap, search_frame_ids[0]
                """we require the frame to be valid but not necessary visible"""
                f_id = self._sample_visible_ids(valid, num_ids=1, min_id=min_id, max_id=max_id)
                if f_id is None:
                    template_frame_ids_extra += [None]
                else:
                    template_frame_ids_extra += f_id

        template_frame_ids = template_frame_id1 + template_frame_ids_extra
        return template_frame_ids, search_frame_ids

            
    def process_single_image(self, image, bboxes, block_size=256):
        """
        处理单张图像，确保目标块随机排列
        :param image: 输入图像，形状为[h, w, 6]
        :param bboxes: 目标框列表，每个目标框为[x,y,w,h]格式
        :return: (blocks, labels, centers) - 包含图像块、标签和每个块中心坐标的元组
        """
        h, w, c = image.shape
        assert c == 6, "输入图像应为6通道"
        
        # 计算需要填充的尺寸，确保至少2x2=4个块
        pad_h = max(0, 2 * block_size - h)
        pad_w = max(0, 2 * block_size - w)
        
        # 对图像进行对称填充（不影响目标框坐标）
        if pad_h > 0 or pad_w > 0:
            image = np.pad(image, 
                        ((0, pad_h), (0, pad_w), (0, 0)),
                        mode='edge')
            h, w = image.shape[0], image.shape[1]
        
        # 计算分块数量（现在至少是2x2）
        num_blocks_h = max(2, (h + block_size - 1) // block_size)
        num_blocks_w = max(2, (w + block_size - 1) // block_size)
        
        # 存储目标中心块坐标
        target_blocks = []
        
        # 处理每个目标框（使用原始坐标，不受填充影响）
        for bbox in bboxes:
            x, y, box_w, box_h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            # 计算目标中心点
            center_x = x + box_w / 2
            center_y = y + box_h / 2
            
            # 计算中心点所在的块索引
            block_x = int(center_x // block_size)
            block_y = int(center_y // block_size)
            
            # 确保索引在有效范围内
            if 0 <= block_x < num_blocks_w and 0 <= block_y < num_blocks_h:
                target_blocks.append((block_y, block_x))  # 存储块坐标
        

        
        # 提取图像块 - 确保至少有4个块（2x2网格）
        blocks = []
        labels = []
        centers = []  # 存储每个块的中心坐标
        
        # 1. 添加所有目标中心块（现在顺序已随机）
        for by, bx in target_blocks:
            y_start = by * block_size
            y_end = min((by + 1) * block_size, h)
            x_start = bx * block_size
            x_end = min((bx + 1) * block_size, w)
            
            block = image[y_start:y_end, x_start:x_end, :]
            if block.shape[0] < block_size or block.shape[1] < block_size:
                padded_block = np.zeros((block_size, block_size, 6), dtype=image.dtype)
                padded_block[:block.shape[0], :block.shape[1], :] = block
                block = padded_block
            
            blocks.append(block)
            labels.append(1)  # 目标块标签为1
            # 计算块中心坐标 (相对于原始图像)
            center_y = (y_start + y_end) / 2
            center_x = (x_start + x_end) / 2
            centers.append((center_x, center_y))
        
        # 2. 确保至少有4个块（2x2网格）
        required_blocks = 2
        current_blocks = len(blocks)
        
        # 获取所有非目标块的坐标
        non_target_coords = []
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                if (i, j) not in target_blocks:
                    non_target_coords.append((i, j))
        
        # 计算需要添加的非目标块数量
        needed_blocks = max(0, required_blocks - current_blocks)
        num_random_blocks = min(needed_blocks, len(non_target_coords))
        
        # 随机选择非目标块
        if num_random_blocks > 0:
            random_indices = np.random.choice(len(non_target_coords), num_random_blocks, replace=False)
            for idx in random_indices:
                i, j = non_target_coords[idx]
                y_start = i * block_size
                y_end = min((i + 1) * block_size, h)
                x_start = j * block_size
                x_end = min((j + 1) * block_size, w)
                
                block = image[y_start:y_end, x_start:x_end, :]
                if block.shape[0] < block_size or block.shape[1] < block_size:
                    padded_block = np.zeros((block_size, block_size, 6), dtype=image.dtype)
                    padded_block[:block.shape[0], :block.shape[1], :] = block
                    block = padded_block
                
                blocks.append(block)
                labels.append(0)  # 非目标块标签为0
                # 计算块中心坐标 (相对于原始图像)
                center_y = (y_start + y_end) / 2
                center_x = (x_start + x_end) / 2
                centers.append((center_x, center_y))

        # 打乱目标块的顺序
        # np.random.shuffle(target_blocks)  # 关键修改：随机打乱目标块顺序
        paired_data = list(zip(blocks, labels,centers))
        np.random.shuffle(paired_data)  # 打乱顺序
        blocks, labels,centers = zip(*paired_data)

        # 3. 如果仍然不足4个块（极少数情况），随机复制已有块
        # while len(blocks) < required_blocks:
        #     idx = np.random.randint(0, len(blocks))
        #     blocks.append(blocks[idx].copy())
        #     labels.append(labels[idx])
        #     centers.append(centers[idx])
        
        return torch.tensor(np.array(blocks)), torch.tensor([labels]), torch.tensor(centers)