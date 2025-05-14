import math
from lib.models.bat import build_batrack
from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os
import vot
from lib.test.tracker.data_utils import PreprocessorMM
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond


class BATTrack(BaseTracker):
    def __init__(self, params):
        super(BATTrack, self).__init__(params)
        network = build_batrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)  
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = PreprocessorMM()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        if getattr(params, 'debug', None) is None:
            setattr(params, 'debug', 0)
        self.use_visdom = True #params.debug   
        #self._init_visdom(None, 1)
        self.debug = params.debug
        self.frame_id = 0
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes

    def initialize(self, image, info: dict):
        # 判断是否重定位，如果为false则对当前搜索区域进行判断，如果为True,则对全局图像进行粗重定位
        self.localization = False
        self.max_score_list = []

        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr  = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr)
        with torch.no_grad():
            self.z_tensor = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.device, template_bbox)

        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, info['init_bbox'], self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)

        #保留初始目标框 然后根据新搜索区域预测的目标得到新的模板，根据新模板预测初始搜索区域中目标框，根据iou判断目标是否检测正确
        self.init_bbox = info['init_bbox']
        self.init_search = self.preprocessor.process(x_patch_arr)

        self.last_search = self.init_search

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)

        img_blocks = torch.cat([self.last_search,search])

        with torch.no_grad():
            x_tensor = search
            # merge the template and the search
            # run the transformer
            if not self.localization:
                out_dict = self.network.forward(
                    template=self.z_tensor, search=x_tensor, ce_template_mask=self.box_mask_z,Test=True,img_blocks=img_blocks)
        # pred_confidence = out_dict['localizaiton_results']
        # pred_confidence_softmax = out_dict['Softmax_localizaiton_results']
        # max_value,max_id = torch.max(pred_confidence_softmax,dim=-1)
        # print(self.frame_id,pred_confidence_softmax)

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes, best_score = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'], return_score=True)
        max_score = best_score[0][0].item()
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        self.max_score_list.append(max_score)
        if len(self.max_score_list)>20:
            self.max_score_list.pop(0)
        average_max_score = sum(self.max_score_list)/len(self.max_score_list)

        # if max_score < average_max_score and max_id.item() == 0 and pred_confidence_softmax[0][0]-pred_confidence_softmax[0][1]>0.5:
        #     print("当前帧失去目标或模态切换")
        #更新上一帧搜索区域
        self.last_search = search
        #self.debug = 1
        
        # for debug
        if self.debug == 1:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image[:,:,:3], cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
            cv2.putText(image_BGR, 'max_score:' + str(round(max_score, 3)), (40, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 255), 2)
            cv2.imshow('debug_vis', image_BGR)
            cv2.waitKey(1)


        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save,
                    "best_score": max_score}
        else:
            return {"target_bbox": self.state,
                    "best_score": max_score}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


    def generate_candidate_boxes(self, image, prev_bbox, window_size=256):
        h, w = image.shape[:2]
        x_,y_,w_,h_ = prev_bbox
        candidate_boxes = []
        
        # 计算x方向分块起始点（覆盖边缘）
        x_steps = list(range(0, w - window_size + 1, window_size))
        if (w - x_steps[-1]) > 0:  # 添加右边缘分块
            x_steps.append(w - window_size)
        
        # 计算y方向分块起始点（同上）
        y_steps = list(range(0, h - window_size + 1, window_size))
        if (h - y_steps[-1]) > 0:  # 添加下边缘分块
            y_steps.append(h - window_size)
        
        # 生成候选框
        for y in y_steps:
            for x in x_steps:
                x1 = x
                y1 = y
                x2 = x + window_size
                y2 = y + window_size
                # 裁剪越界分块
                x2 = min(x2, w)
                y2 = min(y2, h)
                w_box = x2 - x1
                h_box = y2 - y1
                candidate_boxes.append([x1, y1,w_,h_])
        
        return candidate_boxes


def get_tracker_class():
    return BATTrack
