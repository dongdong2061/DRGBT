
from rgbt.utils import *
import os
from rgbt.vis import draw_radar, draw_plot
from rgbt.metrics import PR_LasHeR,SR_LasHeR,NPR,PR,SR
from rgbt import __file__ as basepath

# File paths for evaluating the results of the DEMT dataset
_basepath = 'evaluate_DRGBT603'

def initial_gt_file(gt_path:str, seqs:list, v_name:str, i_name:str, bbox_trans):
    res = {}
    for seq_name in seqs:
        serial_v = load_text(os.path.join(gt_path, seq_name, v_name))
        serial_i = load_text(os.path.join(gt_path, seq_name, i_name))
        seq_serial = {'visible': serial_process(bbox_trans, serial_v), 'infrared': serial_process(bbox_trans,serial_i)}
        res[seq_name] = seq_serial
    return res


def initial_result_file(path:str, seqs:list, bbox_trans, prefix=''):
    res = {}
    for seq_name in seqs:
        serial = load_text(os.path.join(path, prefix+seq_name+'.txt')).round(0)
        res[seq_name] = serial_process(bbox_trans, serial)
    return res

class TrackerResult:
    """
    Your tracking result.
    """
    def __init__(self, tracker_name, path:str, seqs:list, prefix:str, bbox_type:str) -> None:
        self.tracker_name = tracker_name
        self.seqs_name = seqs
        self.bbox_transfun = bbox_type_trans(bbox_type, 'ltwh')
        self.seqs_result = initial_result_file(path, seqs, self.bbox_transfun, prefix)
        self.bbox_type = 'ltwh'

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.seqs_result[self.seqs_name[index]]
        elif isinstance(index, str):
            return self.seqs_result[index]
        else:
            raise KeyError

    def __len__(self):
        return len(self.seqs_name)


class BaseRGBTDataet:
    """
    ground truth.
    """
    def __init__(self, gt_path:str, seqs:list, bbox_type:str, v_name=None, i_name=None) -> None:
        """
        [in] gt_path - str
            The ground truth file path.
        [in] seqs - list
            A list contain all sequence name in one dataset.
        [in] bbox_type - str
            Default is 'ltwh' (top left corner coordinates with width and height), you can also 
            choose 'ltrb' (top left corner and bottom left corner coordinates), 'xywh' (center 
            point coordinates with width and height). 
        [in] v_name - str
            The ground truth file name of visible images.
        [in] i_name - str
            The ground truth file name of infrared images.
        """
        self.gt_path = gt_path

        self.bbox_transfun = bbox_type_trans(bbox_type, 'ltwh')
        self.bbox_type = 'ltwh'

        self.seqs_name = seqs
        self.ALL = tuple(self.seqs_name)
        if v_name!=None and i_name!=None:
            self.seqs_gt = initial_gt_file(self.gt_path, seqs, v_name, i_name, self.bbox_transfun)    # ground truth
        else:
            self.seqs_gt = initial_result_file(self.gt_path, self.seqs_name, self.bbox_transfun, prefix='')

        self.trackers = {}


    def __len__(self):
        return len(self.seqs_name)
    

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.seqs_gt[self.seqs_name[index]]
        elif isinstance(index, str):
            return self.seqs_gt[index]
        else:
            raise KeyError


    def __call__(self, tracker_name, result_path:str, seqs=None, prefix='', bbox_type='ltwh') -> TrackerResult:
        """
        Return the tracker result instance.
        """
        if seqs==None:
            seqs=self.seqs_name
        self.trackers[tracker_name] = TrackerResult(tracker_name, result_path, seqs, prefix, bbox_type)
        return self.trackers[tracker_name]


    def choose_serial_by_att(self, attr):
        raise ImportError


    def get_attr_list(self):
        raise ImportError


    def draw_attributeRadar(self, metric_fun, filename, **argdict):
        """
        Draw a radar chart with all challenge attributes.
        """
        result = [[tracker_name, []] for tracker_name in self.trackers.keys()]
        for attr in self.get_attr_list():
            dict = metric_fun(seqs=getattr(self, attr))
            for i,(k,v) in enumerate(dict.items()):
                result[i][1].append(v[0])

        draw_radar(result=result, attrs=self.get_attr_list(), fn=filename, **argdict)


    def draw_plot(self, axis, metric_fun, filename, y_max:float, y_min:float, title=None, 
                  seqs=None, loc="best", rank="descend", **argdict):
        if seqs==None:
            seqs = self.ALL
        
        result = [[tracker_name, []] for tracker_name in self.trackers.keys()]
        dict = metric_fun(seqs=seqs)
        vals = []
        for i,(k,v) in enumerate(dict.items()):
            vals.append(v[0])
            result[i][0]+=f"[{round(v[0],3)}]"
            result[i][1]=v[1].mean(0)
        if rank=="descend":
            idx = sorted(range(len(vals)), key=lambda x:vals[x], reverse=True)
        else:
            idx = sorted(range(len(vals)), key=lambda x:vals[x], reverse=False)
        result = [result[i] for i in idx]
        
        draw_plot(axis=axis, result=result, fn=filename, title=title, y_max=y_max, y_min=y_min, loc=loc, **argdict)


class DRGBT603(BaseRGBTDataet):
    """
    """
    def __init__(self, gt_path=f'{_basepath}/gt_files/',
                 seq_name_path=f"{_basepath}/test_set.txt") -> None:
        seqs = load_text(seq_name_path, dtype=str)
        super().__init__(gt_path=gt_path, seqs=seqs, bbox_type='ltwh')

        self.name = 'DEMT_test'
        self.PR_fun = PR()
        self.SR_fun = SR()
        self.NPR_fun = NPR()


    def get_attr_list(self):
        return self._attr_list

    def choose_serial_by_att(self, attr):
        if attr==self.ALL:
            return self.seqs_name
        else:
            seqs = []
            for seq in self.seqs_name:
                i = self.get_attr_list().index(attr)
                path = os.path.join(self.gt_path, '..', 'AttriSeqsTxt', seq+'.txt')
                p = load_text(path)[i]
                if p==1.:
                    seqs.append(seq)
            return seqs

    def PR(self, tracker_name=None, seqs=None):
        """
        Parameters
        ----------
        [in] tracker_name - str
            Default is None, evaluate all registered trackers.
        [in] seqs - list
            Sequence to be evaluated, default is all.
        
        Returns
        -------
        [out0] When evaluating a single tracker, return MPR and the precision Rate at different thresholds.
        [out1] Other cases return a dictionary with all tracker results.
        """
        if seqs==None:
            seqs = self.seqs_name

        if tracker_name!=None:
            return self.PR_fun(self, self.trackers[tracker_name], seqs)
        else:
            res = {}
            for k,v in self.trackers.items():
                res[k] = self.PR_fun(self, v, seqs)
            return res

    def NPR(self, tracker_name=None, seqs=None):
        """
        """
        if seqs==None:
            seqs = self.seqs_name

        if tracker_name!=None:
            return self.NPR_fun(self, self.trackers[tracker_name], seqs)
        else:
            res = {}
            for k,v in self.trackers.items():
                res[k] = self.NPR_fun(self, v, seqs)
            return res


    def SR(self, tracker_name=None, seqs=None):
        """
        Parameters
        ----------
        [in] tracker_name - str
            Default is None, evaluate all registered trackers.
        [in] seqs - list
            Sequence to be evaluated, default is all.
        """
        if seqs==None:
            seqs = self.seqs_name

        if tracker_name!=None:
            return self.SR_fun(self, self.trackers[tracker_name], seqs)
        else:
            res = {}
            for k,v in self.trackers.items():
                res[k] = self.SR_fun(self, v, seqs)
            return res


    def draw_attributeRadar(self, metric_fun, filename=None):
        if filename==None:
            filename = self.name
            if metric_fun==self.PR:
                filename+="_PR"
            elif metric_fun==self.SR:
                filename+="_SR"
            filename+="_radar.png"
        return super().draw_attributeRadar(metric_fun, filename)
    

    def draw_plot(self, metric_fun, filename=None, title=None, seqs=None):
        assert metric_fun in [self.NPR, self.PR, self.SR]
        if filename==None:
            filename = self.name
            if metric_fun==self.PR:
                filename+="_PR"
                axis = self.PR_fun.thr
                loc = "lower right"
                x_label = "Location error threshold"
                y_label = "Precision"
            elif metric_fun==self.NPR:
                filename+="_NPR"
                axis = self.NPR_fun.thr
                loc = "lower right"
                x_label = "Normalized Location error threshold"
                y_label = "Normalized Precision"
            elif metric_fun==self.SR:
                filename+="_SR"
                axis = self.SR_fun.thr
                loc = "lower left"
                x_label = "Overlap threshold"
                y_label = "Success Rate"
            filename+="_plot.png"

        if title==None:
            if metric_fun==self.PR:
                title="Precision plots of OPE on LasHeR"
            elif metric_fun==self.NPR:
                title="Normalized Precision plots of OPE on LasHeR"
            elif metric_fun==self.SR:
                title="Success plots of OPE on LasHeR"

        return super().draw_plot(axis=axis, 
                                 metric_fun=metric_fun, 
                                 filename=filename, 
                                 title=title, 
                                 seqs=seqs, y_max=1.0, y_min=0.0, loc=loc,
                                 x_label=x_label, y_label=y_label)
    

class Metric:
    def __init__(self) -> None:
        pass

    def __call__(self, dataset:BaseRGBTDataet, res:TrackerResult):
        pass


    def __call__(self, dataset:BaseRGBTDataet):
        self(dataset, dataset.trackers)


class SR_DEMT(Metric):
    """
    Success Rate.
    Different other dataset, DEMT testingset need to filter some results.
    """
    def __init__(self, thr=np.linspace(0, 1, 21)) -> None:
        super().__init__()
        self.thr = thr


    def __call__(self, dataset:BaseRGBTDataet, result:TrackerResult, seqs:list):
    
        sr=[]
        for seq_name in seqs:
            try:
                gt = dataset[seq_name]
                serial = result[seq_name]
                serial[0] = gt[0]       # ignore the first frame
            except:
                gt = dataset[seq_name]['visible']
                serial = result[seq_name]
                serial[0] = gt[0]       # ignore the first frame
            # cut off tracking result
            serial = serial[:len(gt)]   
            # handle the invailded tracking result
            for i in range(1, len(gt)):
                if serial[i][2]<=0 or serial[i][3]<=0:
                    serial[i] = serial[i-1].copy()
            res = np.array(serial_process(IoU, serial, gt))

            for i in range(len(gt)):
                if sum(gt[i]<=0):
                    res[i]=-1

            sr_cell = []
            for i in self.thr:
                sr_cell.append(np.sum(res>i)/len(res))
            sr.append(sr_cell)

        sr = np.array(sr)
        sr_val = sr.mean()
        return sr_val, sr
    

class PR_DEMT(Metric):
    """
    Precision Rate.
    Different other dataset, DEMT testingset need to filter some results.
    """
    def __init__(self, thr=np.linspace(0, 50, 51)) -> None:
        super().__init__()
        self.thr = thr


    def __call__(self, dataset:BaseRGBTDataet, result:TrackerResult, seqs:list):
        pr=[]
        for seq_name in seqs:
            try:
                gt = dataset[seq_name]
                serial = result[seq_name]
                serial[0] = gt[0]       # ignore the first frame
            except:
                gt = dataset[seq_name]['visible']
                serial = result[seq_name]
                serial[0] = gt[0]       # ignore the first frame
            # cut off tracking result
            serial = serial[:len(gt)]   
            # handle the invailded tracking result
            for i in range(1, len(gt)):
                if serial[i][2]<=0 or serial[i][3]<=0:
                    serial[i] = serial[i-1].copy()
            res = np.array(serial_process(CLE, serial, gt))

            for i in range(len(gt)):
                if sum(gt[i]<=0):
                    res[i]=-1

            pr_cell = []
            for i in self.thr:
                pr_cell.append(np.sum(res<=i)/len(res))
            pr.append(pr_cell)
            
        pr = np.array(pr)
        pr_val = pr.mean(axis=0)[20]
        return pr_val, pr