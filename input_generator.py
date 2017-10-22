"""
从hdfs读取数据，生成sequence
"""
from subprocess import PIPE, Popen
from collections import Counter,defaultdict
import shlex
from keras import utils
import traceback
import logging 
import numpy as np
_loger = logging.getLogger(__name__)
class HDFSBatchReader(utils.Sequence):
    def __init__(self, path, part_num, 
                 feature_index_map_path,
                 timeout = 1800, 
                 path_fmt = "part-%05d"):
        '''
        path: 输入的hdfs路径，下面的part以 path_fmt `part-00000`这种格式命名
        其中，第一列必须是 1:vid#2017091712345 这种形式，1表示target，冒号后面
        表示要预测的下一个视频。如果target == 1表示用户对这个视频兴趣很大（比如播放
        完成率大于50%，或者30s等等评定方法），如果target == 0表示用户不感兴趣
        '''
        # todo: Add weights names
        self.path_base = path
        # batch_size 就等于hdfs的part_num
        self.batch_size = part_num
        self.timeout = timeout
        self.path_fmt = path_fmt
        self.max_feature_num = 0
        self.feature_index_map = {}
        counter = Counter()
        with open(feature_index_map_path) as f:
            for line in enumerate(f):
                f = line.strip()
                grp = self._get_feature_group(f)
                self.feature_index_map[f] = counter[grp]
                counter[grp] += 1
            self.max_feature_num = counter
        _loger.info("hdfs path = %s\npart_num = %d, max_feature_num = %d" % (
            self.path_base,
            self.batch_size,
            self.max_feature_num
            ))
                
    def __len__(self):
        return self.batch_size
    
    def _get_feature_group(self, p):
        ret = p.split(":")[0]
        sep = ":"
        if ret.index("_") >= 0:
            sep = "_"
        elif ret.inde("#") :
            sep = "#"
        return p.split(sep)[0]
    
    def _get_feature_id(self, f):
        if f in self.feature_index_map:
            return self.feature_index_map[f]
        else:
            return self.max_feature_num[self._get_feature_group(f)]
        
    def __getitem__(self, index):
        hdfs_part = ("%s/" + self.path_fmt) % (self.path_base, index)
        _loger.debug("get sample from hdfs:  " + hdfs_part)
        try:
            args = shlex("hadoop fs -cat " + hdfs_part)
            pp = Popen(args, stdout = PIPE)
            pp.wait(self.timeout)
            batch_inputs = []
            targets = []
            for line in pp.stdout:
                seg = line.strip().split()
                if len(seg) < 2:
                    continue
                ids = defaultdict(list)
                p = seg[0].split(":")
                y = float(p[0])
                next_item = p[1] # 下一个将观看的视频(item)
                targets.append(y)
                for p in seg[1:]:
                    # 都是one-hot 特征
                    feature = p.split(":")[0]
                    grp = self._get_feature_group(feature)
                    ids[grp].append(self._get_feature_id(feature))
                inputs = {}
                for grp in ids.keys():
                    inputs[grp] = np.asarray(ids[grp], dtype='int64')
                feature = next_item 
                grp = self._get_feature_group(feature)
                ids[grp+"-0"].append(self._get_feature_id(feature))
                batch_inputs.append(inputs)
            pp.stdout.close()
            return batch_inputs, targets
        except Exception:
            traceback.print_exc()
                