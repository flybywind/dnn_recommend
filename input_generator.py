"""
从hdfs读取数据，生成sequence
"""
from subprocess import PIPE, Popen
import shlex
from keras import utils
import traceback
import logging 
_loger = logging.getLogger(__name__)
class HDFSBatchReader(utils.Sequence):
    def __init__(self, path, part_num, 
                 feature_index_map_path,
                 timeout = 1800, 
                 path_fmt = "part-%05d"):
        self.path_base = path
        # batch_size 就等于hdfs的part_num
        self.batch_size = part_num
        self.timeout = timeout
        self.path_fmt = path_fmt
        self.max_feature_num = 0
        self.feature_index_map = {}
        with open(feature_index_map_path) as f:
            for id, line in enumerate(f):
                self.feature_index_map[line.strip()] = id
                self.max_feature_num = id+1
        _loger.info("hdfs path = %s\npart_num = %d, max_feature_num = %d" % (
            self.path_base,
            self.batch_size,
            self.max_feature_num
            ))
                
    def __len__(self):
        return self.batch_size
    
    def _get_feature_id(self, f):
        if f in self.feature_index_map:
            return self.feature_index_map[f]
        else:
            return self.max_feature_num
    def __getitem__(self, index):
        hdfs_part = ("%s/" + self.path_fmt) % (self.path_base, index)
        _loger.debug("get sample from hdfs: " + hdfs_part)
        try:
            args = shlex("hadoop fs -cat " + hdfs_part)
            pp = Popen(args, stdout = PIPE)
            pp.wait(self.timeout)
            batch_mat = []
            targets = []
            for line in pp.stdout:
                seg = line.strip().split()
                if len(seg) < 2:
                    continue
                ids = []
                y = float(seg[0])
                targets.append(y)
                for p in seg[1:]:
                    # 都是one-hot 特征
                    feature = p.split(":")[0]
                    ids.append(self._get_feature_id(feature))
                batch_mat.append(ids)
            pp.stdout.close()
            return batch_mat, targets
        except Exception:
            traceback.print_exc()
                