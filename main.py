# -*- encoding: utf8 -*- 
from keras import callbacks
from build_model import build_dnn
from input_generator import HDFSBatchReader 
import argparse
import logging 

if __name__ == "__main__":
    _loger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser('''train recomendation system in dnn''')
    parser.add_argument('-f', '--feature_list', default = "",
                        metavar = 'feature_list path, spark计算的结果，已经去掉了低频id')
    parser.add_argument('-d', '--dim', default = 256, type = int,
                        metavar = 'embedding dim')
    parser.add_argument('-p', '--hdfs_path', default = "",
                        metavar = "hdfs base path")
    parser.add_argument("-t", '--parallel_num', default = 16, type = int,
                        metavar = "并行运行的进程数量")
    parser.add_argument('-pn', '--part_num', default=1000, type = int,
                        metavar = "hdfs part num in base_path, 也等于batch_num")
    parser.add_argument('-ig',  '--item_group', default='vid',
                        metavar = 'item name所在的组名')
    parser.add_argument('-ln', '--each_layer_num', default = '512,256',
                        metavar = "each hidden layer number")
    parser.add_argument("-lg", '--log_dir', default = "./log",
                        metavar = "log dir path")
    parser.add_argument("-e", "--epoch", default = 1, type = int)
    parser.add_argument("-m", '--model_path', default = "./models/dnn.model",
                        metavar = "saved model path")
    parser.add_argument("-vp", "--valid_hdfs_path", default = "")
    args = parser.parse_args()
    seq_generator = HDFSBatchReader(path=args.hdfs_path,
                    part_num = args.part_num,
                    feature_index_map_path = args.feature_list)
    kw = {}
    if len(args.valid_hdfs_path) > 0:
        valid_generator = HDFSBatchReader(path=args.valid_hdfs_path,
                    part_num = args.part_num,
                    feature_index_map_path = args.feature_list)
        kw["validation_data": valid_generator,
           "validation_steps": args.part_num]
        
    model, user_layer, item_layer = build_dnn(
        seq_generator.feature_index_map,
        args.dim,
        args.item_group,
        [int(x) for x in args.each_layer_num.split(",")])
    tensor_board_cbk = callbacks.TensorBoard(log_dir=args.log_dir)
    model.fit_generator(seq_generator,
                        args.part_num,
                        epochs = args.epoch,
                        workers = args.parallel_num,
                        callbacks = [callbacks.TerminateOnNaN(),
                                     callbacks.ProgbarLogger(),
                                     tensor_board_cbk,
                                     callbacks.ModelCheckpoint(args.model_path)],
                        use_multiprocessing = True,
                        verbose = 2,
                        **kw)
    
                    
    