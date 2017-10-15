# -*- encoding: utf8 -*- 
from keras.engine import Model
from keras import backend as K
# from keras.models import Sequential
from keras.layers import Dense, Input, Embedding, Concatenate
from ReduceAverage import ReduceAverage

def build_dnn(each_inputs_max_dim, item_max_dim, emb_dim, dense_layers):
    '''
    Params:
        each_inputs_max_dim: 每个种类单独embedding，都有一个单独的最大id数
        item_max_dim:        即视频id的max id数 
    '''
    inputs = []
    cancatenate_out = []
    for max_dim in each_inputs_max_dim:
        var_in = Input(shape = (None,))
        inputs.append(var_in)
        emb = Embedding(max_dim, emb_dim)(var_in)
        avg = ReduceAverage(0)(emb)
        cancatenate_out.append(avg)
    cancatenate_layer = Concatenate()(cancatenate_out)
    last_hidden_layer = cancatenate_layer
    for l in dense_layers[:-1]:
        hidden_layer = Dense(l, activation="relu")(last_hidden_layer)
        last_hidden_layer = hidden_layer
    user_layer = Dense(dense_layers[-1], activation="relu")(last_hidden_layer)
    item_input = Input(shape = (1,))
    inputs.append(item_input)
    item_target = Embedding(item_max_dim, emb_dim)(item_input)
    out = K.sigmoid(K.dot(user_layer, item_target))
    model = Model(inputs = inputs, output = out)
    return model, user_layer