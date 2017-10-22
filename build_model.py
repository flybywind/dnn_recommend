# -*- encoding: utf8 -*- 
from keras.engine import Model
# from keras.models import Sequential
from keras.layers import Activation, Dense, Input, Embedding, Concatenate, Dot
from ReduceAverage import ReduceAverage
from keras import optimizers
from keras import losses

def build_dnn(each_inputs_max_dim, emb_dim, item_name, dense_layers):
    '''
    Params:
        each_inputs_max_dim: dict, 每个种类单独embedding，都有一个单独的最大id数
        item_name:          即视频id的名字，应该也存在于each_inputs_max_dim中 
    '''
    inputs = []
    cancatenate_out = []
    item_layer = None
    for grp, max_dim in each_inputs_max_dim.iteritems():
        var_in = Input(shape = (None,), name = grp)
        inputs.append(var_in)
        emb_layer = Embedding(max_dim, emb_dim)
        if grp == item_name:
            item_layer = emb_layer
        emb = emb_layer(var_in)
        avg = ReduceAverage(1)(emb)
        cancatenate_out.append(avg)
    cancatenate_layer = Concatenate()(cancatenate_out)
    last_hidden_layer = cancatenate_layer
    for l in dense_layers[:-1]:
        hidden_layer = Dense(l, activation="relu")(last_hidden_layer)
        last_hidden_layer = hidden_layer
    user_layer = Dense(dense_layers[-1], activation="relu")
    user_emb = user_layer(last_hidden_layer)
    item_input = Input(shape = (1,), name = item_name + "-0")
    inputs.append(item_input)
    item_emb = item_layer(item_input)
    out = Activation(activation = "sigmoid")(
            Dot(1)([user_emb, item_emb]))
    model = Model(inputs, out)
    ada_grad = optimizers.Adagrad()
    model.compile(optimizer=ada_grad, loss = losses.binary_crossentropy)
    return model, user_layer, item_layer