from keras import backend as K
from keras.layers import Input 
import numpy as np

def lookup_embedding(ids, emb_layer):
    '''
    ids: list of id
    '''
    n = len(ids)
    input_placehoder = Input((1,))
    emb = emb_layer(input_placehoder)
    f = K.function([input], [emb])
    ids_t = np.transpose(
            np.reshape(np.array(ids),
                       (1, n)))
    out_mat = f([ids_t]) # shape = (n, 1, emb_dim)
    out_emb = []
    for i in range(n):
        out_emb.append(out_mat[i][0][:])
    return out_emb
    