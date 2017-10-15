from keras.layers import Layer
import keras.backend as K


class ReduceAverage(Layer):
    """
    average input tensor from previous layer by axis

    # Arguments
        n: integer, axis position.

    # Input shape:
        (a, b, ..., axis, c, ...)
    # Output shape
        (a, b, ..., c, ...)
    """

    def __init__(self, n, **kwargs):
        super(ReduceAverage, self).__init__(**kwargs)
        if n <= 0:
            raise ValueError("axis should greater than 0, because the first dim is batch_size")
        self.axis = n
    
    def compute_output_shape(self, input_shape):
        ndim = len(input_shape)
        if ndim < 3:
            raise ValueError("input ndim should equal or greater than 2")
        if self.axis >= ndim:
            raise ValueError("reducing axis should less than actual ndim, which is " + str(ndim))
        return (input_shape[0], ) + input_shape[1:self.axis] + input_shape[self.axis+1:]

    def call(self, inputs):
        return K.mean(inputs, self.axis)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(ReduceAverage, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))