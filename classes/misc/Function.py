import numpy as np
## CONV2D FUNCTION PACK
conv2d_fpack = {
    'relu': lambda x: np.maximum(0, x),
}
conv2d_fpack_deriv = {
    'relu': lambda x: (x > 0).astype(int),
}

## CONV2D ERROR PACK
conv2d_epack = {
    'relu': lambda y, t: np.square(t - y),
}
conv2d_epack_deriv = {
    'relu': lambda y, t: (t - y),
}

## DENSE FUNCTION PACK
dense_fpack = {
    'relu': lambda x: np.maximum(0, x),
    'sigmoid': lambda x: 1 / (1 + np.exp(-x))
}
dense_fpack_deriv = {
    'relu': lambda x: (x > 0).astype(int),
    'sigmoid': lambda x: dense_fpack['sigmoid'](x)*(1-dense_fpack['sigmoid'](x))
}

## DENSE ERROR PACK
dense_epack = {
    'relu': lambda y, t: np.square(t - y),
    'sigmoid': lambda y, t: np.square(t - y)
}
dense_epack_deriv = {
    'relu': lambda y, t: (t - y),
    'sigmoid': lambda y, t: (t - y)
}

## MISC FUNCTION PACK
misc = {
    'expected_output_dim_length':
        lambda input, filter,
            padding, stride: (input - filter + 2*padding)//stride + 1,
}