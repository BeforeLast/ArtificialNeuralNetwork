from numpy import exp
## CONV2D FUNCTION PACK
conv2d_fpack = {
    'relu': lambda x: x * (x > 0),
}

## DENSE FUNCTION PACK
dense_fpack = {
    'relu': lambda x: x * (x > 0),
    'sigmoid': lambda x: 1 / (1 + exp(-x))
}

## MISC FUNCTION PACK
misc = {
    'expected_output_dim_length':
        lambda input, filter,
            padding, stride: (input - filter + 2*padding)//stride + 1,
}