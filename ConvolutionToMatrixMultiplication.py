import numpy as np
from scipy.linalg import toeplitz, block_diag, pinv
from scipy.sparse import coo_matrix, bmat
from tables.idxutils import col_light
from openpyxl.styles.builtins import output
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

#convolution matrix of input with multible batches and channels with some kernel which is equivalent to convolution of input with kernel
def convolution_matrix_block(input_shape, kernel):
    
    # shape of the kernel, i.e.  number of kernels (output_channels), input_channels, number of rows and columns
    kernel_shape = kernel.shape
    
    #2d convolution with arbitrary channels
    try:
        kernel_num, kernel_channel_num, kernel_row_num, kernel_col_num = kernel_shape
    except ValueError:
        print("Shape of kernel does not fit the settings")
    

    #shape of the input size of the batch, input_channel number and the input size 
    try:
        batch_size, channel_num, row_num, column_num = input_shape
    except ValueError:
        print("Shape of input does not fit the settings")

    kernel = np.pad(kernel,  ((0, 0), (0, 0), (0, 0), (0, column_num-kernel_col_num)),
                                'constant', constant_values=0)
    
    
    #kernel channel and image channel does not fit
    try:
         channel_num == kernel_channel_num
    except:
        print("Input channels and kernel channels do not fit toghether")
    
    #convolution matrix 
    block_list = []
    for num in range(0, kernel_num): #number of different kernels, i.e. the output channel dimension
        row_list = []
        for channel in range(0, channel_num): #number of different channels, i.e. the input channel dimension
            #generate the toeplitz matrices for the convolution
            T = np.zeros((row_num-kernel_row_num+1, row_num, column_num-kernel_col_num+1, column_num))
            toeplitz_blocks = []
            for row_block_num in range(0, row_num-kernel_row_num+1):
                toeplitz_blocks_rows = []
                for col_block_num in range(0, row_num):
                    if col_block_num - row_block_num >=0 and col_block_num - row_block_num < kernel_row_num:
                        toeplitz_col = np.zeros((column_num-kernel_col_num + 1))
                        toeplitz_col[0] = kernel[num][channel][col_block_num-row_block_num][0]
                        T[row_block_num][col_block_num] = toeplitz(toeplitz_col, kernel[num][channel][col_block_num-row_block_num])
                    toeplitz_blocks_rows.append(T[row_block_num][col_block_num])
                toeplitz_blocks.append(toeplitz_blocks_rows)
            row_list.append(np.block(toeplitz_blocks))    
            #set toeplitz block in convolution matrix
        block_list.append(row_list)
    conv_matrix= np.block(block_list)
    return conv_matrix


def convolution(input, kernel, padding = None):
    # shape of the kernel, i.e.  number of kernels (output_channels), input_channels, number of rows and columns
    kernel_shape = kernel.shape
    
    #2d convolution with arbitrary channels
    if len(kernel_shape) == 4:
        kernel_num, channel_num, kernel_row_num, kernel_col_num = kernel_shape


    if padding is not None:
        input = np.pad(input, ((0, 0), (0, 0), (kernel_row_num-1, kernel_row_num-1), (kernel_col_num-1, kernel_col_num-1)), 'constant', constant_values=padding)
    # number of columns and rows of the input 
    input_shape = input.shape

    #shape of the input size of the batch, input_channel number and the input size 
    batch_size, channel_num, row_num, column_num = input_shape

    #calculate the result and reshape the input
    #modify input if convolution with padding should be applied, i.e. add zero padding (possibly other constant values can be performed)
    conv_matrix = convolution_matrix_block(input_shape, kernel)
    return np.reshape(np.dot(block_diag(*[conv_matrix for i in range(0, batch_size)]), input.flatten()), (-1, kernel_num, row_num-kernel_row_num+1, column_num - kernel_col_num + 1))
 

def inverse_convolution(input, kernel, padding = None, output_shape = None):
    inverse_blocks = []
    if output_shape is None and padding is None:
        output_shape = (input.shape[0], kernel.shape[1], input.shape[2] + kernel.shape[2] - 1, input.shape[3] + kernel.shape[3] - 1)
    elif output_shape is None:
        output_shape = (input.shape[0], kernel.shape[1], input.shape[2] + kernel.shape[2] - 1, input.shape[3] + kernel.shape[3] - 1)  
    conv_matrix = convolution_matrix_block(output_shape, kernel)
    inverse_matrix = np.linalg.pinv(conv_matrix, rcond=1e-15)
    print("ConvMatrixDone")
    return np.dot(block_diag(*[inverse_matrix for i in range(0, input.shape[0])]), input.flatten()).reshape(output_shape)
        



def unpooling2d(input, kernel_shape, output_shape = None, algo = None):
    
    #get kernel shape
    kernel_row_num, kernel_col_num = kernel_shape
    
    #get input shape
    batch_size, channel_num, row_num, column_num = input.shape

    if output_shape == None:
        output_shape = (batch_size, channel_num, row_num*kernel_row_num, column_num*kernel_col_num)
    else:
        output_shape = output_shape
    
    output = np.zeros(output_shape)
    
    for batch in range(0, batch_size):
        for channel in range(0, channel_num):
            for row in range(0, output_shape[2]):
                for column in range(0, output_shape[3]):
                    if row//kernel_row_num < len(input[batch][channel]) and column//kernel_col_num < len(input[batch][channel][len(input[batch][channel]) - 1]):
                        value = input[batch][channel][row//kernel_row_num][column//kernel_col_num]
                        output[batch][channel][row][column] = value
                        """
                        if row % kernel_row_num == 0 and column % kernel_col_num == 0: 
                            output[batch][channel][row][column] = value
                        else:
                            output[batch][channel][row][column] = value - random.uniform(0, abs(value))
                        """
    
    return output

def main():
    
    test_kernel = np.random.rand(2, 1, 3, 3)
    test_kernel = np.array([[[[1, 1], [1, 1]]], 
                            [[[1, 1], [1, 0]]], 
                            [[[1, 1], [0, 0]]], 
                            [[[1, 0], [0, 0]]]])
    #y = np.array([[[[1, -1], [2, 1]]]])
    y = np.random.rand(1, 2, 1, 2)
    y = np.array([[[[1, 1]], [[2, 2]], [[4, 4]], [[2, 2]]]])
    #y_unpool = unpooling2d(y, (2, 2))
    
    y_uncolvolved = inverse_convolution(y, test_kernel)
    y_convolved = convolution(y_uncolvolved, test_kernel)
    conv_mat = convolution_matrix_block(y_uncolvolved.shape, test_kernel)
    print("Convolution Matrix", conv_mat)
    print("Pseudo inverse", pinv(conv_mat))
    #print("Unpooled", y_unpool)
    print("Unconvolved", y_uncolvolved)
    print("Reconvolved", y_convolved)
    print("Should be zero", y - y_convolved)
    print(F.max_pool2d(torch.tensor(y_convolved).double(), 2))
    
    """
    #x = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]])
    x = np.random.rand(2, 20, 12, 12)
    #kernel = np.array([[[[1, 1, 1],[1, 1, 1]]]])
    kernel = np.random.rand(50, 20, 5, 5)
    
    conv_pytorch = nn.Conv2d(20, 50, 5, bias = False)
    conv_pytorch.weight.data = torch.tensor(kernel)
    
    
    print("Input", x)
    print("Kernel", kernel)
    
    
    #conv_mat = block_diag(*convolution_matrix_blocks(x.shape, kernel))
    #pseudo_inv = np.linalg.pinv(conv_mat)
    #print(pseudo_inv.shape)
    result = convolution(x, kernel, padding = None)
    print("Result", result)
    print("Should be zero", torch.tensor(result) - conv_pytorch(torch.tensor(x)))
    
    
    #print(inverse_convolution(result, kernel, padding = None))
    #result2 = convolution(inverse_convolution(result, kernel, padding = None), kernel, padding = None)
    
    result2 = conv_pytorch(torch.tensor(inverse_convolution(result, kernel, padding = None)))
    
    print("Inverse Convolution", result2)
    print("Should be zero", torch.tensor(result) - torch.tensor(result2))
    """
if __name__ == '__main__':
    main()
