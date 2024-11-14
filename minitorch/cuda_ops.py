# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps
from .operators import add, mul, lt, eq, is_close, sigmoid, relu, log, exp, id, inv

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs) -> Fn:
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn, **kwargs) -> FakeCUDAKernel:
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def add_zip(a: Tensor, b: Tensor) -> Tensor:
        return CudaOps.zip(add)(a, b)

    @staticmethod
    def mul_zip(a: Tensor, b: Tensor) -> Tensor:
        return CudaOps.zip(mul)(a, b)

    @staticmethod
    def lt_zip(a: Tensor, b: Tensor) -> Tensor:
        return CudaOps.zip(lt)(a, b)

    @staticmethod
    def eq_zip(a: Tensor, b: Tensor) -> Tensor:
        return CudaOps.zip(eq)(a, b)

    @staticmethod
    def is_close_zip(a: Tensor, b: Tensor) -> Tensor:
        return CudaOps.zip(is_close)(a, b)

    @staticmethod
    def sigmoid_map(a: Tensor) -> Tensor:
        return CudaOps.map(sigmoid)(a)

    @staticmethod
    def relu_map(a: Tensor) -> Tensor:
        return CudaOps.map(relu)(a)

    @staticmethod
    def log_map(a: Tensor) -> Tensor:
        return CudaOps.map(log)(a)

    @staticmethod
    def exp_map(a: Tensor) -> Tensor:
        return CudaOps.map(exp)(a)

    @staticmethod
    def id_map(a: Tensor) -> Tensor:
        return CudaOps.map(id)(a)

    @staticmethod
    def inv_map(a: Tensor) -> Tensor:
        return CudaOps.map(inv)(a)

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        
        # Handle the case where thread index exceeds output size
        if i >= out_size:
            return
        
        # Convert position i to indices for input/output
        to_index(i, out_shape, out_index)
        broadcast_index(out_index, out_shape, in_shape, in_index)
        
        # Calculate positions in storage
        in_position = index_to_position(in_index, in_strides)
        out_position = index_to_position(out_index, out_strides)
        
        # Apply function and store result
        out[out_position] = fn(in_storage[in_position])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        
        # Handle the case where thread index exceeds output size
        if i >= out_size:
            return
        
        # Convert position i to indices
        to_index(i, out_shape, out_index)
        broadcast_index(out_index, out_shape, a_shape, a_index)
        broadcast_index(out_index, out_shape, b_shape, b_index)
        
        # Calculate positions in storage
        a_position = index_to_position(a_index, a_strides)
        b_position = index_to_position(b_index, b_strides)
        out_position = index_to_position(out_index, out_strides)
        
        # Apply function and store result
        out[out_position] = fn(a_storage[a_position], b_storage[b_position])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    BLOCK_DIM = 32
    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x
    
    # Load data into shared memory
    cache[pos] = a[i] if i < size else 0.0
    cuda.syncthreads()
    
    # Reduction in shared memory
    stride = BLOCK_DIM // 2
    while stride > 0:
        if pos < stride and i + stride < size:
            cache[pos] += cache[pos + stride]
        cuda.syncthreads()
        stride //= 2
    
    # Write result to output
    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        # Reduce block size to avoid resource exhaustion
        BLOCK_DIM = 256  # Reduced from 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x

        if out_pos >= out_size:
            return

        to_index(out_pos, out_shape, out_index)
        cache[pos] = reduce_value

        # Process in smaller chunks
        chunk_size = min(BLOCK_DIM, a_shape[reduce_dim])
        for j in range(pos, a_shape[reduce_dim], chunk_size):
            if j < a_shape[reduce_dim]:
                in_index = cuda.local.array(MAX_DIMS, numba.int32)
                for i in range(len(out_shape)):
                    in_index[i] = out_index[i]
                in_index[reduce_dim] = j
                
                in_pos = index_to_position(in_index, a_strides)
                cache[pos] = fn(cache[pos], a_storage[in_pos])
        
        cuda.syncthreads()
        
        # Reduce within block with smaller stride
        stride = BLOCK_DIM // 2
        while stride > 0:
            if pos < stride:
                cache[pos] = fn(cache[pos], cache[pos + stride])
            cuda.syncthreads()
            stride //= 2
        
        if pos == 0:
            out_pos = index_to_position(out_index, out_strides)
            out[out_pos] = cache[0]

    return cuda.jit()(_reduce)


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """Practice square MM kernel for matrix multiplication."""
    BLOCK_DIM = 32
    
    # Create shared memory for tiles of a and b
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    
    # Get thread indices
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    
    # Calculate output position
    row = cuda.blockIdx.x * cuda.blockDim.x + tx
    col = cuda.blockIdx.y * cuda.blockDim.y + ty
    
    # Initialize output value
    tmp = 0.0
    
    # Load data into shared memory and compute
    if row < size and col < size:
        # Matrix multiplication with shared memory
        for i in range(0, size, BLOCK_DIM):
            # Load tiles into shared memory
            if i + tx < size and col < size:
                a_shared[ty, tx] = a[row * size + (i + tx)]
            if i + ty < size and row < size:
                b_shared[ty, tx] = b[(i + ty) * size + col]
            
            # Synchronize threads
            cuda.syncthreads()
            
            # Compute partial dot product
            for k in range(min(BLOCK_DIM, size - i)):
                tmp += a_shared[ty, k] * b_shared[k, tx]
            
            # Synchronize before next iteration
            cuda.syncthreads()
        
        # Write result to global memory
        out[row * size + col] = tmp


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function."""
    BLOCK_DIM = 32
    
    # Create shared memory for tiles
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    
    # Get thread indices
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bz = cuda.blockIdx.z
    
    # Calculate batch stride
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    
    # Calculate matrix dimensions
    batch = bz
    row = bx * BLOCK_DIM + tx
    col = by * BLOCK_DIM + ty
    
    # Initialize accumulator
    acc = 0.0
    
    # Loop over tiles
    for tile in range(0, a_shape[-1], BLOCK_DIM):
        # Load data into shared memory
        if row < out_shape[-2] and tile + ty < a_shape[-1]:
            a_pos = batch * a_batch_stride + row * a_strides[-2] + (tile + ty) * a_strides[-1]
            a_shared[tx, ty] = a_storage[a_pos]
        else:
            a_shared[tx, ty] = 0.0
            
        if col < out_shape[-1] and tile + tx < b_shape[-2]:
            b_pos = batch * b_batch_stride + (tile + tx) * b_strides[-2] + col * b_strides[-1]
            b_shared[tx, ty] = b_storage[b_pos]
        else:
            b_shared[tx, ty] = 0.0
            
        cuda.syncthreads()
        
        # Compute partial dot product
        if row < out_shape[-2] and col < out_shape[-1]:
            for k in range(min(BLOCK_DIM, a_shape[-1] - tile)):
                acc += a_shared[tx, k] * b_shared[k, ty]
                
        cuda.syncthreads()
    
    # Write result to global memory
    if row < out_shape[-2] and col < out_shape[-1]:
        out_pos = (batch * out_strides[0] + 
                  row * out_strides[-2] + 
                  col * out_strides[-1])
        out[out_pos] = acc


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
