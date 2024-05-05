# Backends for tensor operations
#--------------------------------
"""
    abstract type AbstractBackend
    
Abstract supertype for all backends that can be used for tensor operations. In particular,
these control different implementations of executing the basic operations.
"""
abstract type AbstractBackend end

"""
    select_backend([tensorfun::Function], tensors...) -> AbstractBackend

Select the default backend for the given tensors or tensortypes. If `tensorfun` is provided,
it is possible to more finely control the backend selection based on the function as well.
"""
select_backend(tensorfun::Function, tensors...) = select_backend(tensors...)

# Strided backends
#-----------------
"""
    StridedNative()

Backend for tensor operations that is based on `StridedView` objects with native Julia
implementations of tensor operations.
"""
struct StridedNative <: AbstractBackend end

"""
    StridedBLAS()

Backend for tensor operations that is based on using `StridedView` objects and rephrasing
the tensor operations as BLAS operations.
"""
struct StridedBLAS <: AbstractBackend end

const StridedBackends = Union{StridedNative,StridedBLAS}
