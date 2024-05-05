# Backends for tensor operations
#--------------------------------
"""
    abstract type AbstractBackend
    
Abstract supertype for all backends that can be used for tensor operations. In particular,
these control different implementations of executing the basic operations.
"""
abstract type AbstractBackend end

"""
    select_backend([tensorfun], tensors...) -> AbstractBackend

Select the default backend for the given tensors or tensortypes. If `tensorfun` is provided,
it is possible to more finely control the backend selection based on the function as well.
"""
select_backend(tensorfun, tensors...) = select_backend(tensors...)
