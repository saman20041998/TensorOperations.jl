# ------------------------------------------------------------------------------------------
# Generic implementation
# ------------------------------------------------------------------------------------------

tensorop(args...) = +(*(args...), *(args...))
"""
    promote_contract(args...)

Promote the scalar types of a tensor contraction to a common type.
"""
promote_contract(args...) = Base.promote_op(tensorop, args...)

"""
    promote_add(args...)

Promote the scalar types of a tensor addition to a common type.
"""
promote_add(args...) = Base.promote_op(+, args...)

"""
    tensoralloc_add(TC, pC, A, conjA, istemp=false, backend::AbstractBackend...)

Allocate a tensor `C` of scalar type `TC` that would be the result of

    `tensoradd!(C, pC, A, conjA)`

The `istemp` argument is used to indicate that a tensor wlil not be used after the `@tensor`
block, and thus will be followed by an explicit call to `tensorfree!`. The `backend` can be
used to implement different allocation strategies.

See also [`tensoralloc`](@ref) and [`tensorfree!`](@ref).
"""
function tensoralloc_add(TC, pC, A, conjA, istemp=false, backend::AbstractBackend...)
    ttype = tensoradd_type(TC, pC, A, conjA)
    structure = tensoradd_structure(pC, A, conjA)
    return tensoralloc(ttype, structure, istemp, backend...)::ttype
end

"""
    tensoralloc_contract(TC, pC, A, pA, conjA, B, pB, conjB, istemp=false, backend::AbstractBackend...)

Allocate a tensor `C` of scalar type `TC` that would be the result of

    `tensorcontract!(C, pC, A, pA, conjA, B, pB, conjB)`

The `istemp` argument is used to indicate that a tensor wlil not be used after the `@tensor`
block, and thus will be followed by an explicit call to `tensorfree!`. The `backend` can be
used to implement different allocation strategies.

See also [`tensoralloc`](@ref) and [`tensorfree!`](@ref).
"""
function tensoralloc_contract(TC, pC, A, pA, conjA, B, pB, conjB,
                              istemp=false, backend::AbstractBackend...)
    ttype = tensorcontract_type(TC, pC, A, pA, conjA, B, pB, conjB)
    structure = tensorcontract_structure(pC, A, pA, conjA, B, pB, conjB)
    return tensoralloc(ttype, structure, istemp, backend...)::ttype
end

# ------------------------------------------------------------------------------------------
# AbstractArray implementation
# ------------------------------------------------------------------------------------------

tensorstructure(A::AbstractArray) = size(A)
tensorstructure(A::AbstractArray, iA::Int, conjA::Symbol) = size(A, iA)

function tensoradd_type(TC, pC::Index2Tuple, A::AbstractArray, conjA::Symbol)
    return Array{TC,sum(length.(pC))}
end

function tensoradd_structure(pC::Index2Tuple, A::AbstractArray, conjA::Symbol)
    return size.(Ref(A), linearize(pC))
end

function tensorcontract_type(TC, pC, A::AbstractArray, pA, conjA,
                             B::AbstractArray, pB, conjB, backend...)
    return Array{TC,sum(length.(pC))}
end

function tensorcontract_structure(pC::Index2Tuple,
                                  A::AbstractArray, pA::Index2Tuple, conjA,
                                  B::AbstractArray, pB::Index2Tuple, conjB)
    return let lA = length(pA[1])
        map(n -> n <= lA ? size(A, pA[1][n]) : size(B, pB[2][n - lA]), linearize(pC))
    end
end

function tensoralloc(ttype, structure, istemp=false, backend::AbstractBackend...)
    C = ttype(undef, structure)
    # fix an issue with undefined references for strided arrays
    if !isbitstype(scalartype(ttype))
        C = zerovector!!(C)
    end
    return C
end

tensorfree!(C, backend::AbstractBackend...) = nothing
