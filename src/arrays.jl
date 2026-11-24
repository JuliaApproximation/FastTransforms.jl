struct ArrayPlan{T, FF<:FTPlan{<:T}, Szs<:Tuple, Dims<:Tuple{<:Int}} <: Plan{T}
    F::FF
    szs::Szs
    dims::Dims
end
size(P::ArrayPlan) = P.szs

function ArrayPlan(F::FTPlan{<:T}, c::AbstractArray{T}, dims::Tuple{<:Int}=(1,)) where T
    szs = size(c)
    @assert F.n == szs[dims[1]]
    ArrayPlan(F, size(c), dims)
end

function *(P::ArrayPlan, f::AbstractArray)
    F, dims, szs = P.F, P.dims, P.szs
    @assert length(dims) == 1
    @assert szs == size(f)
    d = first(dims)

    perm = (d, ntuple(i-> i + (i >= d), ndims(f) -1)...)
    fp = permutedims(f, perm)

    fr = reshape(fp, size(fp,1), :)

    permutedims(reshape(F*fr, size(fp)...), invperm(perm))
end

function \(P::ArrayPlan, f::AbstractArray)
    F, dims, szs = P.F, P.dims, P.szs
    @assert length(dims) == 1
    @assert szs == size(f)
    d = first(dims)

    perm = (d, ntuple(i-> i + (i >= d), ndims(f) -1)...)
    fp = permutedims(f, perm)

    fr = reshape(fp, size(fp,1), :)

    permutedims(reshape(F\fr, size(fp)...), invperm(perm))
end

"""
    NDimsPlan(F, szs, dims)

Plan applying the transform `F` along several dimensions of an array of size `szs`.

If `dims` is a tuple of integers, `F` is a vector transform applied along each of the
dimensions in `dims`, all of which must have the same length.

If `dims` is a tuple of tuples, each entry lists the dimensions spanned by the slices `F`
acts on. For a matrix transform the entry `(i, j)` applies `F` to every matrix slice whose
rows run along dimension `i` and columns along dimension `j`, so for a 4-array `A`,
`dims = ((1,3), (2,4))` first applies `F` to `A[:,k,:,l]` for every `k` and `l`, and then
to `A[k,:,l,:]`. Each slice is copied into a buffer stored in the plan, transformed in place
with `lmul!`/`ldiv!`, and copied back, so `F` only needs to support in-place application to
an `Array` of the slice size. All slices must have the same size. Since the buffer is shared,
a plan must not be applied from several tasks at once.
"""
struct NDimsPlan{T, FF, Szs<:Tuple, Dims<:Tuple, Buf} <: Plan{T}
    F::FF
    szs::Szs
    dims::Dims
    buffer::Buf
    function NDimsPlan(F::ArrayPlan, szs::Tuple, dims::Tuple{Int,Vararg{Int}})
        if length(Set(szs[[dims...]])) > 1
            error("Different size in dims axes not yet implemented in N-dimensional transform.")
        end
        new{eltype(F), typeof(F), typeof(szs), typeof(dims), Nothing}(F, szs, dims, nothing)
    end
    function NDimsPlan(F, szs::Tuple, dims::Tuple{Tuple{Vararg{Int}},Vararg{Tuple{Vararg{Int}}}})
        _checkslicedims(szs, dims)
        slicesize = map(i -> szs[i], first(dims))
        if any(d -> map(i -> szs[i], d) != slicesize, dims)
            error("Different slice sizes in dims not yet implemented in N-dimensional transform.")
        end
        buffer = Array{eltype(F)}(undef, slicesize)
        new{eltype(F), typeof(F), typeof(szs), typeof(dims), typeof(buffer)}(F, szs, dims, buffer)
    end
end

# NDimsPlan of a vector transform along integer dims, implemented via ArrayPlan
const VectorNDimsPlan = NDimsPlan{<:Any, <:ArrayPlan, <:Any, <:Any, Nothing}
# NDimsPlan applying a transform to slices spanned by tuple dims, implemented via a buffer
const SliceNDimsPlan{T} = NDimsPlan{T, <:Any, <:Any, <:Any, <:AbstractArray}

size(P::NDimsPlan) = P.szs

function show(io::IO, P::NDimsPlan)
    print(io, "NDimsPlan for ", join(P.szs, "×"), "-element array along dims ", P.dims, " of ")
    show(io, P.F)
end

function NDimsPlan(F::FTPlan, szs::Tuple, dims::Tuple{Int,Vararg{Int}})
    NDimsPlan(ArrayPlan(F, szs, (first(dims),)), szs, dims)
end

function _checkslicedims(szs::Tuple, dims::Tuple)
    N = length(szs)
    for d in dims
        isempty(d) && throw(ArgumentError("dims entries must be non-empty"))
        all(i -> 1 ≤ i ≤ N, d) || throw(ArgumentError("dims entry $d is out of range for a $N-dimensional array"))
        allunique(d) || throw(ArgumentError("dims entry $d repeats a dimension"))
    end
end

function *(P::VectorNDimsPlan, f::AbstractArray)
    F, dims = P.F, P.dims
    @assert size(P) == size(f)
    g = copy(f)
    t = 1:ndims(g)
    d1 = dims[1]
    for d in dims
        perm = ntuple(k -> k == d1 ? t[d] : k == d ? t[d1] : t[k], ndims(g))
        gp = permutedims(g, perm)
        g = permutedims(F*gp, invperm(perm))
    end
    return g
end

function \(P::VectorNDimsPlan, f::AbstractArray)
    F, dims = P.F, P.dims
    @assert size(P) == size(f)
    g = copy(f)
    t = 1:ndims(g)
    d1 = dims[1]
    for d in dims
        perm = ntuple(k -> k == d1 ? t[d] : k == d ? t[d1] : t[k], ndims(g))
        gp = permutedims(g, perm)
        g = permutedims(F\gp, invperm(perm))
    end
    return g
end

# Apply f!(F, buffer) in place to every slice of x spanned by the dimensions d.
function _slices!(f!, F, buffer::AbstractArray{<:Any,M}, x::AbstractArray{<:Any,N}, d::NTuple{M,Int}) where {M,N}
    perm = (d..., filter(∉(d), ntuple(identity, Val(N)))...)
    _permutedslices!(f!, F, buffer, PermutedDimsArray(x, perm))
    return x
end

# The slices of xp are xp[:, …, :, I] with M colons; this function barrier makes the loop type-stable.
function _permutedslices!(f!, F, buffer::AbstractArray{<:Any,M}, xp::AbstractArray{<:Any,N}) where {M,N}
    colons = ntuple(_ -> Colon(), Val(M))
    for I in CartesianIndices(ntuple(k -> axes(xp, M+k), Val(N-M)))
        v = view(xp, colons..., I)
        copyto!(buffer, v)
        f!(F, buffer)
        copyto!(v, buffer)
    end
    return xp
end

function _checksize(P::NDimsPlan, x::AbstractArray)
    if size(x) != size(P)
        throw(DimensionMismatch("NDimsPlan is for arrays of size $(size(P)), x has size $(size(x))"))
    end
end

function lmul!(P::SliceNDimsPlan{T}, x::AbstractArray{T}) where T
    _checksize(P, x)
    for d in P.dims
        _slices!(lmul!, P.F, P.buffer, x, d)
    end
    return x
end

function ldiv!(P::SliceNDimsPlan{T}, x::AbstractArray{T}) where T
    _checksize(P, x)
    for d in reverse(P.dims)
        _slices!(ldiv!, P.F, P.buffer, x, d)
    end
    return x
end

for fJ in (:lmul!, :ldiv!)
    @eval function $fJ(P::SliceNDimsPlan{T}, x::AbstractArray{Complex{T}}) where T<:Real
        x .= complex.($fJ(P, real(x)), $fJ(P, imag(x)))
        return x
    end
end

*(P::SliceNDimsPlan{T}, x::AbstractArray) where T = lmul!(P, Array{promote_type(T, eltype(x))}(x))
\(P::SliceNDimsPlan{T}, x::AbstractArray) where T = ldiv!(P, Array{promote_type(T, eltype(x))}(x))

# the inverse/adjoint/transpose of a composition applies the factors in reverse order
inv(P::SliceNDimsPlan) = NDimsPlan(inv(P.F), P.szs, reverse(P.dims))
adjoint(P::SliceNDimsPlan) = NDimsPlan(adjoint(P.F), P.szs, reverse(P.dims))
transpose(P::SliceNDimsPlan) = NDimsPlan(transpose(P.F), P.szs, reverse(P.dims))


# Transforms acting on matrices: dims is a non-empty tuple of pairs, e.g. ((1,3),) or ((1,3),(2,4)).
# Returns the size of the matrix slices.
function _matrixslicesize(A::AbstractArray, dims::Tuple)
    if !(dims isa Tuple{NTuple{2,Int},Vararg{NTuple{2,Int}}})
        throw(ArgumentError("dims must be a tuple of pairs of dimensions, e.g. ((1,3),) or ((1,3),(2,4)), got $dims"))
    end
    _checkslicedims(size(A), dims)
    i, j = first(dims)
    return (size(A, i), size(A, j))
end

"""
    plan_disk2cxf(A, [α, β,] dims)

Plan the Zernike--Chebyshev×Fourier transform on the matrix slices of `A` spanned by each
pair of dimensions in `dims`, applied in order. For example, `dims = ((1,3),)` transforms
`A[:,k,:]` for every `k`, and `dims = ((1,3), (2,4))` transforms `A[:,k,:,l]` for every `k`
and `l`, then `A[k,:,l,:]`. All slices must have the same size. See [`NDimsPlan`](@ref).
"""
plan_disk2cxf(A::AbstractArray, dims::Tuple) = plan_disk2cxf(A, 0, 0, dims)
plan_disk2cxf(A::AbstractArray, α, dims::Tuple) = plan_disk2cxf(A, α, 0, dims)
function plan_disk2cxf(A::AbstractArray{T}, α, β, dims::Tuple) where T
    n, _ = _matrixslicesize(A, dims)
    NDimsPlan(plan_disk2cxf(T, n, α, β), size(A), dims)
end

for (plan_f, kind) in ((:plan_disk_analysis, "analysis"), (:plan_disk_synthesis, "synthesis"))
    doc = """
        $plan_f(A, dims; flags=FFTW.ESTIMATE)

    Plan the FFTW Chebyshev×Fourier $kind on the disk on the matrix slices of `A` spanned by
    each pair of dimensions in `dims`, applied in order. For example, `dims = ((1,3),)` acts on
    `A[:,k,:]` for every `k`, and `dims = ((1,3), (2,4))` acts on `A[:,k,:,l]` for every `k`
    and `l`, then on `A[k,:,l,:]`. All slices must have the same size. See [`NDimsPlan`](@ref).
    """
    @eval begin
        @doc $doc function $plan_f(A::AbstractArray{T}, dims::Tuple; kws...) where T
            NDimsPlan($plan_f(T, _matrixslicesize(A, dims)...; kws...), size(A), dims)
        end
    end
end