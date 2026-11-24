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


# dims for a transform acting on M-dimensional slices is a non-empty tuple of M-tuples,
# e.g. ((1,3),) or ((1,3),(2,4)) for M = 2, or a single M-tuple such as (1,3), which is
# shorthand for ((1,3),). The plans do not use NDimsPlan(F, szs, (1,3)) directly, since
# there a tuple of integers means applying a vector transform along each dimension.
_slicedims(dims::NTuple{M,Int}, ::Val{M}) where M = (dims,)
_slicedims(dims::Tuple, ::Val) = dims

# Returns the size of the slices for dims normalized by _slicedims.
function _slicesize(A::AbstractArray, dims::Tuple, M::Int)
    if !(dims isa Tuple{NTuple{M,Int},Vararg{NTuple{M,Int}}})
        throw(ArgumentError("dims must be a $M-tuple of dimensions such as $(ntuple(k -> 2k-1, M)), or a non-empty tuple of them such as $((ntuple(k -> 2k-1, M),)), got $dims"))
    end
    _checkslicedims(size(A), dims)
    return map(i -> size(A, i), first(dims))
end

function _dimsdoc(signature, description, M)
    example = M == 2 ? "`dims = ((1,3),)` acts on `A[:,k,:]` for every `k`, and `dims = ((1,3), (2,4))` acts on `A[:,k,:,l]` for every `k` and `l`, then on `A[k,:,l,:]`" :
                       "`dims = ((1,2,4),)` acts on `A[:,:,k,:]` for every `k`"
    single = M == 2 ? "`dims = (1,3)` is the same as `dims = ((1,3),)`" : "`dims = (1,2,4)` is the same as `dims = ((1,2,4),)`"
    """
        $signature

    Plan the $description on the slices of `A` spanned by each entry of `dims`, applied in
    order. For example, $example. When the transform is applied once, the entry can be given
    on its own, so $single. All slices must have the same size. See [`NDimsPlan`](@ref).
    """
end

# Coefficient transforms: (plan, kind, dimension M of the slices, required parameters, optional parameters).
# The plan only depends on the leading size of the slices.
for (plan_f, K, M, req, opt) in ((:plan_sph2fourier, SPHERE, 2, (), ()),
                                 (:plan_sphv2fourier, SPHEREV, 2, (), ()),
                                 (:plan_disk2cxf, DISK, 2, (), (:α, :β)),
                                 (:plan_ann2cxf, ANNULUS, 2, (), (:α, :β, :γ, :ρ)),
                                 (:plan_rectdisk2cheb, RECTDISK, 2, (), (:β,)),
                                 (:plan_tri2cheb, TRIANGLE, 2, (), (:α, :β, :γ)),
                                 (:plan_tet2cheb, TETRAHEDRON, 3, (), (:α, :β, :γ, :δ)),
                                 (:plan_spinsph2fourier, SPINSPHERE, 2, (:s,), ()))
    for k in 0:length(opt)
        params = (req..., opt[1:k]...)
        ex = :(function $plan_f(A::AbstractArray{T}, $(params...), dims::Tuple) where T
                   slicedims = _slicedims(dims, Val($M))
                   NDimsPlan($plan_f(T, first(_slicesize(A, slicedims, $M)), $(params...)), size(A), slicedims)
               end)
        if k == length(opt)
            args = "A, " * join(map(p -> "$p, ", req)) * (isempty(opt) ? "" : "[" * join(opt, ", ") * ",] ") * "dims"
            doc = _dimsdoc("$plan_f($args)", kind2string(K) * " transform", M)
            @eval @doc $doc $ex
        else
            @eval $ex
        end
    end
end

# FFTW synthesis and analysis: (plan, kind, dimension M of the slices, parameters).
# The plan depends on the full size of the slices.
for (plan_f, K, M, params) in ((:plan_sph_synthesis, SPHERESYNTHESIS, 2, ()),
                               (:plan_sph_analysis, SPHEREANALYSIS, 2, ()),
                               (:plan_sphv_synthesis, SPHEREVSYNTHESIS, 2, ()),
                               (:plan_sphv_analysis, SPHEREVANALYSIS, 2, ()),
                               (:plan_disk_synthesis, DISKSYNTHESIS, 2, ()),
                               (:plan_disk_analysis, DISKANALYSIS, 2, ()),
                               (:plan_annulus_synthesis, ANNULUSSYNTHESIS, 2, (:ρ,)),
                               (:plan_annulus_analysis, ANNULUSANALYSIS, 2, (:ρ,)),
                               (:plan_rectdisk_synthesis, RECTDISKSYNTHESIS, 2, ()),
                               (:plan_rectdisk_analysis, RECTDISKANALYSIS, 2, ()),
                               (:plan_tri_synthesis, TRIANGLESYNTHESIS, 2, ()),
                               (:plan_tri_analysis, TRIANGLEANALYSIS, 2, ()),
                               (:plan_tet_synthesis, TETRAHEDRONSYNTHESIS, 3, ()),
                               (:plan_tet_analysis, TETRAHEDRONANALYSIS, 3, ()),
                               (:plan_spinsph_synthesis, SPINSPHERESYNTHESIS, 2, (:s,)),
                               (:plan_spinsph_analysis, SPINSPHEREANALYSIS, 2, (:s,)))
    args = "A, " * join(map(p -> "$p, ", params)) * "dims; flags=FFTW.ESTIMATE"
    doc = _dimsdoc("$plan_f($args)", kind2string(K), M)
    @eval @doc $doc function $plan_f(A::AbstractArray{T}, $(params...), dims::Tuple; kws...) where T
        slicedims = _slicedims(dims, Val($M))
        NDimsPlan($plan_f(T, _slicesize(A, slicedims, $M)..., $(params...); kws...), size(A), slicedims)
    end
end
