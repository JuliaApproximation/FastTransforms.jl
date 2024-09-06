struct ArrayPlan{T, FF<:FTPlan{<:T}, Szs<:Tuple, Dims<:Tuple{<:Int}} <: Plan{T}
    F::FF
    szs::Szs
    dims::Dims
end
size(P::ArrayPlan) = P.szs
size(P::ArrayPlan, k::Int) = P.szs[k]
size(P::ArrayPlan, k...) = P.szs[[k...]]

function ArrayPlan(F::FTPlan{<:T}, c::AbstractArray{T}, dims::Tuple{<:Int}=(1,)) where T
    szs = size(c)
    @assert F.n == szs[dims[1]]
    ArrayPlan(F, size(c), dims)
end

function inv_perm(d::Vector{<:Int})
    inv_d = Vector{Int}(undef, length(d))
    for (i, val) in enumerate(d)
        inv_d[val] = i
    end
    return inv_d
end
inv_perm(d::Tuple) = inv_perm([d...])

function *(P::ArrayPlan, f::AbstractArray)
    F, dims, szs = P.F, P.dims, P.szs
    @assert length(dims) == 1
    @assert szs == size(f)
    d = first(dims)

    perm = (d, ntuple(i-> i + (i >= d), ndims(f) -1)...)
    fp = permutedims(f, perm)

    fr = reshape(fp, size(fp,1), :)

    permutedims(reshape(F*fr, size(fp)...), inv_perm(perm))
end

function \(P::ArrayPlan, f::AbstractArray)
    F, dims, szs = P.F, P.dims, P.szs
    @assert length(dims) == 1
    @assert szs == size(f)
    d = first(dims)

    perm = (d, ntuple(i-> i + (i >= d), ndims(f) -1)...)
    fp = permutedims(f, perm)

    fr = reshape(fp, size(fp,1), :)

    permutedims(reshape(F\fr, size(fp)...), inv_perm(perm))
end

struct NDimsPlan{T, FF<:ArrayPlan{<:T}, Szs<:Tuple, Dims<:Tuple} <: Plan{T}
    F::FF
    szs::Szs
    dims::Dims
    function NDimsPlan(F, szs, dims)
        if length(Set(szs[[dims...]])) > 1
            error("Different size in dims axes not yet implemented in N-dimensional transform.")
        end
        new{eltype(F), typeof(F), typeof(szs), typeof(dims)}(F, szs, dims)
    end
end

size(P::NDimsPlan) = P.szs
size(P::NDimsPlan, k::Int) = P.szs[k]
size(P::NDimsPlan, k...) = P.szs[[k...]]

function NDimsPlan(F::FTPlan, szs::Tuple, dims::Tuple)
    NDimsPlan(ArrayPlan(F, szs, (first(dims),)), szs, dims)
end

function *(P::NDimsPlan, f::AbstractArray)
    F, dims = P.F, P.dims
    @assert size(P) == size(f)
    g = copy(f)
    t = 1:ndims(g)
    d1 = dims[1]
    for d in dims
        perm = ntuple(k -> k == d1 ? t[d] : k == d ? t[d1] : t[k], ndims(g))
        gp = permutedims(g, perm)
        g = permutedims(F*gp, inv_perm(perm))
    end
    return g
end

function \(P::NDimsPlan, f::AbstractArray)
    F, dims = P.F, P.dims
    @assert size(P) == size(f)
    g = copy(f)
    t = 1:ndims(g)
    d1 = dims[1]
    for d in dims
        perm = ntuple(k -> k == d1 ? t[d] : k == d ? t[d1] : t[k], ndims(g))
        gp = permutedims(g, perm)
        g = permutedims(F\gp, inv_perm(perm))
    end
    return g
end