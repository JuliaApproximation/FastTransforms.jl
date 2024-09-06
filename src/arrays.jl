struct ArrayPlan{T, F<:FTPlan{<:T}, Szs<:Tuple, Dims<:Tuple{<:Int}} <: Plan{T}
    F::F
    szs::Szs
    dims::Dims
end
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

function *(P::ArrayPlan, f::AbstractArray)
    F, dims, szs = P.F, P.dims, P.szs
    @assert length(dims) == 1
    @assert szs == size(f)
    d = first(dims)

    perm = [d; setdiff(1:ndims(f), d)]
    fp = permutedims(f, perm)

    fr = reshape(fp, size(fp,1), prod(size(fp)[2:end]))

    permutedims(reshape(F*fr, size(fp)...), inv_perm(perm))
end

function \(P::ArrayPlan, f::AbstractArray)
    F, dims, szs = P.F, P.dims, P.szs
    @assert length(dims) == 1
    @assert szs == size(f)
    d = first(dims)

    perm = [d; setdiff(1:ndims(f), d)]
    fp = permutedims(f, perm)

    fr = reshape(fp, size(fp,1), prod(size(fp)[2:end]))

    permutedims(reshape(F\fr, size(fp)...), inv_perm(perm))
end