using FFTW
import FFTW: plan_r2r!


"""
    ToeplitzPlan

applies Toeplitz matrices fast along each dimension.
"""

struct ToeplitzPlan{T, N, Dims, S, VECS, P<:Plan{S}, Pi<:Plan{S}} <: Plan{T}
    vectors::VECS # Vector or Tuple of storage
    tmp::Array{S,N}
    dft::P
    idft::Pi
    dims::Dims
end

ToeplitzPlan{T}(v, tmp::Array{S,N}, dft::Plan{S}, idft::Plan{S}, dims) where {T,S,N} = ToeplitzPlan{T,N,typeof(dims),S,typeof(v),typeof(dft), typeof(idft)}(v, tmp, dft, idft, dims)


divdimby2(d::Int, sz1, szs...) = isone(d) ? ((sz1 + 1) ÷ 2, szs...) : (sz1, divdimby2(d-1, szs...)...)
muldimby2(d::Int, sz1, szs...) = isone(d) ? (max(0,2sz1 - 1), szs...) : (sz1, muldimby2(d-1, szs...)...)

function toeplitzplan_size(dims, szs)
    ret = szs
    for d in dims
        ret = divdimby2(d, ret...)
    end
    ret
end

function to_toeplitzplan_size(dims, szs)
    ret = szs
    for d in dims
        ret = muldimby2(d, ret...)
    end
    ret
end


size(A::ToeplitzPlan) = toeplitzplan_size(A.dims, size(A.tmp))


# based on ToeplitzMatrices.jl
"""
    maybereal(::Type{T}, x)

Return real-valued part of `x` if `T` is a type of a real number, and `x` otherwise.
"""
maybereal(::Type, x) = x
maybereal(::Type{<:Real}, x) = real(x)

function *(A::ToeplitzPlan{T,N}, X::AbstractArray{T,N}) where {T,N}
    vcs,Y,dft,idft,dims = A.vectors,A.tmp, A.dft,A.idft,A.dims

    isempty(X) && return X

    fill!(Y, zero(eltype(Y)))
    copyto!(view(Y, axes(X)...), X)

    # Fourier transform each dimension
    dft * Y

    # Multiply by a diagonal matrix along each dimension by permuting
    # to first dimension
    for (vc,d) in zip(vcs,dims)
        applydim!(v -> v .= vc .* v, Y, d, :)
    end

    # Transform back
    idft * Y

    X .= maybereal.(T, view(Y, axes(X)...))
    X
end


function uppertoeplitz_padvec(v::AbstractVector{T}) where T
    n = length(v)
    S = complex(float(T))
    tmp = zeros(S, max(0,2n-1))
    if n ≠ 0
        tmp[1] = v[1]
        copyto!(tmp, n+1, Iterators.reverse(v), 1, n-1)
    end
    tmp
end

safe_fft!(A) = isempty(A) ? A : fft!(A)

uppertoeplitz_vecs(v, dims::AbstractVector, szs) = [safe_fft!(uppertoeplitz_padvec(v[1:szs[d]])) for d in dims]
uppertoeplitz_vecs(v, dims::Tuple{}, szs) = ()
uppertoeplitz_vecs(v, dims::Tuple, szs) = (safe_fft!(uppertoeplitz_padvec(v[1:szs[first(dims)]])), uppertoeplitz_vecs(v, tail(dims), szs)...)
uppertoeplitz_vecs(v, d::Int, szs) = (safe_fft!(uppertoeplitz_padvec(v[1:szs[d]])),)


# allow FFT to work by making sure tmp is non-empty
safe_tmp(tmp::AbstractArray{<:Any,N}) where N = isempty(tmp) ? similar(tmp, ntuple(_ -> 1, Val(N))...) : tmp

function plan_uppertoeplitz!(v::AbstractVector{T}, szs::NTuple{N,Int}, dim=ntuple(identity,Val(N))) where {T,N}
    S = complex(float(T))
    
    tmp = zeros(S, to_toeplitzplan_size(dim, szs)...)
    dft = plan_fft!(safe_tmp(tmp), dim)
    idft = plan_ifft!(safe_tmp(similar(tmp)), dim)
    
    return ToeplitzPlan{float(T)}(uppertoeplitz_vecs(v, dim, szs), tmp, dft, idft, dim)
end

plan_uppertoeplitz!(v::AbstractVector{T}) where T = plan_uppertoeplitz!(v, size(v))
