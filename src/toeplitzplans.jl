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

ToeplitzPlan{T}(v::AbstractVector{<:Number}, tmp::Array{S,N}, dft::Plan{S}, idft::Plan{S}, dims) where {T,S,N} = ToeplitzPlan{T}((v,), tmp, dft, idft, dims)
ToeplitzPlan{T}(v, tmp::Array{S,N}, dft::Plan{S}, idft::Plan{S}, dims) where {T,S,N} = ToeplitzPlan{T,N,typeof(dims),S,typeof(v),typeof(dft), typeof(idft)}(v, tmp, dft, idft, dims)


divdimby2(d::Int, sz1, szs...) = isone(d) ? ((sz1 + 1) ÷ 2, szs...) : (sz1, divdimby2(d-1, szs...)...)
muldimby2(d::Int, sz1, szs...) = isone(d) ? (2sz1 - 1, szs...) : (sz1, muldimby2(d-1, szs...)...)

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
        Ỹ = PermutedDimsArray(Y, _permfirst(d, N))
        Ỹ .= vc .* Ỹ
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

function plan_uppertoeplitz!(v::AbstractVector{T}) where T
    tmp = uppertoeplitz_padvec(v)
    dft = plan_fft!(tmp)
    idft = plan_ifft!(similar(tmp))
    return ToeplitzPlan{float(T)}(dft * tmp, similar(tmp), dft, idft, (1,))
end

# TODO: support different transforms
# function plan_uppertoeplitz!(v1::AbstractVector{T}, v2::AbstractVector{T}) where T
#     S = float(T)
#     m,n = length(v1), length(v2)
#     tmp = zeros(S, 2m-1, 2n-1)
#     pv1 = uppertoeplitz_padvec(v1)
#     pv2 = uppertoeplitz_padvec(v2)
#     dft = plan_r2r!(tmp, FFTW.R2HC)
#     return ToeplitzPlan((r2r!(pv1, FFTW.R2HC), r2r!(pv2, FFTW.R2HC)), tmp, dft, 1:2)
# end

function plan_uppertoeplitz!(v::AbstractVector{T}, szs::NTuple{2,Int}, dim::Int) where T
    S = complex(float(T))
    m,n = szs
    if isone(dim)
        tmp = zeros(S, max(0,2m-1), n)
        pv = uppertoeplitz_padvec(v[1:m])
    else # dim == 2
        tmp = zeros(S, m, max(0,2n-1))
        pv = uppertoeplitz_padvec(v[1:n])
    end
    if isempty(tmp)
        # dummy plans just to create type
        dft = plan_fft!(similar(tmp, 1, 1), dim)
        idft = plan_ifft!(similar(tmp, 1, 1), dim)
        ToeplitzPlan{float(T)}(pv, tmp, dft, idft, dim)
    else
        dft = plan_fft!(tmp, dim)
        idft = plan_ifft!(similar(tmp), dim)
        return ToeplitzPlan{float(T)}(fft!(pv), tmp, dft, idft, dim)
    end
end

function plan_uppertoeplitz!(v::AbstractVector{T}, szs::NTuple{3,Int}, dim::Int) where T
    S = complex(float(T))
    m,n,l = szs
    if isone(dim)
        tmp = zeros(S, 2m-1, n, l)
        pv = uppertoeplitz_padvec(v[1:m])
    elseif dim == 2
        tmp = zeros(S, m, 2n-1, l)
        pv = uppertoeplitz_padvec(v[1:n])
    else
        @assert dim == 3
        tmp = zeros(S, m, n, 2l-1)
        pv = uppertoeplitz_padvec(v[1:l])
    end
    dft = plan_fft!(tmp, dim)
    idft = plan_ifft!(similar(tmp), dim)
    return ToeplitzPlan{float(T)}(fft!(pv), tmp, dft, idft, dim)
end


uppertoeplitz_vecs(v, dims::AbstractVector, szs) = [fft!(uppertoeplitz_padvec(v[1:szs[d]])) for d in dims]
uppertoeplitz_vecs(v, dims::Tuple{}, szs) = ()
uppertoeplitz_vecs(v, dims::Tuple, szs) = (fft!(uppertoeplitz_padvec(v[1:szs[first(dims)]])), uppertoeplitz_vecs(v, tail(dims), szs)...)

function plan_uppertoeplitz!(v::AbstractVector{T}, szs::NTuple{2,Int}, dim=(1,2)) where T
    S = complex(float(T))
    @assert dim == (1,2)
    tmp = zeros(S, to_toeplitzplan_size(dim, szs)...)
    dft = plan_fft!(tmp, dim)
    idft = plan_ifft!(similar(tmp), dim)
    
    return ToeplitzPlan{float(T)}(uppertoeplitz_vecs(v, dim, szs), tmp, dft, idft, dim)
end
