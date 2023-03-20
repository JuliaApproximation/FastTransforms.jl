using FFTW
import FFTW: plan_r2r!

struct ToeplitzPlan{T, N, S, VECS<:Tuple{Vararg{Vector{S}}}, P<:Plan{S}, DIMS} <: Plan{T}
    vectors::VECS
    tmp::Array{S,N}
    dft::P
    dims::DIMS
end

ToeplitzPlan{T}(v::AbstractVector, tmp, dft, dims) where T = ToeplitzPlan{T}((v,), tmp, dft, dims)
ToeplitzPlan{T}(v::Tuple{Vararg{Vector{S}}}, tmp::Array{S,N}, dft::Plan{S}, dims) where {T,S,N} = ToeplitzPlan{T,N,S,typeof(v),typeof(dft), typeof(dims)}(v, tmp, dft, dims)

# based on ToeplitzMatrices.jl
"""
    maybereal(::Type{T}, x)

Return real-valued part of `x` if `T` is a type of a real number, and `x` otherwise.
"""
maybereal(::Type, x) = x
maybereal(::Type{<:Real}, x) = real(x)

function *(A::ToeplitzPlan{T,1}, x::AbstractVector{T}) where T
    vc,tmp,dft = A.vectors[1],A.tmp, A.dft
    S = eltype(tmp)
    N = length(tmp)
    n = length(x)
    if 2n-1 ≠ N
        throw(DimensionMismatch("Toeplitz plan does not match size of input"))
    end
    copyto!(view(tmp, 1:n), x)
    fill!(view(tmp, n+1:N), zero(S))
    dft * tmp
    tmp .*= vc
    dft \ tmp
    @inbounds for k = 1:n
        x[k] = maybereal(T, tmp[k])
    end
    x
end

function *(A::ToeplitzPlan{T,2, S, <:Tuple{Any}}, x::AbstractMatrix{T}) where {T,S}
    vc,tmp,dft = A.vectors[1],A.tmp, A.dft
    M,N = size(tmp)
    m,n = size(x)

    if A.dims == 1
        copyto!(view(tmp, 1:m, :), x)
        fill!(view(tmp, m+1:M, :), zero(S))
        dft * tmp
        tmp .= vc .* tmp
    else
        copyto!(view(tmp, :, 1:n), x)
        fill!(view(tmp, :, n+1:N), zero(S))
        dft * tmp
        tmp .= tmp .* transpose(vc)
    end
    dft \ tmp
    @inbounds for k = 1:m, j = 1:n
        x[k,j] = maybereal(T, tmp[k,j])
    end
    x
end

function *(A::ToeplitzPlan{T,2, S, <:Tuple{Any,Any}}, x::AbstractMatrix{T}) where {T,S}
    vcs,tmp,dft = A.vectors,A.tmp, A.dft
    vc1,vc2 = vcs
    M,N = size(tmp)
    m,n = size(x)

    @assert A.dims == (1,2)
    copyto!(view(tmp, 1:m, 1:n), x)
    fill!(view(tmp, m+1:M, :), zero(S))
    fill!(view(tmp, 1:m, n+1:N), zero(S))
    dft * tmp
    tmp .= vc1 .* tmp .* transpose(vc2)
    dft \ tmp
    @inbounds for k = 1:m, j = 1:n
        x[k,j] = maybereal(T, tmp[k,j])
    end
    x
end


function uppertoeplitz_padvec(v::AbstractVector{T}, n) where T
    S = complex(float(T))
    tmp = zeros(S, 2n-1)
    tmp[1] = v[1]
    copyto!(tmp, n+1, Iterators.reverse(v), 1, n-1)
end

function plan_uppertoeplitz!(v::AbstractVector{T}) where T
    tmp = uppertoeplitz_padvec(v, length(v))
    dft = plan_fft!(tmp)
    return ToeplitzPlan{float(T)}(dft * tmp, similar(tmp), dft, (1,))
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
        tmp = zeros(S, 2m-1, n)
        pv = uppertoeplitz_padvec(v, m)
    else # dim == 2
        tmp = zeros(S, m, 2n-1)
        pv = uppertoeplitz_padvec(v, n)
    end
    dft = plan_fft!(tmp, dim)
    return ToeplitzPlan{float(T)}(fft!(pv), tmp, dft, dim)
end

function plan_uppertoeplitz!(v::AbstractVector{T}, szs::NTuple{2,Int}, dim=(1,2)) where T
    @assert dim == (1,2)
    S = complex(float(T))
    m,n = szs
    tmp = zeros(S, 2m-1, 2n-1)
    pv1 = uppertoeplitz_padvec(v, m)
    pv2 = uppertoeplitz_padvec(v, n)
    dft = plan_fft!(tmp, dim)
    return ToeplitzPlan{float(T)}((fft!(pv1), fft!(pv2)), tmp, dft, dim)
end