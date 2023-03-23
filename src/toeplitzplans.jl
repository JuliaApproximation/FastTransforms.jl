using FFTW
import FFTW: plan_r2r!

struct ToeplitzPlan{T, N, M, S, VECS<:Tuple{Vararg{Vector{S}}}, P<:Plan{S}, Pi<:Plan{S}} <: Plan{T}
    vectors::VECS
    tmp::Array{S,N}
    dft::P
    idft::Pi
    dims::NTuple{M,Int}
end

ToeplitzPlan{T}(v::AbstractVector, tmp, dft, idft, dims) where T = ToeplitzPlan{T}((v,), tmp, dft, idft, dims)
ToeplitzPlan{T}(v::Tuple{Vararg{Vector{S}}}, tmp::Array{S,N}, dft::Plan{S}, idft::Plan{S}, dims::NTuple{M,Int}) where {T,S,N,M} = ToeplitzPlan{T,N,M,S,typeof(v),typeof(dft), typeof(idft)}(v, tmp, dft, idft, dims)
ToeplitzPlan{T}(v::Tuple{Vararg{Vector{S}}}, tmp::Array{S,N}, dft::Plan{S}, idft::Plan{S}, dims::Int) where {T,S,N} = ToeplitzPlan{T}(v, tmp, dft, idft, (dims,))

size(A::ToeplitzPlan{<:Any,1}) = ((length(A.tmp)+1) ÷ 2,)
function size(A::ToeplitzPlan{<:Any,2,1})
    if A.dims == (1,)
        ((size(A.tmp,1)+1) ÷ 2, size(A.tmp,2))
    else # A.dims == (2,)
        (size(A.tmp,1), (size(A.tmp,2)+1) ÷ 2)
    end
end

function size(A::ToeplitzPlan{<:Any,3,1})
    if A.dims == (1,)
        ((size(A.tmp,1)+1) ÷ 2, size(A.tmp,2), size(A.tmp,3))
    elseif A.dims == (2,)
        (size(A.tmp,1), (size(A.tmp,2)+1) ÷ 2, size(A.tmp,3))
    else
        (size(A.tmp,1), size(A.tmp,2), (size(A.tmp,3)+1) ÷ 2)
    end
end


size(A::ToeplitzPlan{<:Any,2,2}) = ((size(A.tmp,1)+1) ÷ 2, (size(A.tmp,2)+1) ÷ 2)

# based on ToeplitzMatrices.jl
"""
    maybereal(::Type{T}, x)

Return real-valued part of `x` if `T` is a type of a real number, and `x` otherwise.
"""
maybereal(::Type, x) = x
maybereal(::Type{<:Real}, x) = real(x)

function *(A::ToeplitzPlan{T,1}, x::AbstractVector{T}) where T
    vc,tmp,dft,idft = A.vectors[1],A.tmp, A.dft,A.idft
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
    idft * tmp
    @inbounds for k = 1:n
        x[k] = maybereal(T, tmp[k])
    end
    x
end

function *(A::ToeplitzPlan{T,2,1, S}, x::AbstractMatrix{T}) where {T,S}
    vc,tmp,dft = A.vectors[1],A.tmp, A.dft
    M,N = size(tmp)
    m,n = size(x)

    if A.dims == (1,)
        copyto!(view(tmp, 1:m, :), x)
        fill!(view(tmp, m+1:M, :), zero(S))
        dft * tmp
        tmp .= vc .* tmp
    else
        @assert A.dims == (2,)
        copyto!(view(tmp, :, 1:n), x)
        fill!(view(tmp, :, n+1:N), zero(S))
        dft * tmp
        tmp .= tmp .* transpose(vc)
    end
    dft \ tmp
    x .= maybereal.(T, view(tmp,1:m,1:n))
end


function *(A::ToeplitzPlan{T,2,2, S}, X::AbstractMatrix{T}) where {T,S}
    vcs,tmp,dft = A.vectors,A.tmp, A.dft
    vc1,vc2 = vcs
    M,N = size(tmp)
    m,n = size(X)

    @assert A.dims == (1,2)
    copyto!(view(tmp, 1:m, 1:n), X)
    fill!(view(tmp, m+1:M, :), zero(S))
    fill!(view(tmp, 1:m, n+1:N), zero(S))
    dft * tmp
    tmp .= vc1 .* tmp .* transpose(vc2)
    dft \ tmp
    @inbounds for k = 1:m, j = 1:n
        X[k,j] = maybereal(T, tmp[k,j])
    end
    X
end

function *(A::ToeplitzPlan{T,3,1, S}, x::AbstractArray{T,3}) where {T,S}
    vc,tmp,dft = A.vectors[1],A.tmp, A.dft
    M,N,L = size(tmp)
    m,n,l = size(x)

    if A.dims == (1,)
        copyto!(view(tmp, 1:m, :, :), x)
        fill!(view(tmp, m+1:M, :, :), zero(S))
        dft * tmp
        tmp .= vc .* tmp
    elseif A.dims == (2,)
        copyto!(view(tmp, :, 1:n, :), x)
        fill!(view(tmp, :, n+1:N, :), zero(S))
        dft * tmp
        tmp .= tmp .* transpose(vc)
    else
        copyto!(view(tmp, :, :, 1:l), x)
        fill!(view(tmp, :, :, l+1:L), zero(S))
        dft * tmp
        tmp .= tmp .* reshape(vc, 1, 1, L)
    end
    dft \ tmp
    @inbounds for k = 1:m, j = 1:n, ℓ = 1:l
        x[k,j,ℓ] = maybereal(T, tmp[k,j,ℓ])
    end
    x
end



function uppertoeplitz_padvec(v::AbstractVector{T}) where T
    n = length(v)
    S = complex(float(T))
    tmp = zeros(S, 2n-1)
    tmp[1] = v[1]
    copyto!(tmp, n+1, Iterators.reverse(v), 1, n-1)
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
        tmp = zeros(S, 2m-1, n)
        pv = uppertoeplitz_padvec(v[1:m])
    else # dim == 2
        tmp = zeros(S, m, 2n-1)
        pv = uppertoeplitz_padvec(v[1:n])
    end
    dft = plan_fft!(tmp, dim)
    idft = plan_ifft!(similar(tmp), dim)
    return ToeplitzPlan{float(T)}(fft!(pv), tmp, dft, idft, dim)
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

function plan_uppertoeplitz!(v::AbstractVector{T}, szs::NTuple{2,Int}, dim=(1,2)) where T
    @assert dim == (1,2)
    S = complex(float(T))
    m,n = szs
    tmp = zeros(S, 2m-1, 2n-1)
    pv1 = uppertoeplitz_padvec(v[1:m])
    pv2 = uppertoeplitz_padvec(v[1:n])
    dft = plan_fft!(tmp, dim)
    idft = plan_ifft!(similar(tmp), dim)
    return ToeplitzPlan{float(T)}((fft!(pv1), fft!(pv2)), tmp, dft, idft, dim)
end
