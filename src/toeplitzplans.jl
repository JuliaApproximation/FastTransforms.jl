using FFTW
import FFTW: plan_r2r!

struct ToeplitzPlan{T, N, VECS<:NTuple{N,Vector{T}}, P<:Plan{T}} <: Plan{T}
    vectors::VECS
    tmp::Array{T,N}
    dft::P
end

ToeplitzPlan(v::AbstractVector, tmp, dft) = ToeplitPlan((v,), tmp, dft)
ToeplitzPlan(v::NTuple{N,Vector{T}}, tmp::Array{T,N}, dft::Plan{T}) = ToeplitPlan{T,N,typeof(v),typeof(dft)}(v, tmp, dft)

function uppertoeplitz_padvec(v, n)
    S = float(T)
    tmp = zeros(S, 2n-1)
    tmp[1] = v[1]
    copyto!(tmp, n+1, Iterators.reverse(v), 1, n-1)
end

function plan_uppertoeplitz(v::AbstractVector{T}) where T
    tmp = uppertoeplitz_padvec(v, length(v))
    dft = plan_r2r!(tmp, FFTW.R2HC)
    return ToeplitzPlan(dft * tmp, similar(tmp), dft, 1:1)
end

# TODO: support different transforms
# function plan_uppertoeplitz(v1::AbstractVector{T}, v2::AbstractVector{T}) where T
#     S = float(T)
#     m,n = length(v1), length(v2)
#     tmp = zeros(S, 2m-1, 2n-1)
#     pv1 = uppertoeplitz_padvec(v1)
#     pv2 = uppertoeplitz_padvec(v2)
#     dft = plan_r2r!(tmp, FFTW.R2HC)
#     return ToeplitzPlan((r2r!(pv1, FFTW.R2HC), r2r!(pv2, FFTW.R2HC)), tmp, dft, 1:2)
# end

function plan_uppertoeplitz(v::AbstractVector{T}, szs::NTuple{2,Int}, dim::Int) where T
    S = float(T)
    m,n = szs
    if isone(dim)
        tmp = zeros(S, 2m-1, n)
        pv = uppertoeplitz_padvec(v, m)
    else # dim == 2
        tmp = zeros(S, m, 2n-1)
        pv = uppertoeplitz_padvec(v, n)
    end
    dft = plan_r2r!(tmp, FFTW.R2HC, dim)
    return ToeplitzPlan(r2r!(pv, FFTW.R2HC), tmp, dft, dim)
end

function plan_uppertoeplitz(v::AbstractVector{T}, szs::NTuple{2,Int}, dim=(1,2)) where T
    @assert dim == (1,2)
    S = float(T)
    m,n = szs
    tmp = zeros(S, 2m-1, 2n-1)
    pv1 = uppertoeplitz_padvec(v, m)
    pv2 = uppertoeplitz_padvec(v, n)
    dft = plan_r2r!(tmp, FFTW.R2HC, dim)
    return ToeplitzPlan((r2r!(pv1, FFTW.R2HC),r2r!(pv2, FFTW.R2HC)), tmp, dft, dim)
end
