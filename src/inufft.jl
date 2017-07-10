doc"""
Pre-computes an inverse nonuniform fast Fourier transform of type `N`.

For best performance, choose the right number of threads by `FFTW.set_num_threads(4)`, for example.
"""
immutable iNUFFTPlan{N,T,S,PT} <: Base.DFT.Plan{T}
    pt::PT
    TP::Toeplitz{T}
    ϵ::S
end

doc"""
Pre-computes an inverse nonuniform fast Fourier transform of type I.
"""
function plan_inufft1{T<:AbstractFloat}(ω::AbstractVector{T}, ϵ::T)
    N = length(ω)
    p = plan_nufft1(ω, ϵ)
    pt = plan_nufft2(ω/N, ϵ)
    c = p*ones(Complex{T}, N)
    r = conj(c)
    avg = (r[1]+c[1])/2
    r[1] = avg
    c[1] = avg
    TP = Toeplitz(c, r)

    iNUFFTPlan{1, eltype(TP), typeof(ϵ), typeof(pt)}(pt, TP, ϵ)
end

doc"""
Pre-computes an inverse nonuniform fast Fourier transform of type II.
"""
function plan_inufft2{T<:AbstractFloat}(x::AbstractVector{T}, ϵ::T)
    N = length(x)
    pt = plan_nufft1(N*x, ϵ)
    r = pt*ones(Complex{T}, N)
    c = conj(r)
    avg = (r[1]+c[1])/2
    r[1] = avg
    c[1] = avg
    TP = Toeplitz(c, r)

    iNUFFTPlan{2, eltype(TP), typeof(ϵ), typeof(pt)}(pt, TP, ϵ)
end

function (*){N,T,V}(p::iNUFFTPlan{N,T}, x::AbstractVector{V})
    A_mul_B!(zeros(promote_type(T,V), length(x)), p, x)
end

function Base.A_mul_B!{T}(c::AbstractVector{T}, P::iNUFFTPlan{1,T}, f::AbstractVector{T})
    pt, TP, ϵ = P.pt, P.TP, P.ϵ
    cg(TP, c, f, 50, 100ϵ)
    conj!(A_mul_B!(c, pt, conj!(c)))
end

function Base.A_mul_B!{T}(c::AbstractVector{T}, P::iNUFFTPlan{2,T}, f::AbstractVector{T})
    pt, TP, ϵ = P.pt, P.TP, P.ϵ
    cg(TP, c, conj!(pt*conj!(f)), 50, 100ϵ)
    conj!(f)
    c
end

doc"""
Computes an inverse nonuniform fast Fourier transform of type I.
"""
inufft1{T<:AbstractFloat}(c::AbstractVector, ω::AbstractVector{T}, ϵ::T) = plan_inufft1(ω, ϵ)*c

doc"""
Computes an inverse nonuniform fast Fourier transform of type II.
"""
inufft2{T<:AbstractFloat}(c::AbstractVector, x::AbstractVector{T}, ϵ::T) = plan_inufft2(x, ϵ)*c

function cg{T}(A::ToeplitzMatrices.AbstractToeplitz{T}, x::AbstractVector{T}, b::AbstractVector{T}, max_it::Integer, tol::Real)
	n = length(b)
	n1, n2 = size(A)
	n == n1 == n2 || throw(DimensionMismatch(""))
    nrmb = norm(b)
    if nrmb == 0 nrmb = one(typeof(nrmb)) end
	copy!(x, b)
    r = zero(x)
    p = zero(x)
    Ap = zero(x)
    # r = b - A*x
    copy!(r, b)
    A_mul_B!(-one(T), A, x, one(T), r)
	copy!(p, r)
	nrm2 = r⋅r
    for k = 1:max_it
        # Ap = A*p
        A_mul_B!(one(T), A, p, zero(T), Ap)
		α = nrm2/(p⋅Ap)
        @inbounds @simd for l = 1:n
            x[l] += α*p[l]
            r[l] -= α*Ap[l]
        end
		nrm2new = r⋅r
        cst = nrm2new/nrm2
        @inbounds @simd for l = 1:n
            p[l] = muladd(cst, p[l], r[l])
        end
		nrm2 = nrm2new
        if sqrt(abs(nrm2)) ≤ tol*nrmb break end
	end
    return x
end
