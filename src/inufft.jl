"""
Pre-computes an inverse nonuniform fast Fourier transform of type `N`.

For best performance, choose the right number of threads by `FFTW.set_num_threads(4)`, for example.
"""
struct iNUFFTPlan{N,T,S,PT} <: Plan{T}
    pt::PT
    TP::Toeplitz{T}
    r::Vector{T}
    p::Vector{T}
    Ap::Vector{T}
    ϵ::S
end

"""
Pre-computes an inverse nonuniform fast Fourier transform of type I.
"""
function plan_inufft1(ω::AbstractVector{T}, ϵ::T) where T<:AbstractFloat
    N = length(ω)
    p = plan_nufft1(ω, ϵ)
    pt = plan_nufft2(ω/N, ϵ)
    c = p*ones(Complex{T}, N)
    r = conj(c)
    avg = (r[1]+c[1])/2
    r[1] = avg
    c[1] = avg
    TP = Toeplitz(c, r)
    r = zero(c)
    p = zero(c)
    Ap = zero(c)

    iNUFFTPlan{1, eltype(TP), typeof(ϵ), typeof(pt)}(pt, TP, r, p, Ap, ϵ)
end

"""
Pre-computes an inverse nonuniform fast Fourier transform of type II.
"""
function plan_inufft2(x::AbstractVector{T}, ϵ::T) where T<:AbstractFloat
    N = length(x)
    pt = plan_nufft1(N*x, ϵ)
    r = pt*ones(Complex{T}, N)
    c = conj(r)
    avg = (r[1]+c[1])/2
    r[1] = avg
    c[1] = avg
    TP = Toeplitz(c, r)
    r = zero(c)
    p = zero(c)
    Ap = zero(c)

    iNUFFTPlan{2, eltype(TP), typeof(ϵ), typeof(pt)}(pt, TP, r, p, Ap, ϵ)
end


function (*)(p::iNUFFTPlan{N,T}, x::AbstractVector{V}) where {N,T,V}
    A_mul_B!(zeros(promote_type(T,V), length(x)), p, x)
end

function mul!(c::AbstractVector{T}, P::iNUFFTPlan{1,T}, f::AbstractVector{T}) where T
    pt, TP, r, p, Ap, ϵ = P.pt, P.TP, P.r, P.p, P.Ap, P.ϵ
    cg_for_inufft(TP, c, f, r, p, Ap, 50, 100ϵ)
    conj!(mul!(c, pt, conj!(c)))
end


function mul!(c::AbstractVector{T}, P::iNUFFTPlan{2,T}, f::AbstractVector{T}) where T
    pt, TP, r, p, Ap, ϵ = P.pt, P.TP, P.r, P.p, P.Ap, P.ϵ
    cg_for_inufft(TP, c, conj!(pt*conj!(f)), r, p, Ap, 50, 100ϵ)
    conj!(f)
    c
end

"""
Computes an inverse nonuniform fast Fourier transform of type I.
"""
inufft1(c::AbstractVector, ω::AbstractVector{T}, ϵ::T) where {T<:AbstractFloat} = plan_inufft1(ω, ϵ)*c

"""
Computes an inverse nonuniform fast Fourier transform of type II.
"""
inufft2(c::AbstractVector, x::AbstractVector{T}, ϵ::T) where {T<:AbstractFloat} = plan_inufft2(x, ϵ)*c

function cg_for_inufft(A::ToeplitzMatrices.AbstractToeplitz{T}, x::AbstractVector{T}, b::AbstractVector{T}, r::AbstractVector{T}, p::AbstractVector{T}, Ap::AbstractVector{T}, max_it::Integer, tol::Real) where T
	n = length(b)
	n1, n2 = size(A)
	n == n1 == n2 || throw(DimensionMismatch(""))
    nrmb = norm(b)
    if nrmb == 0 nrmb = one(typeof(nrmb)) end
	copy!(x, b)
    fill!(r, zero(T))
    fill!(p, zero(T))
    fill!(Ap, zero(T))
    # r = b - A*x
    copy!(r, b)
    mul!(r, A, x, -one(T), one(T))
	copy!(p, r)
	nrm2 = r⋅r
    for k = 1:max_it
        # Ap = A*p
        mul!(Ap, A, p)
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
