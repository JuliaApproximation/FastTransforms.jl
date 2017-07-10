doc"""
Pre-compute a nonuniform fast Fourier transform of type `N`.

For best performance, choose the right number of threads by `FFTW.set_num_threads(4)`, for example.
"""
immutable NUFFTPlan{N,T,FFT} <: Base.DFT.Plan{T}
    U::Matrix{T}
    V::Matrix{T}
    p::FFT
    t::Vector{Int}
    temp::Matrix{T}
    temp2::Matrix{T}
    Ones::Vector{T}
end

doc"""
Computes a nonuniform fast Fourier transform of type I:

```math
f_j = \sum_{k=1}^N c_k e^{-2\pi{\rm i} (j-1)/N \omega_k},\quad{\rm for}\quad 1 \le j \le N.
```
"""
function plan_nufft1{T<:AbstractFloat}(ω::AbstractVector{T}, ϵ::T)
    N = length(ω)
    ωdN = ω/N
    t = AssignClosestEquispacedFFTpoint(ωdN)
    γ = PerturbationParameter(ωdN, AssignClosestEquispacedGridpoint(ωdN))
    K = FindK(γ, ϵ)
    U = constructU( ωdN, K)
    V = constructV( ωdN, K)
    p = plan_ifft!(V, 1)
    temp = zeros(Complex{T}, N, K)
    temp2 = zeros(Complex{T}, N, K)
    Ones = ones(Complex{T}, K)

    NUFFTPlan{1, eltype(U), typeof(p)}(U, V, p, t, temp, temp2, Ones)
end

doc"""
Computes a nonuniform fast Fourier transform of type II:

```math
f_j = \sum_{k=1}^N c_k e^{-2\pi{\rm i} x_j (k-1)},\quad{\rm for}\quad 1 \le j \le N.
```
"""
function plan_nufft2{T<:AbstractFloat}(x::AbstractVector{T}, ϵ::T)
    N = length(x)
    t = AssignClosestEquispacedFFTpoint(x)
    γ = PerturbationParameter(x, AssignClosestEquispacedGridpoint(x))
    K = FindK(γ, ϵ)
    U = constructU(x, K)
    V = constructV(x, K)
    p = plan_fft!(U, 1)
    temp = zeros(Complex{T}, N, K)
    temp2 = zeros(Complex{T}, N, K)
    Ones = ones(Complex{T}, K)

    NUFFTPlan{2, eltype(U), typeof(p)}(U, V, p, t, temp, temp2, Ones)
end

function (*){N,T,V}(p::NUFFTPlan{N,T}, x::AbstractVector{V})
    A_mul_B!(zeros(promote_type(T,V), length(x)), p, x)
end

function Base.A_mul_B!{T}(y::AbstractVector{T}, P::NUFFTPlan{1,T}, c::AbstractVector{T})
    U, V, p, t, temp, temp2, Ones = P.U, P.V, P.p, P.t, P.temp, P.temp2, P.Ones

    # (V.*(N*conj(ifft(In[:,t]*conj(Diagonal(c)*U),1))))*ones(K)

    broadcast!(*, temp, c, U)
    conj!(temp)
    fill!(temp2, zero(T))
    recombine_rows!(temp, t, temp2)
    p*temp2
    conj!(temp2)
    broadcast!(*, temp, V, temp2)
    A_mul_B!(y, temp, Ones)
    scale!(length(c), y)

    y
end

function Base.A_mul_B!{T}(y::AbstractVector{T}, P::NUFFTPlan{2,T}, c::AbstractVector{T})
    U, V, p, t, temp, temp2, Ones = P.U, P.V, P.p, P.t, P.temp, P.temp2, P.Ones

    # (U.*(fft(Diagonal(c)*V,1)[t+1,:]))*ones(K)

    broadcast!(*, temp, c, V)
    p*temp
    reindex_temp!(temp, t, temp2)
    broadcast!(*, temp, U, temp2)
    A_mul_B!(y, temp, Ones)

    y
end

function reindex_temp!{T}(temp::Matrix{T}, t::Vector{Int}, temp2::Matrix{T})
    @inbounds for j = 1:size(temp, 2)
        for i = 1:size(temp, 1)
            temp2[i, j] = temp[t[i], j]
        end
    end
    temp2
end

function recombine_rows!{T}(temp::Matrix{T}, t::Vector{Int}, temp2::Matrix{T})
    @inbounds for j = 1:size(temp, 2)
        for i = 1:size(temp, 1)
            temp2[t[i], j] += temp[i, j]
        end
    end
    temp2
end

doc"""
Pre-compute a nonuniform fast Fourier transform of type I.
"""
nufft1{T<:AbstractFloat}(c::AbstractVector, ω::AbstractVector{T}, ϵ::T) = plan_nufft1(ω, ϵ)*c

doc"""
Pre-compute a nonuniform fast Fourier transform of type II.
"""
nufft2{T<:AbstractFloat}(c::AbstractVector, x::AbstractVector{T}, ϵ::T) = plan_nufft2(x, ϵ)*c

FindK{T<:AbstractFloat}(γ::T, ϵ::T) = Int(ceil(5*γ*exp(lambertw(log(10/ϵ)/γ/7))))
AssignClosestEquispacedGridpoint{T<:AbstractFloat}(x::AbstractVector{T})::AbstractVector{T} = round.([Int], size(x, 1)*x)
AssignClosestEquispacedFFTpoint{T<:AbstractFloat}(x::AbstractVector{T})::Array{Int,1} = mod.(round.([Int], size(x, 1)*x), size(x, 1)) + 1
PerturbationParameter{T<:AbstractFloat}(x::AbstractVector{T}, s_vec::AbstractVector{T})::AbstractFloat = norm(size(x, 1)*x - s_vec, Inf)

function constructU{T<:AbstractFloat}(x::AbstractVector{T}, K::Int)
    # Construct a low rank approximation, using Chebyshev expansions
    # for AK = exp(-2*pi*1im*(x[j]-j/N)*k):
    N = size(x, 1)
    #(s_vec, t, γ) = FindAlgorithmicParameters( x )
    s_vec = AssignClosestEquispacedGridpoint(x)
    er = N*x - s_vec
    γ = norm(er, Inf)
    # colspace vectors:
    Diagonal(exp.(-im*(pi*er)))*ChebyshevP(K-1, er/γ)*Bessel_coeffs(K, γ)
end

function constructV{T<:AbstractFloat}(x::AbstractVector{T}, K::Int)
    N = size(x, 1)
    complex(ChebyshevP(K-1, two(T)*collect(0:N-1)/N - ones(N) ))
end

function Bessel_coeffs{T<:AbstractFloat}(K::Int, γ::T)
    # Calculate the Chebyshev coefficients of exp(-2*pi*1im*x*y) on [-gam,gam]x[0,1]
    cfs = zeros(Complex{T}, K, K)
    arg = -γ*π/two(T)
    for p = 0:K-1
     	for q = mod(p,2):2:K-1
    		cfs[p+1,q+1] = 4*(1im)^q*besselj((p+q)/2,arg).*besselj((q-p)/2,arg)
    	end
    end
    cfs[1,:] = cfs[1,:]/two(T)
    cfs[:,1] = cfs[:,1]/two(T)
    return cfs
end

function ChebyshevP{T<:AbstractFloat}(n::Int, x::AbstractVector{T})
    # Evaluate Chebyshev polynomials of degree 0,...,n at x:
    N = size(x, 1)
    Tcheb = Matrix{T}(N, n+1)

    # T_0(x) = 1.0
    One = convert(eltype(x),1.0)
    @inbounds for j = 1:N
        Tcheb[j, 1] = One
    end
    # T_1(x) = x
    if ( n > 0 )
        @inbounds for j = 1:N
            Tcheb[j, 2] = x[j]
        end
    end
    # 3-term recurrence relation:
    twoX = 2x
    @inbounds for k = 2:n
        @simd for j = 1:N
            Tcheb[j, k+1] = twoX[j]*Tcheb[j, k] - Tcheb[j, k-1]
        end
    end
    return Tcheb
end
