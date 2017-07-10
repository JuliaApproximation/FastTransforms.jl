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
    U = constructU(ωdN, K)
    V = constructV(linspace(zero(T), N-1, N), K)
    p = plan_bfft!(V, 1)
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
    V = constructV(linspace(zero(T), N-1, N), K)
    p = plan_fft!(U, 1)
    temp = zeros(Complex{T}, N, K)
    temp2 = zeros(Complex{T}, N, K)
    Ones = ones(Complex{T}, K)

    NUFFTPlan{2, eltype(U), typeof(p)}(U, V, p, t, temp, temp2, Ones)
end

doc"""
Computes a nonuniform fast Fourier transform of type III:

```math
f_j = \sum_{k=1}^N c_k e^{-2\pi{\rm i} x_j \omega_k},\quad{\rm for}\quad 1 \le j \le N.
```
"""
function plan_nufft3{T<:AbstractFloat}(x::AbstractVector{T}, ω::AbstractVector{T}, ϵ::T)
    N = length(x)
    s = AssignClosestEquispacedGridpoint(x)
    t = AssignClosestEquispacedFFTpoint(x)
    γ = PerturbationParameter(x, s)
    K = FindK(γ, ϵ)
    u = constructU(x, K)
    v = constructV(ω, K)

    p = plan_nufft1(ω, ϵ)

    D1 = Diagonal(1-(s-t+1)/N)
    D2 = Diagonal((s-t+1)/N)
    D3 = Diagonal(exp.(-2*im*T(π)*ω))
    U = hcat(D1*u, D2*u)
    V = hcat(v, D3*v)

    temp = zeros(Complex{T}, N, 2K)
    temp2 = zeros(Complex{T}, N, 2K)
    Ones = ones(Complex{T}, 2K)

    NUFFTPlan{3, eltype(U), typeof(p)}(U, V, p, t, temp, temp2, Ones)
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

    y
end

function Base.A_mul_B!{T}(Y::Matrix{T}, P::NUFFTPlan{1,T}, C::Matrix{T})
    for J = 1:size(Y, 2)
        A_mul_B_col_J!(Y, P, C, J)
    end
    Y
end

function A_mul_B_col_J!{T}(Y::Matrix{T}, P::NUFFTPlan{1,T}, C::Matrix{T}, J::Int)
    U, V, p, t, temp, temp2, Ones = P.U, P.V, P.p, P.t, P.temp, P.temp2, P.Ones

    broadcast_col_J!(*, temp, C, U, J)
    conj!(temp)
    fill!(temp2, zero(T))
    recombine_rows!(temp, t, temp2)
    p*temp2
    conj!(temp2)
    broadcast!(*, temp, V, temp2)
    COLSHIFT = size(C, 1)*(J-1)
    A_mul_B!(Y, temp, Ones, 1+COLSHIFT, 1)

    Y
end

function broadcast_col_J!(f, temp::Matrix, C::Matrix, U::Matrix, J::Int)
    N = size(C, 1)
    COLSHIFT = N*(J-1)
    @inbounds for j = 1:size(temp, 2)
        for i = 1:N
            temp[i,j] = f(C[i+COLSHIFT],U[i,j])
        end
    end
    temp
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

function Base.A_mul_B!{T}(y::AbstractVector{T}, P::NUFFTPlan{3,T}, c::AbstractVector{T})
    U, V, p, t, temp, temp2, Ones = P.U, P.V, P.p, P.t, P.temp, P.temp2, P.Ones

    broadcast!(*, temp2, c, V)
    A_mul_B!(temp, p, temp2)
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

doc"""
Pre-compute a nonuniform fast Fourier transform of type III.
"""
nufft3{T<:AbstractFloat}(c::AbstractVector, x::AbstractVector{T}, ω::AbstractVector{T}, ϵ::T) = plan_nufft3(x, ω, ϵ)*c

const nufft = nufft3
const plan_nufft = plan_nufft3

FindK{T<:AbstractFloat}(γ::T, ϵ::T) = Int(ceil(5*γ*exp(lambertw(log(10/ϵ)/γ/7))))
AssignClosestEquispacedGridpoint{T<:AbstractFloat}(x::AbstractVector{T})::AbstractVector{T} = round.([Int], size(x, 1)*x)
AssignClosestEquispacedFFTpoint{T<:AbstractFloat}(x::AbstractVector{T})::Array{Int,1} = mod.(round.([Int], size(x, 1)*x), size(x, 1)) + 1
PerturbationParameter{T<:AbstractFloat}(x::AbstractVector{T}, s_vec::AbstractVector{T})::AbstractFloat = norm(size(x, 1)*x - s_vec, Inf)

function constructU{T<:AbstractFloat}(x::AbstractVector{T}, K::Int)
    # Construct a low rank approximation, using Chebyshev expansions
    # for AK = exp(-2*pi*1im*(x[j]-j/N)*k):
    N = length(x)
    s = AssignClosestEquispacedGridpoint(x)
    er = N*x - s
    γ = norm(er, Inf)
    Diagonal(exp.(-im*(π*er)))*ChebyshevP(K-1, er/γ)*Bessel_coeffs(K, γ)
end

function constructV{T<:AbstractFloat}(ω::AbstractVector{T}, K::Int)
    complex(ChebyshevP(K-1, ω*(two(T)/length(ω)) - 1))
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
