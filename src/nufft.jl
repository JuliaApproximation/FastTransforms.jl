doc"""
Pre-computes a nonuniform fast Fourier transform of type `N`.

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
Pre-computes a nonuniform fast Fourier transform of type I.
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
Pre-computes a nonuniform fast Fourier transform of type II.
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
Pre-computes a nonuniform fast Fourier transform of type III.
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

function (*){N,T,V}(p::NUFFTPlan{N,T}, c::AbstractArray{V})
    A_mul_B!(zeros(promote_type(T,V), size(c)), p, c)
end

function Base.A_mul_B!{T}(f::AbstractVector{T}, P::NUFFTPlan{1,T}, c::AbstractVector{T})
    U, V, p, t, temp, temp2, Ones = P.U, P.V, P.p, P.t, P.temp, P.temp2, P.Ones

    broadcast!(*, temp, c, U)
    conj!(temp)
    fill!(temp2, zero(T))
    recombine_rows!(temp, t, temp2)
    p*temp2
    conj!(temp2)
    broadcast!(*, temp, V, temp2)
    A_mul_B!(f, temp, Ones)

    f
end

function Base.A_mul_B!{N,T}(F::Matrix{T}, P::NUFFTPlan{N,T}, C::Matrix{T})
    for J = 1:size(F, 2)
        A_mul_B_col_J!(F, P, C, J)
    end
    F
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

function A_mul_B_col_J!{T}(F::Matrix{T}, P::NUFFTPlan{1,T}, C::Matrix{T}, J::Int)
    U, V, p, t, temp, temp2, Ones = P.U, P.V, P.p, P.t, P.temp, P.temp2, P.Ones

    broadcast_col_J!(*, temp, C, U, J)
    conj!(temp)
    fill!(temp2, zero(T))
    recombine_rows!(temp, t, temp2)
    p*temp2
    conj!(temp2)
    broadcast!(*, temp, V, temp2)
    COLSHIFT = size(C, 1)*(J-1)
    A_mul_B!(F, temp, Ones, 1+COLSHIFT, 1)

    F
end

function Base.A_mul_B!{T}(f::AbstractVector{T}, P::NUFFTPlan{2,T}, c::AbstractVector{T})
    U, V, p, t, temp, temp2, Ones = P.U, P.V, P.p, P.t, P.temp, P.temp2, P.Ones

    broadcast!(*, temp, c, V)
    p*temp
    reindex_temp!(temp, t, temp2)
    broadcast!(*, temp, U, temp2)
    A_mul_B!(f, temp, Ones)

    f
end

function A_mul_B_col_J!{T}(F::Matrix{T}, P::NUFFTPlan{2,T}, C::Matrix{T}, J::Int)
    U, V, p, t, temp, temp2, Ones = P.U, P.V, P.p, P.t, P.temp, P.temp2, P.Ones

    broadcast_col_J!(*, temp, C, V, J)
    p*temp
    reindex_temp!(temp, t, temp2)
    broadcast!(*, temp, U, temp2)
    COLSHIFT = size(C, 1)*(J-1)
    A_mul_B!(F, temp, Ones, 1+COLSHIFT, 1)

    F
end

function Base.A_mul_B!{T}(f::AbstractVector{T}, P::NUFFTPlan{3,T}, c::AbstractVector{T})
    U, V, p, t, temp, temp2, Ones = P.U, P.V, P.p, P.t, P.temp, P.temp2, P.Ones

    broadcast!(*, temp2, c, V)
    A_mul_B!(temp, p, temp2)
    reindex_temp!(temp, t, temp2)
    broadcast!(*, temp, U, temp2)
    A_mul_B!(f, temp, Ones)

    f
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
Computes a nonuniform fast Fourier transform of type I:

```math
f_j = \sum_{k=1}^N c_k e^{-2\pi{\rm i} (j-1)/N \omega_k},\quad{\rm for}\quad 1 \le j \le N.
```
"""
nufft1{T<:AbstractFloat}(c::AbstractVector, ω::AbstractVector{T}, ϵ::T) = plan_nufft1(ω, ϵ)*c

doc"""
Computes a nonuniform fast Fourier transform of type II:

```math
f_j = \sum_{k=1}^N c_k e^{-2\pi{\rm i} x_j (k-1)},\quad{\rm for}\quad 1 \le j \le N.
```
"""
nufft2{T<:AbstractFloat}(c::AbstractVector, x::AbstractVector{T}, ϵ::T) = plan_nufft2(x, ϵ)*c

doc"""
Computes a nonuniform fast Fourier transform of type III:

```math
f_j = \sum_{k=1}^N c_k e^{-2\pi{\rm i} x_j \omega_k},\quad{\rm for}\quad 1 \le j \le N.
```
"""
nufft3{T<:AbstractFloat}(c::AbstractVector, x::AbstractVector{T}, ω::AbstractVector{T}, ϵ::T) = plan_nufft3(x, ω, ϵ)*c

const nufft = nufft3
const plan_nufft = plan_nufft3


doc"""
Pre-computes a 2D nonuniform fast Fourier transform.

For best performance, choose the right number of threads by `FFTW.set_num_threads(4)`, for example.
"""
immutable NUFFT2DPlan{T,P1,P2} <: Base.DFT.Plan{T}
    p1::P1
    p2::P2
    temp::Vector{T}
end

doc"""
Pre-computes a 2D nonuniform fast Fourier transform of type II-II.
"""
function plan_nufft2{T<:AbstractFloat}(x::AbstractVector{T}, y::AbstractVector{T}, ϵ::T)
    p1 = plan_nufft2(x, ϵ)
    p2 = plan_nufft2(y, ϵ)
    temp = zeros(Complex{T}, length(y))

    NUFFT2DPlan(p1, p2, temp)
end

function (*){T,V}(p::NUFFT2DPlan{T}, C::Matrix{V})
    A_mul_B!(zeros(promote_type(T,V), size(C)), p, C)
end

function Base.A_mul_B!{T}(F::Matrix{T}, P::NUFFT2DPlan{T}, C::Matrix{T})
    p1, p2, temp = P.p1, P.p2, P.temp

    # Apply 1D x-plan to all columns
    A_mul_B!(F, p1, C)

    # Apply 1D y-plan to all rows
    for i = 1:size(C, 1)
        @inbounds for j = 1:size(F, 2) temp[j] = F[i,j] end
        A_mul_B!(temp, p2, temp)
        @inbounds for j = 1:size(F, 2) F[i,j] = temp[j] end
    end

    F
end

doc"""
Computes a 2D nonuniform fast Fourier transform of type II-II:

```math
f_{j_1,j_2} = \sum_{k_1=1}^M\sum_{k_2=1}^N C_{k_1,k_2} e^{-2\pi{\rm i} (x_{j_1}(k_1-1) + x_{j_2}(k_2-1))},\quad{\rm for}\quad 1 \le j_1 \le M,\quad 1 \le j_2 \le N.
```
"""
nufft2{T<:AbstractFloat}(C::Matrix, x::AbstractVector{T}, y::AbstractVector{T}, ϵ::T) = plan_nufft2(x, y, ϵ)*C


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
