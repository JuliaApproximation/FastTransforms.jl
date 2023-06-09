import Base.Math: @horner

const FORWARD  =  true
const BACKWARD = false

const sqrtpi = 1.772453850905516027298
const edivsqrt2pi = 1.084437551419227546612

"""
Compute a typed 0.5.
"""
half(x::Number) = oftype(x,0.5)
half(x::Integer) = half(float(x))
half(::Type{T}) where {T<:Number} = convert(T,0.5)
half(::Type{T}) where {T<:Integer} = half(AbstractFloat)

"""
Compute a typed 2.
"""
two(x::Number) = oftype(x,2)
two(::Type{T}) where {T<:Number} = convert(T,2)

"""
The Kronecker ``\\delta`` function:

```math
\\delta_{k,j} = \\left\\{\\begin{array}{ccc} 1 & {\\rm for} & k = j,\\\\ 0 & {\\rm for} & k \\ne j.\\end{array}\\right.
```
"""
δ(k::Integer,j::Integer) = k == j ? 1 : 0


"""
Pochhammer symbol ``(x)_n = \\frac{\\Gamma(x+n)}{\\Gamma(x)}`` for the rising factorial.
"""
function pochhammer(x::Number,n::Integer)
    ret = one(x)
    if n≥0
        for i=0:n-1
            ret *= x+i
        end
    else
        ret /= pochhammer(x+n,-n)
    end
    ret
end

pochhammer(x::Number,n::Number) = isinteger(n) ? pochhammer(x,Int(n)) : ogamma(x)/ogamma(x+n)

function pochhammer(x::Number,n::UnitRange{T}) where T<:Real
    ret = Vector{promote_type(typeof(x),T)}(undef,length(n))
    ret[1] = pochhammer(x,first(n))
    for i=2:length(n)
        ret[i] = (x+n[i]-1)*ret[i-1]
    end
    ret
end

lgamma(x) = logabsgamma(x)[1]

ogamma(x::Number) = (isinteger(x) && x<0) ? zero(float(x)) : inv(gamma(x))

"""
Stirling's asymptotic series for ``\\Gamma(z)``.
"""
stirlingseries(z) = gamma(z)*sqrt((z/π)/2)*exp(z)/z^z

function stirlingseries(z::Float64)
    if z ≥ 3274.12075200175       # N =  4
        @horner(inv(z),1.0,0.08333333333333333,0.003472222222222222,-0.0026813271604938273)
    elseif z ≥ 590.1021805526798  # N =  5
        @horner(inv(z),1.0,0.08333333333333333,0.003472222222222222,-0.0026813271604938273,-0.00022947209362139917)
    elseif z ≥ 195.81733962412835 # N =  6
        @horner(inv(z),1.0,0.08333333333333333,0.003472222222222222,-0.0026813271604938273,-0.00022947209362139917,0.0007840392217200666)
    elseif z ≥ 91.4692823071966   # N =  7
        @horner(inv(z),1.0,0.08333333333333333,0.003472222222222222,-0.0026813271604938273,-0.00022947209362139917,0.0007840392217200666,6.972813758365857e-5)
    elseif z ≥ 52.70218954633605  # N =  8
        @horner(inv(z),1.0,0.08333333333333333,0.003472222222222222,-0.0026813271604938273,-0.00022947209362139917,0.0007840392217200666,6.972813758365857e-5,-0.0005921664373536939)
    elseif z ≥ 34.84031591198865  # N =  9
        @horner(inv(z),1.0,0.08333333333333333,0.003472222222222222,-0.0026813271604938273,-0.00022947209362139917,0.0007840392217200666,6.972813758365857e-5,-0.0005921664373536939,-5.171790908260592e-5)
    elseif z ≥ 25.3173982783047   # N = 10
        @horner(inv(z),1.0,0.08333333333333333,0.003472222222222222,-0.0026813271604938273,-0.00022947209362139917,0.0007840392217200666,6.972813758365857e-5,-0.0005921664373536939,-5.171790908260592e-5,0.0008394987206720873)
    elseif z ≥ 19.685015283078513 # N = 11
        @horner(inv(z),1.0,0.08333333333333333,0.003472222222222222,-0.0026813271604938273,-0.00022947209362139917,0.0007840392217200666,6.972813758365857e-5,-0.0005921664373536939,-5.171790908260592e-5,0.0008394987206720873,7.204895416020011e-5)
    elseif z ≥ 16.088669099569266 # N = 12
        @horner(inv(z),1.0,0.08333333333333333,0.003472222222222222,-0.0026813271604938273,-0.00022947209362139917,0.0007840392217200666,6.972813758365857e-5,-0.0005921664373536939,-5.171790908260592e-5,0.0008394987206720873,7.204895416020011e-5,-0.0019144384985654776)
    elseif z ≥ 13.655055978888104 # N = 13
        @horner(inv(z),1.0,0.08333333333333333,0.003472222222222222,-0.0026813271604938273,-0.00022947209362139917,0.0007840392217200666,6.972813758365857e-5,-0.0005921664373536939,-5.171790908260592e-5,0.0008394987206720873,7.204895416020011e-5,-0.0019144384985654776,-0.00016251626278391583)
    elseif z ≥ 11.93238782087875  # N = 14
        @horner(inv(z),1.0,0.08333333333333333,0.003472222222222222,-0.0026813271604938273,-0.00022947209362139917,0.0007840392217200666,6.972813758365857e-5,-0.0005921664373536939,-5.171790908260592e-5,0.0008394987206720873,7.204895416020011e-5,-0.0019144384985654776,-0.00016251626278391583,0.00640336283380807)
    elseif z ≥ 10.668852439197263 # N = 15
        @horner(inv(z),1.0,0.08333333333333333,0.003472222222222222,-0.0026813271604938273,-0.00022947209362139917,0.0007840392217200666,6.972813758365857e-5,-0.0005921664373536939,-5.171790908260592e-5,0.0008394987206720873,7.204895416020011e-5,-0.0019144384985654776,-0.00016251626278391583,0.00640336283380807,0.0005401647678926045)
    elseif z ≥ 9.715358216638403  # N = 16
        @horner(inv(z),1.0,0.08333333333333333,0.003472222222222222,-0.0026813271604938273,-0.00022947209362139917,0.0007840392217200666,6.972813758365857e-5,-0.0005921664373536939,-5.171790908260592e-5,0.0008394987206720873,7.204895416020011e-5,-0.0019144384985654776,-0.00016251626278391583,0.00640336283380807,0.0005401647678926045,-0.02952788094569912)
    elseif z ≥ 8.979120323411497  # N = 17
        @horner(inv(z),1.0,0.08333333333333333,0.003472222222222222,-0.0026813271604938273,-0.00022947209362139917,0.0007840392217200666,6.972813758365857e-5,-0.0005921664373536939,-5.171790908260592e-5,0.0008394987206720873,7.204895416020011e-5,-0.0019144384985654776,-0.00016251626278391583,0.00640336283380807,0.0005401647678926045,-0.02952788094569912,-0.002481743600264998)
    else
        gamma(z)*sqrt(z/2π)*exp(z)/z^z
    end
end



stirlingremainder(z::Number,N::Int) = (1+zeta(N))*gamma(N)/((2π)^(N+1)*z^N)/stirlingseries(z)

Aratio(n::Int,α::Float64,β::Float64) = exp((n/2+α+1/4)*log1p(-β/(n+α+β+1))+(n/2+β+1/4)*log1p(-α/(n+α+β+1))+(n/2+1/4)*log1p(α/(n+1))+(n/2+1/4)*log1p(β/(n+1)))
Aratio(n::Number,α::Number,β::Number) = (1+(α+1)/n)^(n+α+1/2)*(1+(β+1)/n)^(n+β+1/2)/(1+(α+β+1)/n)^(n+α+β+1/2)/(1+(zero(α)+zero(β))/n)^(n+1/2)

Cratio(n::Int,α::Float64,β::Float64) = exp((n+α+1/2)*log1p((α-β)/(2n+α+β+2))+(n+β+1/2)*log1p((β-α)/(2n+α+β+2))-log1p((α+β+2)/2n)/2)/sqrt(n)
Cratio(n::Number,α::Number,β::Number) = n^(-1/2)*(1+(α+1)/n)^(n+α+1/2)*(1+(β+1)/n)^(n+β+1/2)/(1+(α+β+2)/2n)^(2n+α+β+3/2)


Anαβ(n::Number,α::Number,β::Number) = 2^(α+β+1)/(2n+α+β+1)*exp(lgamma(n+α+1)-lgamma(n+α+β+1)+lgamma(n+β+1)-lgamma(n+1))
function Anαβ(n::Integer,α::Number,β::Number)
    if n==0
        2^(α+β+1)*beta(α+1,β+1)
    else
        val = Anαβ(0,α,β)
        for i=1:n
            val *= (i+α)*(i+β)/(i+α+β)/i*(2i+α+β-1)/(2i+α+β+1)
        end
        val
    end
end

function Anαβ(n::Integer,α::Float64,β::Float64)
    if n+min(α,β,α+β,0) ≥ 7.979120323411497
        2 ^ (α+β+1)/(2n+α+β+1)*stirlingseries(n+α+1)*Aratio(n,α,β)/stirlingseries(n+α+β+1)*stirlingseries(n+β+1)/stirlingseries(n+one(Float64))
    else
        (n+1)*(n+α+β+1)/(n+α+1)/(n+β+1)*Anαβ(n+1,α,β)*((2n+α+β+3)/(2n+α+β+1))
    end
end


"""
The Lambda function ``\\Lambda(z) = \\frac{\\Gamma(z+\\frac{1}{2})}{\\Gamma(z+1)}`` for the ratio of gamma functions.
"""
Λ(z::Number) = Λ(z, half(z), one(z))

"""
For 64-bit floating-point arithmetic, the Lambda function uses the asymptotic series for ``\\tau`` in Appendix B of

I. Bogaert and B. Michiels and J. Fostier, 𝒪(1) computation of Legendre polynomials and Gauss–Legendre nodes and weights for parallel computing, *SIAM J. Sci. Comput.*, **34**:C83–C101, 2012.
"""
function Λ(x::Float64)
    if x > 9.84475
        xp = x+0.25
        @horner(inv(xp^2),1.0,-1.5625e-02,2.5634765625e-03,-1.2798309326171875e-03,1.343511044979095458984375e-03,-2.432896639220416545867919921875e-03,6.7542375336415716446936130523681640625e-03)/sqrt(xp)
    else
        (x+1.0)*Λ(x+1.0)/(x+0.5)
    end
end

"""
The Lambda function ``\\Lambda(z,λ₁,λ₂) = \\frac{\\Gamma(z+\\lambda_1)}{Γ(z+\\lambda_2)}`` for the ratio of gamma functions.
"""
function Λ(z::Real, λ₁::Real, λ₂::Real)
    if z+λ₁ > 0 && z+λ₂ > 0
        exp(lgamma(z+λ₁)-lgamma(z+λ₂))
    else
        gamma(z+λ₁)/gamma(z+λ₂)
    end
end
function Λ(x::Float64, λ₁::Float64, λ₂::Float64)
    if min(x+λ₁,x+λ₂) ≥ 8.979120323411497
        exp(λ₂-λ₁+(x-.5)*log1p((λ₁-λ₂)/(x+λ₂)))*(x+λ₁)^λ₁/(x+λ₂)^λ₂*stirlingseries(x+λ₁)/stirlingseries(x+λ₂)
    else
        (x+λ₂)/(x+λ₁)*Λ(x + 1.0, λ₁, λ₂)
    end
end

## TODO: deprecate when Lambert-W is supported in a mainstream repository such as SpecialFunctions.jl
"""
The principal branch of the Lambert-W function, defined by ``x = W_0(x) e^{W_0(x)}``, computed using Halley's method for ``x \\in [-e^{-1},\\infty)``.
"""
function lambertw(x::AbstractFloat)
    if x < -exp(-one(x))
        return throw(DomainError())
    elseif x == -exp(-one(x))
        return -one(x)
    elseif x < 0
        w0 = ℯ*x/(1+inv(inv(sqrt(2*ℯ*x+2))+inv(ℯ-1)-inv(sqrt(2))))
    else
        log1px = log1p(x)
        w0 = log1px*(1-log1p(log1px)/(2+log1px))
    end
    expw0 = exp(w0)
    w1 = w0 - (w0*expw0 - x)/((w0 + 1)*expw0 -
        (w0 + 2) * (w0*expw0 - x)/(2w0 + 2))
    while abs(w1/w0 - 1) > 2eps(typeof(x))
        w0 = w1
        expw0 = exp(w0)
        w1 = w0 - (w0*expw0 - x)/((w0 + 1)*expw0 -
            (w0 + 2) * (w0*expw0 - x)/(2w0 + 2))
    end
    return w1
end
lambertw(x::Real) = lambertw(float(x))


Cnλ(n::Integer,λ::Float64) = 2^λ/sqrtpi*Λ(n+λ)
Cnλ(n::Integer,λ::Number) = 2^λ/sqrt(oftype(λ,π))*Λ(n+λ)
function Cnλ(n::UnitRange{T},λ::Number) where T<:Integer
    ret = Vector{typeof(λ)}(undef, length(n))
    ret[1] = Cnλ(first(n),λ)
    for i=2:length(n)
        ret[i] = (n[i]+λ-half(λ))/(n[i]+λ)*ret[i-1]
    end
    ret
end

function Cnmλ(n::Integer,m::Integer,λ::Number)
    if m == 0
        Cnλ(n,λ)
    else
        (λ+m-1)/2/m*(m-λ)/(n+λ+m)*Cnmλ(n,m-1,λ)
    end
end


function Cnαβ(n::Integer,α::Number,β::Number)
    if n==0
        2^(α+β+1)*beta(α+1,β+1)/π
    else
        val = Cnαβ(0,α,β)
        for i=1:n
            val *= (i+α)*(i+β)/(i+(α+β+1)/2)/(i+(α+β)/2)
        end
        val
    end
end

function Cnαβ(n::Integer,α::Float64,β::Float64)
    if n+min(α,β) ≥ 7.979120323411497
        stirlingseries(n+α+1)/sqrtpi/stirlingseries(2n+α+β+2)*Cratio(n,α,β)*stirlingseries(n+β+1)
    else
        (n+(α+β+3)/2)/(n+β+1)*(n+(α+β+2)/2)/(n+α+1)*Cnαβ(n+1,α,β)
    end
end

function Cnmαβ(n::Integer,m::Integer,α::Number,β::Number)
    if m == 0
        Cnαβ(n,α,β)
    else
        Cnmαβ(n,m-1,α,β)/2(2n+α+β+m+1)
    end
end


function Cnmαβ(n::Integer,m::Integer,α::AbstractArray{T},β::AbstractArray{T}) where T<:Number
    shp = promote_shape(size(α),size(β))
    reshape([ Cnmαβ(n,m,α[i],β[i]) for i in eachindex(α,β) ], shp)
end


"""
Modified Chebyshev moments of the first kind:

```math
    \\int_{-1}^{+1} T_n(x) {\\rm\\,d}x.
```
"""
function chebyshevmoments1(::Type{T}, N::Int) where T
    μ = zeros(T, N)
    for i = 0:2:N-1
        @inbounds μ[i+1] = two(T)/T(1-i^2)
    end
    μ
end

"""
Modified Chebyshev moments of the first kind with respect to the Jacobi weight:

```math
    \\int_{-1}^{+1} T_n(x) (1-x)^\\alpha(1+x)^\\beta{\\rm\\,d}x.
```
"""
function chebyshevjacobimoments1(::Type{T}, N::Int, α, β) where T
    μ = zeros(T, N)
    N > 0 && (μ[1] = 2 .^ (α+β+1)*beta(α+1,β+1))
    if N > 1
        μ[2] = μ[1]*(β-α)/(α+β+2)
        for i=1:N-2
            @inbounds μ[i+2] = (2(β-α)*μ[i+1]-(α+β-i+2)*μ[i])/(α+β+i+2)
        end
    end
    μ
end

"""
Modified Chebyshev moments of the first kind with respect to the logarithmic weight:

```math
    \\int_{-1}^{+1} T_n(x) \\log\\left(\\frac{1-x}{2}\\right){\\rm\\,d}x.
```
"""
function chebyshevlogmoments1(::Type{T}, N::Int) where T
    μ = zeros(T, N)
    N > 0 && (μ[1] = -two(T))
    if N > 1
        μ[2] = -one(T)
        for i=1:N-2
            cst = isodd(i) ? T(4)/T(i^2-4) : T(4)/T(i^2-1)
            @inbounds μ[i+2] = ((i-2)*μ[i]+cst)/(i+2)
        end
    end
    μ
end

"""
Modified Chebyshev moments of the second kind:

```math
    \\int_{-1}^{+1} U_n(x) {\\rm\\,d}x.
```
"""
function chebyshevmoments2(::Type{T}, N::Int) where T
    μ = zeros(T, N)
    for i = 0:2:N-1
        @inbounds μ[i+1] = two(T)/T(i+1)
    end
    μ
end

"""
Modified Chebyshev moments of the second kind with respect to the Jacobi weight:

```math
    \\int_{-1}^{+1} U_n(x) (1-x)^\\alpha(1+x)^\\beta{\\rm\\,d}x.
```
"""
function chebyshevjacobimoments2(::Type{T}, N::Int, α, β) where T
    μ = zeros(T, N)
    N > 0 && (μ[1] = 2 .^ (α+β+1)*beta(α+1,β+1))
    if N > 1
        μ[2] = 2μ[1]*(β-α)/(α+β+2)
        for i=1:N-2
            @inbounds μ[i+2] = (2(β-α)*μ[i+1]-(α+β-i)*μ[i])/(α+β+i+2)
        end
    end
    μ
end

"""
Modified Chebyshev moments of the second kind with respect to the logarithmic weight:

```math
    \\int_{-1}^{+1} U_n(x) \\log\\left(\\frac{1-x}{2}\\right){\\rm\\,d}x.
```
"""
function chebyshevlogmoments2(::Type{T}, N::Int) where T
    μ = chebyshevlogmoments1(T, N)
    if N > 1
        μ[2] *= two(T)
        for i=1:N-2
            @inbounds μ[i+2] = 2μ[i+2] + μ[i]
        end
    end
    μ
end


function sphrand(::Type{T}, m::Int, n::Int) where T
    A = zeros(T, m, n)
    for i = 1:m
        A[i,1] = rand(T)
    end
    for j = 1:n÷2
        for i = 1:m-j
            A[i,2j] = rand(T)
            A[i,2j+1] = rand(T)
        end
    end
    A
end

function sphrandn(::Type{T}, m::Int, n::Int) where T
    A = zeros(T, m, n)
    for i = 1:m
        A[i,1] = randn(T)
    end
    for j = 1:n÷2
        for i = 1:m-j
            A[i,2j] = randn(T)
            A[i,2j+1] = randn(T)
        end
    end
    A
end

function sphones(::Type{T}, m::Int, n::Int) where T
    A = zeros(T, m, n)
    for i = 1:m
        A[i,1] = one(T)
    end
    for j = 1:n÷2
        for i = 1:m-j
            A[i,2j] = one(T)
            A[i,2j+1] = one(T)
        end
    end
    A
end

sphzeros(::Type{T}, m::Int, n::Int) where T = zeros(T, m, n)

"""
Pointwise evaluation of real orthonormal spherical harmonic:

```math
Y_\\ell^m(\\theta,\\varphi) = (-1)^{|m|}\\sqrt{(\\ell+\\frac{1}{2})\\frac{(\\ell-|m|)!}{(\\ell+|m|)!}} P_\\ell^{|m|}(\\cos\\theta) \\sqrt{\\frac{2-\\delta_{m,0}}{2\\pi}} \\left\\{\\begin{array}{ccc} \\cos m\\varphi & {\\rm for} & m \\ge 0,\\\\ \\sin(-m\\varphi) & {\\rm for} & m < 0.\\end{array}\\right.
```
"""
sphevaluate(θ, φ, L, M) = sphevaluatepi(θ/π, φ/π, L, M)

sphevaluatepi(θ::Number, φ::Number, L::Integer, M::Integer) = sphevaluatepi(θ, L, M)*sphevaluatepi(φ, M)

function sphevaluatepi(θ::Number, L::Integer, M::Integer)
    ret = one(θ)/sqrt(two(θ))
    if M < 0 M = -M end
    c, s = cospi(θ), sinpi(θ)
    for m = 1:M
        ret *= sqrt((m+half(θ))/m)*s
    end
    tc = two(c)*c

    if L == M
        return ret
    elseif L == M+1
        return sqrt(two(θ)*M+3)*c*ret
    else
        temp = ret
        ret *= sqrt(two(θ)*M+3)*c
        for l = M+1:L-1
            ret, temp = (sqrt(l+half(θ))*tc*ret - sqrt((l-M)*(l+M)/(l-half(θ)))*temp)/sqrt((l-M+1)*(l+M+1)/(l+3half(θ))), ret
        end
        return ret
    end
end

sphevaluatepi(φ::Number, M::Integer) = sqrt((two(φ)-δ(M, 0))/(two(φ)*π))*(M ≥ 0 ? cospi(M*φ) : sinpi(-M*φ))

function sphvrand(::Type{T}, m::Int, n::Int) where T
    A = zeros(T, m, n)
    for i = 1:m-1
        A[i,1] = rand(T)
    end
    for j = 1:n÷2
        for i = 1:m-j+1
            A[i,2j] = rand(T)
            A[i,2j+1] = rand(T)
        end
    end
    A
end

function sphvrandn(::Type{T}, m::Int, n::Int) where T
    A = zeros(T, m, n)
    for i = 1:m-1
        A[i,1] = randn(T)
    end
    for j = 1:n÷2
        for i = 1:m-j+1
            A[i,2j] = randn(T)
            A[i,2j+1] = randn(T)
        end
    end
    A
end

function sphvones(::Type{T}, m::Int, n::Int) where T
    A = zeros(T, m, n)
    for i = 1:m-1
        A[i,1] = one(T)
    end
    for j = 1:n÷2
        for i = 1:m-j+1
            A[i,2j] = one(T)
            A[i,2j+1] = one(T)
        end
    end
    A
end

sphvzeros(::Type{T}, m::Int, n::Int) where T = sphzeros(T, m, n)

function diskrand(::Type{T}, m::Int, n::Int) where T
    A = zeros(T, m, n)
    for i = 1:m
        A[i,1] = rand(T)
    end
    for j = 1:n÷2
        for i = 1:m-(j+1)÷2
            A[i,2j] = rand(T)
            A[i,2j+1] = rand(T)
        end
    end
    A
end

function diskrandn(::Type{T}, m::Int, n::Int) where T
    A = zeros(T, m, n)
    for i = 1:m
        A[i,1] = randn(T)
    end
    for j = 1:n÷2
        for i = 1:m-(j+1)÷2
            A[i,2j] = randn(T)
            A[i,2j+1] = randn(T)
        end
    end
    A
end

function diskones(::Type{T}, m::Int, n::Int) where T
    A = zeros(T, m, n)
    for i = 1:m
        A[i,1] = one(T)
    end
    for j = 1:n÷2
        for i = 1:m-(j+1)÷2
            A[i,2j] = one(T)
            A[i,2j+1] = one(T)
        end
    end
    A
end

diskzeros(::Type{T}, m::Int, n::Int) where T = zeros(T, m, n)

function trirand(::Type{T}, m::Int, n::Int) where T
    A = zeros(T, m, n)
    for j = 1:n
        for i = 1:m+1-j
            A[i,j] = rand(T)
        end
    end
    A
end

function trirandn(::Type{T}, m::Int, n::Int) where T
    A = zeros(T, m, n)
    for j = 1:n
        for i = 1:m+1-j
            A[i,j] = randn(T)
        end
    end
    A
end

function triones(::Type{T}, m::Int, n::Int) where T
    A = zeros(T, m, n)
    for j = 1:n
        for i = 1:m+1-j
            A[i,j] = one(T)
        end
    end
    A
end

trizeros(::Type{T}, m::Int, n::Int) where T = zeros(T, m, n)

const rectdiskrand = trirand
const rectdiskrandn = trirandn
const rectdiskones = triones
const rectdiskzeros = trizeros

"""
Pointwise evaluation of triangular harmonic:

```math
\\tilde{P}_{\\ell,m}^{(\\alpha,\\beta,\\gamma)}(x,y).
```
"""
trievaluate(x, y, L, M, α, β, γ) = trievaluate(x, L, M, α, β, γ)*trievaluate(x, y, M, β, γ)

function trievaluate(x::Number, L::Integer, M::Integer, α::Number, β::Number, γ::Number)

end

function trievaluate(x::Number, y::Number, M::Integer, β::Number, γ::Number)

end

function tetrand(::Type{T}, l::Int, m::Int, n::Int) where T
    A = zeros(T, l, m, n)
    for k = 1:n
        for j = 1:m+1-k
            for i = 1:l+2-k-j
                A[i,j,k] = rand(T)
            end
        end
    end
    A
end

function tetrandn(::Type{T}, l::Int, m::Int, n::Int) where T
    A = zeros(T, l, m, n)
    for k = 1:n
        for j = 1:m+1-k
            for i = 1:l+2-k-j
                A[i,j,k] = randn(T)
            end
        end
    end
    A
end

function tetones(::Type{T}, l::Int, m::Int, n::Int) where T
    A = zeros(T, l, m, n)
    for k = 1:n
        for j = 1:m+1-k
            for i = 1:l+2-k-j
                A[i,j,k] = one(T)
            end
        end
    end
    A
end

tetzeros(::Type{T}, l::Int, m::Int, n::Int) where T = zeros(T, l, m, n)

function spinsphrand(::Type{T}, m::Int, n::Int, s::Int) where T
    A = zeros(T, m, n)
    as = abs(s)
    for i = 1:m-as
        A[i,1] = rand(T)
    end
    for j = 1:n÷2
        for i = 1:m-max(j, as)
            A[i,2j] = rand(T)
            A[i,2j+1] = rand(T)
        end
    end
    A
end

function spinsphrandn(::Type{T}, m::Int, n::Int, s::Int) where T
    A = zeros(T, m, n)
    as = abs(s)
    for i = 1:m-as
        A[i,1] = randn(T)
    end
    for j = 1:n÷2
        for i = 1:m-max(j, as)
            A[i,2j] = randn(T)
            A[i,2j+1] = randn(T)
        end
    end
    A
end

function spinsphones(::Type{T}, m::Int, n::Int, s::Int) where T
    A = zeros(T, m, n)
    as = abs(s)
    for i = 1:m-as
        A[i,1] = one(T)
    end
    for j = 1:n÷2
        for i = 1:m-max(j, as)
            A[i,2j] = one(T)
            A[i,2j+1] = one(T)
        end
    end
    A
end

spinsphzeros(::Type{T}, m::Int, n::Int) where T = zeros(T, m, n)
