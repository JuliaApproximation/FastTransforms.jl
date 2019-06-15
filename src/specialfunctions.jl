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

pochhammer(x::AbstractArray{T,1},n::Integer) where {T<:Number} = [pochhammer(x[i],n) for i=1:length(x)]
pochhammer(x::AbstractArray{T,2},n::Integer) where {T<:Number} = [pochhammer(x[i,j],n) for i=1:size(x,1),j=1:size(x,2)]
pochhammer(x::AbstractArray{T},n::Integer) where {T<:Number} = reshape([ pochhammer(x[i],n) for i in eachindex(x) ], size(x))

pochhammer(x::Number,n::Number) = isinteger(n) ? pochhammer(x,Int(n)) : newgamma(x+n)/newgamma(x)
pochhammer(x::AbstractArray{T},n::Number) where {T<:Number} = isinteger(n) ? pochhammer(x,Int(n)) : newgamma.(x.+n)./newgamma.(x)

function pochhammer(x::Number,n::UnitRange{T}) where T<:Real
    ret = Vector{promote_type(typeof(x),T)}(length(n))
    ret[1] = pochhammer(x,first(n))
    for i=2:length(n)
        ret[i] = (x+n[i]-1)*ret[i-1]
    end
    ret
end

function newgamma(x::Number)
    if isinteger(x) && x<0
        Inf
    else
        gamma(x)
    end
end

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
Λ(z::Number) = exp(lgamma(z+half(z))-lgamma(z+one(z)))
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
Λ(z::Number,λ₁::Number,λ₂::Number) = exp(lgamma(z+λ₁)-lgamma(z+λ₂))
function Λ(x::Float64,λ₁::Float64,λ₂::Float64)
    if min(x+λ₁,x+λ₂) ≥ 8.979120323411497
        exp(λ₂-λ₁+(x-.5)*log1p((λ₁-λ₂)/(x+λ₂)))*(x+λ₁)^λ₁/(x+λ₂)^λ₂*stirlingseries(x+λ₁)/stirlingseries(x+λ₂)
    else
        (x+λ₂)/(x+λ₁)*Λ(x+1.,λ₁,λ₂)
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


function absf(α::Number,β::Number,m::Int,θ::Number)
    ret = zero(θ)
    for l=0:m
        ret += pochhammer(half(α)+α,l)*pochhammer(half(α)-α,l)*pochhammer(half(β)+β,m-l)*pochhammer(half(β)-β,m-l)/factorial(l)/factorial(m-l)/sinpi(θ/2)^(l+α+half(α))/cospi(θ/2)^(m-l+β+half(β))
    end
    ret
end

function absf(α::AbstractArray{T},β::AbstractArray{T},m::Int,θ::Number) where T<:Number
    shp = promote_shape(size(α),size(β))
    reshape([ absf(α[i],β[i],m,θ) for i in eachindex(α,β) ], shp)
end


function absf(α::Number,β::Number,m::Int,θ::AbstractArray{T,1}) where T<:Number
    ret = zero(θ)
    cfs = zeros(T,m+1)
    for l=0:m
        @inbounds cfs[l+1] = pochhammer(half(α)+α,l)*pochhammer(half(α)-α,l)*pochhammer(half(β)+β,m-l)*pochhammer(half(β)-β,m-l)/factorial(l)/factorial(m-l)
    end
    @inbounds for i=1:length(θ),l=0:m
        ret[i] += cfs[l+1]/sinpi(θ[i]/2)^(l+α+half(α))/cospi(θ[i]/2)^(m-l+β+half(β))
    end
    ret
end
absf(α::Number,β::Number,m::Int,θ::AbstractArray{T,2}) where {T<:Number} = [ absf(α,β,m,θ[i,j]) for i=1:size(θ,1), j=1:size(θ,2) ]
absf(α::Number,β::Number,m::Int,θ::AbstractArray{T}) where {T<:Number} = reshape([ absf(α,β,m,θ[i]) for i in eachindex(θ) ], size(θ))

function compute_absf!(ret::Vector{T},cfs::Matrix{T},α::T,β::T,tempcos::Vector{T},tempsin::Vector{T},tempcosβsinα::Vector{T},m::Int) where T<:AbstractFloat
    @inbounds for i=1:length(ret)
        temp = inv(tempcos[i]^m*tempcosβsinα[i])
        ret[i] = cfs[m+1,1]*temp
        for l=1:m
            temp *= tempcos[i]/tempsin[i]
            ret[i] += cfs[m+1,l+1]*temp
        end
    end
    ret
end

function compute_absf!(ret::Vector{T},tempsin::Vector{T},tempsinλ::Vector{T},m::Int) where T<:AbstractFloat
    for i=1:length(ret)
        @inbounds ret[i] = inv(tempsin[i]^m*tempsinλ[i])
    end
    ret
end

function compute_umvm!(um::Vector{T},vm::Vector{T},cfs::Matrix{T},α::T,β::T,tempcos::Vector{T},tempsin::Vector{T},tempcosβsinα::Vector{T},m::Int,θ::Vector{T},ir::UnitRange{Int}) where T<:AbstractFloat
    @inbounds for i in ir
        temp = inv(tempcos[i]^m*tempcosβsinα[i])
        ϑ = (α+half(α))/2-(α+β+m+1)*θ[i]/2
        um[i] = cfs[m+1,1]*cospi(ϑ)*temp
        vm[i] = cfs[m+1,1]*sinpi(ϑ)*temp
        @inbounds for l=1:m
            temp *= tempcos[i]/tempsin[i]
            ϑ = (α+l+half(α))/2-(α+β+m+1)*θ[i]/2
            um[i] += cfs[m+1,l+1]*cospi(ϑ)*temp
            vm[i] += cfs[m+1,l+1]*sinpi(ϑ)*temp
        end
    end
end

function compute_umvm!(um::Vector{T},vm::Vector{T},λ::T,tempsinλm::Vector{T},m::Int,θ::Vector{T},ir::UnitRange{Int}) where T<:AbstractFloat
    @inbounds @simd for i in ir
        temp = inv(tempsinλm[i])
        ϑ = (m+λ)*(half(T)-θ[i])
        um[i] = cospi(ϑ)*temp
        vm[i] = sinpi(ϑ)*temp
    end
end

function findmindices!(Rαβjm::Vector{T},cfs::Matrix{T},α::T,β::T,j::Int,m::Int,tempcos::Vector{T},tempsin::Vector{T},tempcosβsinα::Vector{T}) where T<:AbstractFloat
    compute_absf!(Rαβjm,cfs,α,β,tempcos,tempsin,tempcosβsinα,m)
    rmul!(Rαβjm,Cnmαβ(j,m,α,β))
    rmin,imin = findmin(Rαβjm)
    if rmin < eps(T)
        i₁ = imin-1
        while i₁ ≥ 3
            if Rαβjm[i₁] < eps(T)
                i₁-=1
            else
                break
            end
        end
        i₂ = imin+1
        while i₂ ≤ length(Rαβjm)-2
            if Rαβjm[i₂] < eps(T)
                i₂+=1
            else
                break
            end
        end
        return i₁,i₂
    else
        return 1,0# error("There are no indices such that the interior asymptotic formula is valid.")# but adding an error is type-unstable :(.
    end
end

function findmindices!(Rαβjm::Vector{T},λ::T,j::Int,m::Int,tempsin::Vector{T},tempsinλ::Vector{T}) where T<:AbstractFloat
    compute_absf!(Rαβjm,tempsin,tempsinλ,m)
    rmul!(Rαβjm,Cnmλ(j,m,λ))
    rmin,imin = findmin(Rαβjm)
    if rmin < eps(T)
        i₁ = imin-1
        while i₁ ≥ 3
            if Rαβjm[i₁] < eps(T)
                i₁-=1
            else
                break
            end
        end
        i₂ = imin+1
        while i₂ ≤ length(Rαβjm)-2
            if Rαβjm[i₂] < eps(T)
                i₂+=1
            else
                break
            end
        end
        return i₁,i₂
    else
        return 1,0# error("There are no indices such that the interior asymptotic formula is valid.")# but adding an error is type-unstable :(.
    end
end

# initialization methods

function init_cfs(α::T,β::T,M::Int) where T<:AbstractFloat
    cfs = zeros(T,M+1,M+1)
    @inbounds for m=0:M,l=0:m
        cfs[m+1,l+1] = pochhammer(half(α)+α,l)*pochhammer(half(α)-α,l)*pochhammer(half(β)+β,m-l)*pochhammer(half(β)-β,m-l)/factorial(l)/factorial(m-l)
    end
    cfs
end

function init_c₁c₂!(c₁::Vector,c₂::Vector,a::Vector,b::Vector,j₁::Int,j₂::Int)
    @inbounds for j=1:j₁-1 c₁[j] = 0 end
    @inbounds for j=j₁:j₂ c₁[j] = a[j]*b[j] end
    @inbounds for j=j₂+1:length(c₁) c₁[j] = 0 end
    copyto!(c₂,c₁)
end

function init_c₁c₂!(c₁::Vector,c₂::Vector,u::Vector,v::Vector,c::Vector,i₁::Int,i₂::Int)
    @inbounds for i=1:i₁-1
        c₁[i] = 0
        c₂[i] = 0
    end
    @inbounds for i=i₁:i₂
        c₁[i] = u[i]*c[i]
        c₂[i] = v[i]*c[i]
    end
    @inbounds for i=i₂+1:length(c₁)
        c₁[i] = 0
        c₂[i] = 0
    end
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

"""
Compute Jacobi expansion coefficients in ``P_n^{(\\alpha+1,\\beta)}(x)`` given Jacobi expansion coefficients in ``P_n^{(\\alpha,\\beta)}(x)`` in-place.
"""
function incrementα!(c::AbstractVector,α,β)
    αβ,N = α+β,length(c)
    N > 1 && (c[1] -= (β+1)/(αβ+3)*c[2])
    @inbounds for i=2:N-1 c[i] = (αβ+i)/(αβ+2i-1)*c[i] - (β+i)/(αβ+2i+1)*c[i+1] end
    N > 1 && (c[N] *= (αβ+N)/(αβ+2N-1))
    c
end

"""
Compute Jacobi expansion coefficients in ``P_n^{(\\alpha,\\beta+1)}(x)`` given Jacobi expansion coefficients in ``P_n^{(\\alpha,\\beta)}(x)`` in-place.
"""
function incrementβ!(c::AbstractVector,α,β)
    αβ,N = α+β,length(c)
    N > 1 && (c[1] += (α+1)/(αβ+3)*c[2])
    @inbounds for i=2:N-1 c[i] = (αβ+i)/(αβ+2i-1)*c[i] + (α+i)/(αβ+2i+1)*c[i+1] end
    N > 1 && (c[N] *= (αβ+N)/(αβ+2N-1))
    c
end

"""
Compute Jacobi expansion coefficients in ``P_n^{(\\alpha+1,\\alpha+1)}(x)`` given Jacobi expansion coefficients in ``P_n^{(\\alpha,\\alpha)}(x)`` in-place.
"""
function incrementαβ!(c::AbstractVector,α,β)
    @assert α == β
    N = length(c)
    if N == 2
        c[2] *= (2α+2)/(2α+4)
    elseif N > 2
        c[1] -= (α+2)/(4α+10)*c[3]
        @inbounds for i=2:N-2 c[i] = (2α+i)*(2α+i+1)/(2α+2i-1)/(2α+2i)*c[i] - (α+i+1)/(4α+4i+6)*c[i+2] end
        c[N-1] *= (2α+N-1)*(2α+N)/(2α+2N-3)/(2α+2N-2)
        c[N] *= (2α+N)*(2α+N+1)/(2α+2N-1)/(2α+2N)
    end
    c
end

"""
Compute Jacobi expansion coefficients in ``P_n^{(\\alpha-1,\\beta)}(x)`` given Jacobi expansion coefficients in ``P_n^{(\\alpha,\\beta)}(x)`` in-place.
"""
function decrementα!(c::AbstractVector,α,β)
    αβ,N = α+β,length(c)
    N > 1 && (c[N] *= (αβ+2N-2)/(αβ+N-1))
    @inbounds for i=N-1:-1:2 c[i] = (αβ+2i-2)/(αβ+i-1)*c[i] + (αβ+2i-2)/(αβ+2i)*(β+i)/(αβ+i-1)*c[i+1] end
    N > 1 && (c[1] += (β+1)/(αβ+2)*c[2])
    c
end

"""
Compute Jacobi expansion coefficients in ``P_n^{(\\alpha,\\beta-1)}(x)`` given Jacobi expansion coefficients in ``P_n^{(\\alpha,\\beta)}(x)`` in-place.
"""
function decrementβ!(c::AbstractVector,α,β)
    αβ,N = α+β,length(c)
    N > 1 && (c[N] *= (αβ+2N-2)/(αβ+N-1))
    @inbounds for i=N-1:-1:2 c[i] = (αβ+2i-2)/(αβ+i-1)*c[i] - (αβ+2i-2)/(αβ+2i)*(α+i)/(αβ+i-1)*c[i+1] end
    N > 1 && (c[1] -= (α+1)/(αβ+2)*c[2])
    c
end

"""
Compute Jacobi expansion coefficients in ``P_n^{(\\alpha-1,\\alpha-1)}(x)`` given Jacobi expansion coefficients in ``P_n^{(\\alpha,\\alpha)}(x)`` in-place.
"""
function decrementαβ!(c::AbstractVector,α,β)
    @assert α == β
    N = length(c)
    if N == 2
        c[2] *= (2α+2)/(2α)
    elseif N > 2
        c[N] *= (2α+2N-3)*(2α+2N-2)/(2α+N-2)/(2α+N-1)
        c[N-1] *= (2α+2N-5)*(2α+2N-4)/(2α+N-3)/(2α+N-2)
        @inbounds for i=N-2:-1:2 c[i] = (2α+2i-3)*(2α+2i-2)/(2α+i-2)/(2α+i-1)*(c[i] + (α+i)/(4α+4i+2)*c[i+2]) end
        c[1] += (α+1)/(4α+6)*c[3]
    end
    c
end


function modαβ(α)
    if -0.5 < α ≤ 0.5
        a = α
    else
        a = α%1
        a > 0.5 && (a-=1)
        a ≤ -0.5 && (a+=1)
    end
    a
end

function modλ(λ)
    if 0 ≤ λ < 1
        l = λ
    else
        l = λ%1
        l < 0 && (l+=1)
    end
    l
end

function tosquare!(ret::AbstractVector,α,β)
    a,b = modαβ(α),modαβ(β)
    A,B = α-a,β-b
    if α ≤ -0.5 && β ≤ -0.5
        incrementα!(ret,α,β)
        incrementβ!(ret,a,β)
    elseif α ≤ -0.5
        incrementα!(ret,α,β)
        for j=B:-1:1
            decrementβ!(ret,a,j+b)
        end
    elseif β ≤ -0.5
        incrementβ!(ret,α,β)
        for i=A:-1:1
            decrementα!(ret,i+a,b)
        end
    else
        for i=A:-1:1
            decrementα!(ret,i+a,β)
        end
        for j=B:-1:1
            decrementβ!(ret,a,j+b)
        end
    end
    ret
end

function fromsquare!(ret::AbstractVector,α,β)
    a,b = modαβ(α),modαβ(β)
    A,B = α-a,β-b
    if α ≤ -0.5 && β ≤ -0.5
        decrementα!(ret,a,b)
        decrementβ!(ret,α,b)
    elseif α ≤ -0.5
        decrementα!(ret,a,b)
        for j=0:B-1
            incrementβ!(ret,α,j+b)
        end
    elseif β ≤ -0.5
        decrementβ!(ret,a,b)
        for i=0:A-1
            incrementα!(ret,i+a,β)
        end
    else
        for i=0:A-1
            incrementα!(ret,i+a,b)
        end
        for j=0:B-1
            incrementβ!(ret,α,j+b)
        end
    end
    ret
end


function toline!(ret::AbstractVector,α,β)
    @assert α == β
    a,b = modαβ(α),modαβ(β)
    A,B = α-a,β-b
    if α ≤ -0.5
        incrementαβ!(ret,α,β)
    else
        for i=A:-1:1
            decrementαβ!(ret,i+a,i+a)
        end
    end
    ret
end

function fromline!(ret::AbstractVector,α,β)
    @assert α == β
    a,b = modαβ(α),modαβ(β)
    A,B = α-a,β-b
    if α ≤ -0.5
        decrementαβ!(ret,a,b)
    else
        for i=0:A-1
            incrementαβ!(ret,i+a,i+a)
        end
    end
    ret
end
