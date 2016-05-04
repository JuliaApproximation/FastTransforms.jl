# API-INDEX


## MODULE: FastTransforms

---

## Methods [Exported]

[cjt(c,  α,  β)](FastTransforms.md#method__cjt.1)  Computes the Chebyshev expansion coefficients

[gaunt(m::Int64,  n::Int64,  μ::Int64,  ν::Int64)](FastTransforms.md#method__gaunt.1)  Calculates the Gaunt coefficients in 64-bit floating-point arithmetic.

[gaunt{T}(::Type{T},  m::Int64,  n::Int64,  μ::Int64,  ν::Int64)](FastTransforms.md#method__gaunt.2)  Calculates the Gaunt coefficients, defined by:

[icjt(c,  α,  β)](FastTransforms.md#method__icjt.1)  Computes the Jacobi expansion coefficients with parameters ``α`` and ``β``

[jjt(c,  α,  β,  γ,  δ)](FastTransforms.md#method__jjt.1)  Computes the Jacobi expansion coefficients with parameters ``γ`` and ``δ``

[plan_cjt(c::AbstractArray{T, 1},  α,  β)](FastTransforms.md#method__plan_cjt.1)  Pre-plan optimized DCT-I and DST-I plans and pre-allocate the necessary

[plan_icjt(c::AbstractArray{T, 1},  α,  β)](FastTransforms.md#method__plan_icjt.1)  Pre-plan optimized DCT-I and DST-I plans and pre-allocate the necessary

---

## Methods [Internal]

[chebyshevjacobimoments1{T<:AbstractFloat}(N::Int64,  α::T<:AbstractFloat,  β::T<:AbstractFloat)](FastTransforms.md#method__chebyshevjacobimoments1.1)  Modified Chebyshev moments of the first kind with respect to the Jacobi weight:

[chebyshevjacobimoments2{T<:AbstractFloat}(N::Int64,  α::T<:AbstractFloat,  β::T<:AbstractFloat)](FastTransforms.md#method__chebyshevjacobimoments2.1)  Modified Chebyshev moments of the second kind with respect to the Jacobi weight:

[clenshawcurtisweights{T<:AbstractFloat}(N::Int64,  α::T<:AbstractFloat,  β::T<:AbstractFloat)](FastTransforms.md#method__clenshawcurtisweights.1)  Compute weights of the Clenshaw—Curtis quadrature rule with a Jacobi weight.

[clenshawcurtis{T<:AbstractFloat}(N::Int64,  α::T<:AbstractFloat,  β::T<:AbstractFloat)](FastTransforms.md#method__clenshawcurtis.1)  Compute nodes and weights of the Clenshaw—Curtis quadrature rule with a Jacobi weight.

[decrementα!(c::AbstractArray{T, 1},  α,  β)](FastTransforms.md#method__decrement945.1)  Compute Jacobi expansion coefficients in Pₙ^(α-1,β) given Jacobi expansion coefficients in Pₙ^(α,β) in-place.

[decrementαβ!(c::AbstractArray{T, 1},  α,  β)](FastTransforms.md#method__decrement945946.1)  Compute Jacobi expansion coefficients in Pₙ^(α-1,α-1) given Jacobi expansion coefficients in Pₙ^(α,α) in-place.

[decrementβ!(c::AbstractArray{T, 1},  α,  β)](FastTransforms.md#method__decrement946.1)  Compute Jacobi expansion coefficients in Pₙ^(α,β-1) given Jacobi expansion coefficients in Pₙ^(α,β) in-place.

[fejer1{T<:AbstractFloat}(N::Int64,  α::T<:AbstractFloat,  β::T<:AbstractFloat)](FastTransforms.md#method__fejer1.1)  Compute nodes and weights of Fejer's first quadrature rule with a Jacobi weight.

[fejer2{T<:AbstractFloat}(N::Int64,  α::T<:AbstractFloat,  β::T<:AbstractFloat)](FastTransforms.md#method__fejer2.1)  Compute nodes and weights of Fejer's second quadrature rule with a Jacobi weight.

[fejerweights1{T<:AbstractFloat}(N::Int64,  α::T<:AbstractFloat,  β::T<:AbstractFloat)](FastTransforms.md#method__fejerweights1.1)  Compute weights of Fejer's first quadrature rule with a Jacobi weight.

[fejerweights2{T<:AbstractFloat}(N::Int64,  α::T<:AbstractFloat,  β::T<:AbstractFloat)](FastTransforms.md#method__fejerweights2.1)  Compute weights of Fejer's second quadrature rule with a Jacobi weight.

[half(x::Number)](FastTransforms.md#method__half.1)  Compute a typed 0.5.

[incrementα!(c::AbstractArray{T, 1},  α,  β)](FastTransforms.md#method__increment945.1)  Compute Jacobi expansion coefficients in Pₙ^(α+1,β) given Jacobi expansion coefficients in Pₙ^(α,β) in-place.

[incrementαβ!(c::AbstractArray{T, 1},  α,  β)](FastTransforms.md#method__increment945946.1)  Compute Jacobi expansion coefficients in Pₙ^(α+1,α+1) given Jacobi expansion coefficients in Pₙ^(α,α) in-place.

[incrementβ!(c::AbstractArray{T, 1},  α,  β)](FastTransforms.md#method__increment946.1)  Compute Jacobi expansion coefficients in Pₙ^(α,β+1) given Jacobi expansion coefficients in Pₙ^(α,β) in-place.

[pochhammer(x::Number,  n::Integer)](FastTransforms.md#method__pochhammer.1)  Pochhammer symbol (x)_n = Γ(x+n)/Γ(x) for the rising factorial.

[stirlingseries(z)](FastTransforms.md#method__stirlingseries.1)  Stirling series for Γ(z).

[two(x::Number)](FastTransforms.md#method__two.1)  Compute a typed 2.

[Λ(x::Float64)](FastTransforms.md#method__923.1)  For 64-bit floating-point arithmetic, the Lambda function uses the asymptotic series for τ in Appendix B of

[Λ(x::Number)](FastTransforms.md#method__923.2)  The Lambda function Λ(z) = Γ(z+½)/Γ(z+1) for the ratio of gamma functions.

[δ(k::Integer,  j::Integer)](FastTransforms.md#method__948.1)  The Kronecker δ function.

