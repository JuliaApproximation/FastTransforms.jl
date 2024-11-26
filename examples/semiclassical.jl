# # Semi-classical Jacobi polynomials
# In this example, we will consider the semi-classical orthogonal polynomials with respect to the inner product:
# ```math
# \langle f, g \rangle = \int_{-1}^1 f(x) g(x) w(x){\rm d} x,
# ```
# where $w(x) = w^{(\alpha,\beta,\gamma,\delta,\epsilon)}(x) = (1-x)^\alpha(1+x)^\beta(2+x)^\gamma(3+x)^\delta(5-x)^\epsilon$ is a modification of the Jacobi weight.
# We shall use results from [this paper](https://arxiv.org/abs/2302.08448) to consider these semi-classical orthogonal polynomials as modifications of the orthonormalized Jacobi polynomials $\tilde{P}_n^{(\alpha,\beta)}(x)$.

using ApproxFun, FastTransforms, LazyArrays, LinearAlgebra, Plots, LaTeXStrings
const GENFIGS = joinpath(pkgdir(FastTransforms), "docs/src/generated")
!isdir(GENFIGS) && mkdir(GENFIGS)
plotlyjs()

# We set the five parameters:
α,β,γ,δ,ϵ = -0.125, -0.25, 0.123, 0.456, 0.789

# We use `ApproxFun` to construct a finite normalized Jacobi series as a proxy for $(2+x)^\gamma(3+x)^\delta(5-x)^\epsilon$.
u = Fun(x->(2+x)^γ*(3+x)^δ*(5-x)^ϵ, NormalizedJacobi(β, α))

# Our working polynomial degree will be:
n = 100

# We compute the connection coefficients between the modified orthogonal polynomials and the Jacobi polynomials:
P = plan_modifiedjac2jac(Float64, n+1, α, β, u.coefficients)

# We store the connection to first kind Chebyshev polynomials:
P1 = plan_jac2cheb(Float64, n+1, α, β; normjac = true)

# We compute the Chebyshev series for the degree-$k\le n$ modified polynomial and its values at the Chebyshev points:
q = k -> lmul!(P1, lmul!(P, [zeros(k); 1.0; zeros(n-k)]))
qvals = k -> ichebyshevtransform(q(k))

# With the symmetric Jacobi matrix for $\tilde{P}_n^{(\alpha, \beta)}(x)$ and the modified plan, we may compute the modified Jacobi matrix and the corresponding roots (as eigenvalues):
x = Fun(x->x, NormalizedJacobi(β, α))
XP = SymTridiagonal(Symmetric(Multiplication(x, space(x))[1:n+1, 1:n+1]))
XQ = FastTransforms.modified_jacobi_matrix(P, XP)
view(XQ, 1:7, 1:7)

# And we plot:
x = chebyshevpoints(Float64, n+1, Val(1))
p = plot(x, qvals(0); linewidth=2.0, legend = false, xlim=(-1,1), xlabel=L"x",
         ylabel=L"Q_n(x)", title="Semi-classical Jacobi Polynomials and Their Roots",
         extra_plot_kwargs = KW(:include_mathjax => "cdn"))
for k in 1:10
    λ = eigvals(SymTridiagonal(XQ.dv[1:k], XQ.ev[1:k-1]))
    plot!(x, qvals(k); linewidth=2.0, color=palette(:default)[k+1])
    scatter!(λ, zero(λ); markersize=2.5, color=palette(:default)[k+1])
end
p
savefig(joinpath(GENFIGS, "semiclassical.html"))
###```@raw html
###<object type="text/html" data="../semiclassical.html" style="width:100%;height:400px;"></object>
###```

# By [Theorem 2.20](https://arxiv.org/abs/2302.08448) it turns out that the *derivatives* of these particular semi-classical Jacobi polynomials are a linear combination of at most four polynomials orthogonal with respect to the weight $w^{(\alpha+1,\beta+1,\gamma+1,\delta+1,\epsilon+1)}(x)$ on $(-1,1)$. This fact enables us to compute the banded differentiation matrix:
v = Fun(x->(2+x)^(γ+1)*(3+x)^(δ+1)*(5-x)^(ϵ+1), NormalizedJacobi(β+1, α+1))
function threshold!(A::AbstractArray, ϵ)
    for i in eachindex(A)
        if abs(A[i]) < ϵ A[i] = 0 end
    end
    A
end
P′ = plan_modifiedjac2jac(Float64, n+1, α+1, β+1, v.coefficients)
DP = UpperTriangular(diagm(1=>[sqrt(n*(n+α+β+1)) for n in 1:n])) # The classical differentiation matrix representing 𝒟 P^{(α,β)}(x) = P^{(α+1,β+1)}(x) D_P.
DQ = UpperTriangular(threshold!(P′\(DP*(P*I)), 100eps())) # The semi-classical differentiation matrix representing 𝒟 Q(x) = Q̂(x) D_Q.
UpperTriangular(DQ[1:9, 1:9])

# A faster method now exists via the `GramMatrix` architecture and its associated displacement equation. Given the modified orthogonal polynomial moments implied by the normalized Jacobi series for $u(x)$, we pad this vector to the necessary size and construct the `GramMatrix` with these moments, the multiplication operator, and the constant $\tilde{P}_0^{(\alpha,\beta)}(x)$:
μ = PaddedVector(u.coefficients, 2n+1)
x = Fun(x->x, NormalizedJacobi(β, α))
XP2 = SymTridiagonal(Symmetric(Multiplication(x, space(x))[1:2n+1, 1:2n+1]))
p0 = Fun(NormalizedJacobi(β, α), [1])(0)
G = GramMatrix(μ, XP2, p0)
view(G, 1:7, 1:7)

# And compute its cholesky factorization. The upper-triangular Cholesky factor represents the connection between original Jacobi and semi-classical Jacobi as ${\bf P}^{(\alpha,\beta)}(x) = {\bf Q}(x) R$.
R = cholesky(G).U
UpperTriangular(view(R, 1:7, 1:7))

# Every else works almost as before, including evaluation on a Chebyshev grid:
q = k -> lmul!(P1, ldiv!(R, [zeros(k); 1.0; zeros(n-k)]))
qvals = k -> ichebyshevtransform(q(k))

# Computation of the modified Jacobi matrix:
XQ1 = FastTransforms.modified_jacobi_matrix(R, XP)
norm(XQ-XQ1)/norm(XQ)

# Plotting:
x = chebyshevpoints(Float64, n+1, Val(1))
p = plot(x, qvals(0); linewidth=2.0, legend = false, xlim=(-1,1), xlabel=L"x",
         ylabel=L"Q_n(x)", title="Semi-classical Jacobi Polynomials and Their Roots",
         extra_plot_kwargs = KW(:include_mathjax => "cdn"))
for k in 1:10
    λ = eigvals(SymTridiagonal(XQ1.dv[1:k], XQ1.ev[1:k-1]))
    plot!(x, qvals(k); linewidth=2.0, color=palette(:default)[k+1])
    scatter!(λ, zero(λ); markersize=2.5, color=palette(:default)[k+1])
end
p
savefig(joinpath(GENFIGS, "semiclassical1.html"))
###```@raw html
###<object type="text/html" data="../semiclassical1.html" style="width:100%;height:400px;"></object>
###```

# And banded differentiation:
μ′ = PaddedVector(v.coefficients, 2n+1)
x′ = Fun(x->x, NormalizedJacobi(β+1, α+1))
XP′ = SymTridiagonal(Symmetric(Multiplication(x′, space(x′))[1:2n+1, 1:2n+1]))
p0′ = Fun(NormalizedJacobi(β+1, α+1), [1])(0)
G′ = GramMatrix(μ′, XP′, p0′)
R′ = cholesky(G′).U
DP = UpperTriangular(diagm(1=>[sqrt(n*(n+α+β+1)) for n in 1:n])) # The classical differentiation matrix representing 𝒟 P^{(α,β)}(x) = P^{(α+1,β+1)}(x) D_P.
DQ = UpperTriangular(threshold!(R′*(DP*(R\I)), 100eps())) # The semi-classical differentiation matrix representing 𝒟 Q(x) = Q̂(x) D_Q.
UpperTriangular(DQ[1:9, 1:9])
