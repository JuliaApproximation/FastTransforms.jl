# # Half-range Chebyshev polynomials
# In [this paper](https://doi.org/10.1137/090752456), [Daan Huybrechs](https://github.com/daanhb) introduced the so-called half-range Chebyshev polynomials
# as the semi-classical orthogonal polynomials with respect to the inner product:
# ```math
# \langle f, g \rangle = \int_0^1 f(x) g(x)\frac{{\rm d} x}{\sqrt{1-x^2}}.
# ```
# By the variable transformation $y = 2x-1$, the resulting polynomials can be related to
# orthogonal polynomials on $(-1,1)$ with the Jacobi weight $(1-y)^{-\frac{1}{2}}$ modified by the weight $(3+y)^{-\frac{1}{2}}$.
#
# We shall use the fact that:
# ```math
# \frac{1}{\sqrt{3+y}} = \sqrt{\frac{2}{3+\sqrt{8}}}\sum_{n=0}^\infty P_n(y) \left(\frac{-1}{3+\sqrt{8}}\right)^n,
# ```
# and results from [this paper](https://arxiv.org/abs/2302.08448) to consider the half-range Chebyshev polynomials as
# modifications of the Jacobi polynomials $P_n^{(-\frac{1}{2},0)}(y)$.

using FastTransforms, LinearAlgebra, Plots, LaTeXStrings
const GENFIGS = joinpath(pkgdir(FastTransforms), "docs/src/generated")
!isdir(GENFIGS) && mkdir(GENFIGS)
plotlyjs()

# We truncate the generating function to ensure a relative error less than `eps()` in the uniform norm on $(-1,1)$:
z = -1/(3+sqrt(8))
K = sqrt(-2z)
N = ceil(Int, log(abs(z), eps()/2*(1-abs(z))/K) - 1)
d = K .* z .^(0:N)

# Then, we convert this representation to the expansion in Jacobi polynomials $P_n^{(-\frac{1}{2}, 0)}(y)$:
u = jac2jac(d, 0.0, 0.0, -0.5, 0.0; norm1 = false, norm2 = true)

# Our working polynomial degree will be:
n = 100

# We compute the connection coefficients between the modified orthogonal polynomials and the Jacobi polynomials:
P = plan_modifiedjac2jac(Float64, n+1, -0.5, 0.0, u)

# We store the connection to first kind Chebyshev polynomials:
P1 = plan_jac2cheb(Float64, n+1, -0.5, 0.0; normjac = true)

# We compute the Chebyshev series for the degree-$k\le n$ modified polynomial and its values at the Chebyshev points:
q = k -> lmul!(P1, lmul!(P, [zeros(k); 1.0; zeros(n-k)]))
qvals = k-> ichebyshevtransform(q(k))

# With the symmetric Jacobi matrix for $P_n^{(-\frac{1}{2}, 0)}(y)$ and the modified plan, we may compute the modified Jacobi matrix and the corresponding roots (as eigenvalues):
XP = SymTridiagonal([-inv((4n-1)*(4n-5)) for n in 1:n+1], [4n*(2n-1)/(4n-1)/sqrt((4n-3)*(4n+1)) for n in 1:n])
XQ = FastTransforms.modified_jacobi_matrix(P, XP);

# And we plot:
x = (chebyshevpoints(Float64, n+1, Val(1)) .+ 1 ) ./ 2
p = plot(x, qvals(0); linewidth=2.0, legend = false, xlim=(0,1), xlabel=L"x",
         ylabel=L"T^h_n(x)", title="Half-Range Chebyshev Polynomials and Their Roots",
         extra_plot_kwargs = KW(:include_mathjax => "cdn"))
for k in 1:10
    λ = (eigvals(SymTridiagonal(XQ.dv[1:k], XQ.ev[1:k-1])) .+ 1) ./ 2
    plot!(x, qvals(k); linewidth=2.0, color=palette(:default)[k+1])
    scatter!(λ, zero(λ); markersize=2.5, color=palette(:default)[k+1])
end
p
savefig(joinpath(GENFIGS, "halfrange.html"))
###```@raw html
###<object type="text/html" data="../halfrange.html" style="width:100%;height:400px;"></object>
###```

# By [Theorem 2.20](https://arxiv.org/abs/2302.08448) it turns out that the *derivatives* of the half-range Chebyshev polynomials are a linear combination of at most two polynomials orthogonal with respect to $\sqrt{(3+y)(1-y)}(1+y)$ on $(-1,1)$. This fact enables us to compute the banded differentiation matrix:
v̂ = 3*u+XP[1:N+1,1:N]*u
v = jac2jac(v̂, -0.5, 0.0, 0.5, 1.0; norm1 = true, norm2 = true)
function threshold!(A::AbstractArray, ϵ)
    for i in eachindex(A)
        if abs(A[i]) < ϵ A[i] = 0 end
    end
    A
end
P′ = plan_modifiedjac2jac(Float64, n+1, 0.5, 1.0, v)
DP = UpperTriangular(diagm(1=>[sqrt(n*(n+1/2)) for n in 1:n])) # The classical differentiation matrix representing 𝒟 P^{(-1/2,0)}(y) = P^{(1/2,1)}(y) D_P.
DQ = UpperTriangular(threshold!(P′\(DP*(P*I)), 100eps())) # The semi-classical differentiation matrix representing 𝒟 Q(y) = Q̂(y) D_Q.
