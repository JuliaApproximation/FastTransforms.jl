#############
# This demonstrates the Triangle harmonic transform and inverse transform,
# explaining precisely the normalization and points
#
# Note we use the duffy map
# x == (s+1)/2
# y== (t+1)/2*(1-(s+1)/2)
#############


using ApproxFun, FastTransforms

jacobinorm(n,a,b) = if n ≠ 0
        sqrt((2n+a+b+1))*exp((lgamma(n+a+b+1)+lgamma(n+1)-log(2)*(a+b+1)-lgamma(n+a+1)-lgamma(n+b+1))/2)
    else
        sqrt(exp(lgamma(a+b+2)-log(2)*(a+b+1)-lgamma(a+1)-lgamma(b+1)))
    end
njacobip(n,a,b,x) = jacobinorm(n,a,b) * jacobip(n,a,b,x)

P = (ℓ,m,x,y) -> (2*(1-x))^m*njacobip(ℓ-m,2m,0,2x-1)*njacobip(m,-0.5,-0.5,2y/(1-x)-1)



p_T = chebyshevpoints(40)
f = (x,y) -> exp(x + cos(y))
    f̃ = (s,t) -> f((s+1)/2, (t+1)/2*(1-(s+1)/2))

    F = f̃.(p_T, p_T')
        for j = 1:size(F,2)
            F[:,j] = chebyshevtransform(F[:,j])
        end
        for k = 1:size(F,1)
            F[k,:] = chebyshevtransform(F[k,:])
        end

    F̌ = cheb2tri(F, 0.0, -0.5, -0.5)


f̃ = function(x,y)
        ret = 0.0
        for j=1:size(F,2), k=1:size(F,1)-j+1
            ret += F̌[k,j] * P(k+j-2,j-1,x,y)
        end
        ret
    end

f̃(0.1,0.2) ≈ f(0.1,0.2)
