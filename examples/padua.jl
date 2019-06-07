#############
# This demonstrates the Padua transform and inverse transform,
# explaining precisely the normalization and points
#############

using FastTransforms

N = 15
pts = paduapoints(N)
x = pts[:,1]; y = pts[:,2]

f = (x,y) -> exp(x + cos(y))
f̌ = paduatransform(f.(x , y))
f̃ = (x,y) -> begin
    j = 1
    ret = 0.0
    for n in 0:N, k in 0:n
        ret += f̌[j]*cos((n-k)*acos(x)) * cos(k*acos(y))
        j += 1
    end
    ret
end

f̃(0.1,0.2) ≈ f(0.1,0.2)

ipaduatransform(f̌) ≈ f̃.(x,y)
