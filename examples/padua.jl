# # Padua transform
# This demonstrates the Padua transform and inverse transform,
# explaining precisely the normalization and points

using FastTransforms

# We define the Padua points and extract Cartesian components:
N = 15
pts = paduapoints(N)
x = pts[:,1];
y = pts[:,2];

# We take the Padua transform of the function:
f = (x,y) -> exp(x + cos(y))
f̌ = paduatransform(f.(x , y));

# and use the coefficients to create an approximation to the function $f$:
f̃ = (x,y) -> begin
    j = 1
    ret = 0.0
    for n in 0:N, k in 0:n
        ret += f̌[j]*cos((n-k)*acos(x)) * cos(k*acos(y))
        j += 1
    end
    ret
end

# At a particular point, is the function well-approximated?
f̃(0.1,0.2) ≈ f(0.1,0.2)

# Does the inverse transform bring us back to the grid?
ipaduatransform(f̌) ≈ f̃.(x,y)
