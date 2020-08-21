function threshold!(A::AbstractArray, ϵ)
    for i in eachindex(A)
        if abs(A[i]) < ϵ A[i] = 0 end
    end
    A
end

using FastTransforms, LinearAlgebra, Random, Test

# The colatitudinal grid (mod π):
N = 10
θ = (0.5:N-0.5)/N

# The longitudinal grid (mod π):
M = 2*N-1
φ = (0:M-1)*2/M

x = [cospi(φ)*sinpi(θ) for θ in θ, φ in φ]
y = [sinpi(φ)*sinpi(θ) for θ in θ, φ in φ]
z = [cospi(θ) for θ in θ, φ in φ]

P = plan_sph2fourier(Float64, N)
PA = plan_sph_analysis(Float64, N, M)
J = FastTransforms.plan_sph_isometry(Float64, N)


f = (x, y, z) -> x^2+y^4+x^2*y*z^3-x*y*z^2


F = f.(x, y, z)
V = PA*F
U = threshold!(P\V, 100eps())
FastTransforms.execute_sph_ZY_axis_exchange!(J, U)
FR = f.(x, -z, -y)
VR = PA*FR
UR = threshold!(P\VR, 100eps())
@test U ≈ UR
norm(U-UR)


α, β, γ = 0.123, 0.456, 0.789

# Isometry built up from ZYZR
A = [cos(α) -sin(α) 0; sin(α) cos(α) 0; 0 0 1]
B = [cos(β) 0 -sin(β); 0 1 0; sin(β) 0 cos(β)]
C = [cos(γ) -sin(γ) 0; sin(γ) cos(γ) 0; 0 0 1]
R = diagm([1, 1, 1.0])
Q = A*B*C*R

# Transform the sampling grid. Note that `Q` is transposed here.
u = Q[1,1]*x + Q[2,1]*y + Q[3,1]*z
v = Q[1,2]*x + Q[2,2]*y + Q[3,2]*z
w = Q[1,3]*x + Q[2,3]*y + Q[3,3]*z

F = f.(x, y, z)
V = PA*F
U = threshold!(P\V, 100eps())
FastTransforms.execute_sph_rotation!(J, α, β, γ, U)
FR = f.(u, v, w)
VR = PA*FR
UR = threshold!(P\VR, 100eps())
@test U ≈ UR
norm(U-UR)


F = f.(x, y, z)
V = PA*F
U = threshold!(P\V, 100eps())
FastTransforms.execute_sph_polar_reflection!(U)
FR = f.(x, y, -z)
VR = PA*FR
UR = threshold!(P\VR, 100eps())
@test U ≈ UR
norm(U-UR)


# Isometry built up from planar reflection
W = [0.123, 0.456, 0.789]
H = w -> I - 2/(w'w)*w*w'
Q = H(W)

# Transform the sampling grid. Note that `Q` is transposed here.
u = Q[1,1]*x + Q[2,1]*y + Q[3,1]*z
v = Q[1,2]*x + Q[2,2]*y + Q[3,2]*z
w = Q[1,3]*x + Q[2,3]*y + Q[3,3]*z

F = f.(x, y, z)
V = PA*F
U = threshold!(P\V, 100eps())
FastTransforms.execute_sph_reflection!(J, W, U)
FR = f.(u, v, w)
VR = PA*FR
UR = threshold!(P\VR, 100eps())
@test U ≈ UR
norm(U-UR)

F = f.(x, y, z)
V = PA*F
U = threshold!(P\V, 100eps())
FastTransforms.execute_sph_reflection!(J, (W[1], W[2], W[3]), U)
FR = f.(u, v, w)
VR = PA*FR
UR = threshold!(P\VR, 100eps())
@test U ≈ UR
norm(U-UR)

# Random orthogonal transformation
Random.seed!(0)
Q = qr(rand(3, 3)).Q

# Transform the sampling grid, note that `Q` is transposed here.
u = Q[1,1]*x + Q[2,1]*y + Q[3,1]*z
v = Q[1,2]*x + Q[2,2]*y + Q[3,2]*z
w = Q[1,3]*x + Q[2,3]*y + Q[3,3]*z

F = f.(x, y, z)
V = PA*F
U = threshold!(P\V, 100eps())
FastTransforms.execute_sph_orthogonal_transformation!(J, Q, U)
FR = f.(u, v, w)
VR = PA*FR
UR = threshold!(P\VR, 100eps())
@test U ≈ UR
norm(U-UR)
