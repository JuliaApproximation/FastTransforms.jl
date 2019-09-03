using FastTransforms, Test


@testset "Test vector field transforms" begin
    # f = (θ,φ) -> cospi(θ) + sinpi(θ)*(1+cospi(2θ))*sinpi(φ) + sinpi(θ)^5*(cospi(5φ)-sinpi(5φ))
    n = 6
    θ = (0.5:n-0.5)/n
    φ = (0:2n-2)*2/(2n-1)
    ∇θf = (θ,φ) -> -sinpi(θ) + (cospi(θ)*(1+cospi(2θ)) - 2*sinpi(θ)*sinpi(2θ))*sinpi(φ) + 5*sinpi(θ)^4*cospi(θ)*(cospi(5φ)-sinpi(5φ))
    ∇φf = (θ,φ) -> (1+cospi(2θ))*cospi(φ) - 5*sinpi(θ)^4*(sinpi(5φ)+cospi(5φ))

    ∇θF = [∇θf(θ,φ) for θ in θ, φ in φ]
    ∇φF = [∇φf(θ,φ) for θ in θ, φ in φ]
    V1 = zero(∇θF)
    V2 = zero(∇φF)
    Pa = FastTransforms.plan_analysis(∇θF)
    mul!(V1, V2, Pa, ∇θF, ∇φF)
    P = SlowSphericalHarmonicPlan(V1)

    U1 = zero(V1)
    U2 = zero(V2)
    mul!(U1, U2, transpose(P), V1, V2)

    W1 = zero(U1)
    W2 = zero(U2)

    mul!(W1, W2, P, U1, U2)

    Ps = FastTransforms.plan_synthesis(W1)

    G1 = zero(∇θF)
    G2 = zero(∇φF)

    mul!(G1, G2, Ps, W1, W2)

    @test norm(∇θF - G1)/norm(∇θF) < n*eps()
    @test norm(∇φF - G2)/norm(∇φF) < n*eps()

    for k in (10, 20, 40)
        y = (1.0, 2.0, 3.0)
        n = 12k
        θ = (0.5:n-0.5)/n
        φ = (0:2n-2)*2/(2n-1)
        ∇θf = (θ,φ) -> -2k*sin(k*((sinpi(θ)*cospi(φ) - y[1])^2 + (sinpi(θ)*sinpi(φ) - y[2])^2 + (cospi(θ) - y[3])^2))*( (sinpi(θ)*cospi(φ) - y[1])*(cospi(θ)*cospi(φ)) + (sinpi(θ)*sinpi(φ) - y[2])*(cospi(θ)*sinpi(φ)) - (cospi(θ) - y[3])*sinpi(θ) )
        ∇φf = (θ,φ) -> -2k*sin(k*((sinpi(θ)*cospi(φ) - y[1])^2 + (sinpi(θ)*sinpi(φ) - y[2])^2 + (cospi(θ) - y[3])^2))*( (sinpi(θ)*cospi(φ) - y[1])*(-sinpi(φ)) + (sinpi(θ)*sinpi(φ) - y[2])*(cospi(φ)) )

        ∇θF = [∇θf(θ,φ) for θ in θ, φ in φ]
        ∇φF = [∇φf(θ,φ) for θ in θ, φ in φ]
        V1 = zero(∇θF)
        V2 = zero(∇φF)
        Pa = FastTransforms.plan_analysis(∇θF)
        mul!(V1, V2, Pa, ∇θF, ∇φF)
        P = SlowSphericalHarmonicPlan(V1)

        U1 = zero(V1)
        U2 = zero(V2)
        FastTransforms.mul!(U1, U2, transpose(P), V1, V2)

        W1 = zero(U1)
        W2 = zero(U2)

        mul!(W1, W2, P, U1, U2)

        Ps = FastTransforms.plan_synthesis(W1)

        G1 = zero(∇θF)
        G2 = zero(∇φF)

        mul!(G1, G2, Ps, W1, W2)

        @test norm(∇θF - G1)/norm(∇θF) < n*eps()
        @test norm(∇φF - G2)/norm(∇φF) < n*eps()
    end
end
