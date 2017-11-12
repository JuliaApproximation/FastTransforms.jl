using FastTransforms, Compat
using Compat.Test

@testset "Chebyshev--Legendre transform" begin
    for k in round.([Int],logspace(1,4,20))
        r = randn(k)./(√).(1:k) # Proven 𝒪(√(log N)) error for ASY method.
        @test leg2cheb(r) ≈ cjt(r,0.,0.)
    end

    @test leg2chebu([1.0,2,3,4,5])  ≈ [0.546875,0.5,0.5390625,1.25,1.3671875]

    c = randn(1000)./(√).(1:1000);

    @test leg2cheb(cheb2leg(c)) ≈ c
    @test cheb2leg(leg2cheb(c)) ≈ c

    @test norm(jac2jac(c,0.,√2/2,-1/4,√2/2)-jjt(c,0.,√2/2,-1/4,√2/2),Inf) < 10length(c)*eps()

    @test norm(ultra2ultra(ultra2ultra(c,.5,.75),.75,.5)-c,Inf) < 10length(c)*eps()
end
