using FastTransforms, Test
import FastTransforms: th_leg2cheb, th_cheb2leg, th_leg2chebu, th_ultra2ultra,th_jac2jac, th_leg2chebu,
                        lib_leg2cheb, lib_cheb2leg, lib_ultra2ultra, lib_jac2jac,
                        plan_th_cheb2leg!, plan_th_leg2chebu!, plan_th_leg2cheb!, plan_th_ultra2ultra!, plan_th_jac2jac!

@testset "ToeplitzHankel" begin
    for x in ([1.0], [1.0,2,3,4,5], [1.0+im,2-3im,3+4im,4-5im,5+10im], collect(1.0:1000))
        @test th_leg2cheb(x) ≈ lib_leg2cheb(x)
        @test th_cheb2leg(x) ≈ lib_cheb2leg(x)
        @test th_leg2chebu(x) ≈ lib_ultra2ultra(x, 0.5, 1.0)
        @test th_ultra2ultra(x,0.1, 0.2) ≈ lib_ultra2ultra(x, 0.1, 0.2)
        @test th_jac2jac(x,0.1, 0.2,0.1,0.4) ≈ lib_jac2jac(x, 0.1, 0.2,0.1,0.4)
        @test th_jac2jac(x,0.1, 0.2,0.3,0.2) ≈ lib_jac2jac(x, 0.1, 0.2,0.3,0.2)


        @test th_cheb2leg(th_leg2cheb(x)) ≈ x atol=1E-9
        @test th_leg2cheb(th_cheb2leg(x)) ≈ x atol=1E-10
    end

    for X in (randn(5,4), randn(5,4) + im*randn(5,4))
        @test th_leg2cheb(X, 1) ≈ hcat([leg2cheb(X[:,j]) for j=1:size(X,2)]...)
        @test_broken th_leg2cheb(X, 1) ≈ leg2cheb(X, 1) # matrices not supported in FastTransforms
        @test th_leg2cheb(X, 2) ≈ vcat([permutedims(leg2cheb(X[k,:])) for k=1:size(X,1)]...)
        @test_broken th_leg2cheb(X, 2) ≈ leg2cheb(X, 2)
        @test th_leg2cheb(X) ≈ th_leg2cheb(th_leg2cheb(X, 1), 2)
        @test_broken th_leg2cheb(X) ≈ leg2cheb(X)

        @test th_cheb2leg(X, 1) ≈ hcat([cheb2leg(X[:,j]) for j=1:size(X,2)]...)
        @test th_cheb2leg(X, 2) ≈ vcat([permutedims(cheb2leg(X[k,:])) for k=1:size(X,1)]...)
        @test th_cheb2leg(X) ≈ th_cheb2leg(th_cheb2leg(X, 1), 2)

        @test th_cheb2leg(X) == plan_th_cheb2leg!(X, 1:2)*copy(X)
        @test th_leg2cheb(X) == plan_th_leg2cheb!(X, 1:2)*copy(X)

        @test th_leg2cheb(th_cheb2leg(X)) ≈ X

        @test th_leg2chebu(X, 1) ≈ hcat([ultra2ultra(X[:,j], 0.5, 1.0) for j=1:size(X,2)]...)
        @test th_leg2chebu(X, 2) ≈ vcat([permutedims(ultra2ultra(X[k,:], 0.5, 1.0)) for k=1:size(X,1)]...)
        @test th_leg2chebu(X) ≈ th_leg2chebu(th_leg2chebu(X, 1), 2)

        @test th_leg2chebu(X) == plan_th_leg2chebu!(X, 1:2)*copy(X)

        @test th_ultra2ultra(X, 0.1, 0.6, 1) ≈ hcat([ultra2ultra(X[:,j], 0.1, 0.6) for j=1:size(X,2)]...)
        @test th_ultra2ultra(X, 0.1, 0.6, 2) ≈ vcat([permutedims(ultra2ultra(X[k,:], 0.1, 0.6)) for k=1:size(X,1)]...)
        @test th_ultra2ultra(X, 0.1, 0.6) ≈ th_ultra2ultra(th_ultra2ultra(X, 0.1, 0.6, 1), 0.1, 0.6, 2)

        @test th_ultra2ultra(X, 0.1, 0.6) == plan_th_ultra2ultra!(X, 0.1, 0.6, 1:2)*copy(X)
        @test th_ultra2ultra(X, 0.1, 0.6) == plan_th_ultra2ultra!(X, 0.1, 0.6, 1:2)*copy(X)

        @test th_ultra2ultra(th_ultra2ultra(X, 0.1, 0.6), 0.6, 0.1) ≈ X

        @test th_jac2jac(X, 0.1, 0.6, 0.1, 0.8, 1) ≈ hcat([jac2jac(X[:,j], 0.1, 0.6, 0.1, 0.8) for j=1:size(X,2)]...)
        @test th_jac2jac(X, 0.1, 0.6, 0.1, 0.8, 2) ≈ vcat([permutedims(jac2jac(X[k,:], 0.1, 0.6, 0.1, 0.8)) for k=1:size(X,1)]...)
        @test th_jac2jac(X, 0.1, 0.6, 0.1, 0.8) ≈ th_jac2jac(th_jac2jac(X, 0.1, 0.6, 0.1, 0.8, 1), 0.1, 0.6, 0.1, 0.8, 2)

        @test th_jac2jac(X, 0.1, 0.6, 0.1, 0.8) == plan_th_jac2jac!(X, 0.1, 0.6, 0.1, 0.8, 1:2)*copy(X)
        @test th_jac2jac(X, 0.1, 0.6, 0.1, 0.8) == plan_th_jac2jac!(X, 0.1, 0.6, 0.1, 0.8, 1:2)*copy(X)

        @test th_jac2jac(th_jac2jac(X, 0.1, 0.6, 0.1, 0.8), 0.1, 0.8, 0.1, 0.6) ≈ X
    end

    @testset "BigFloat" begin
        n = 10
        x = big.(collect(1.0:n))
        @test th_leg2cheb(x) ≈ lib_leg2cheb(x)
        @test th_cheb2leg(x) ≈ lib_cheb2leg(x)
    end

    @testset "jishnub example" begin
        x = chebyshevpoints(4096);
        f = x -> cospi(1000x);  
        y = f.(x);
        v = th_cheb2leg(chebyshevtransform(y))
        @test norm(v - th_cheb2leg(th_leg2cheb(v)), Inf) ≤ 1E-13
        @test norm(v - th_cheb2leg(th_leg2cheb(v)))/norm(v) ≤ 1E-14
    end
end