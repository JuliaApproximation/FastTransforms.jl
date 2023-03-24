using FastTransforms, Test
import FastTransforms: th_leg2cheb, th_cheb2leg, th_ultra2ultra,th_jac2jac, th_leg2chebu

@testset "ToeplitzHankel" begin
    for x in ([1.0,2,3,4,5], [1.0+im,2-3im,3+4im,4-5im,5+10im])
        @test th_leg2cheb(x) ≈ leg2cheb(x)
        @test th_cheb2leg(x) ≈ cheb2leg(x)
        @test th_leg2chebu(x) ≈ ultra2ultra(x, 0.5, 1.0)
        @test th_ultra2ultra(x,0.1, 0.2) ≈ ultra2ultra(x, 0.1, 0.2)
        @test_broken th_jac2jac(x,0.1, 0.2,0.1,0.4) ≈ jac2jac(x, 0.1, 0.2,0.1,0.4)
        @test_broken th_jac2jac(x,0.1, 0.2,0.3,0.2) ≈ jac2jac(x, 0.1, 0.2,0.3,0.2)
    end

    for X in (randn(5,4), randn(5,4) + im*randn(5,4))
        @test th_leg2cheb(X, 1) ≈ hcat([leg2cheb(X[:,j]) for j=1:size(X,2)]...)
        @test th_leg2cheb(X, 2) ≈ vcat([permutedims(leg2cheb(X[k,:])) for k=1:size(X,1)]...)
        @test th_leg2cheb(X) ≈ th_leg2cheb(th_leg2cheb(X, 1), 2)

        @test th_cheb2leg(X, 1) ≈ hcat([cheb2leg(X[:,j]) for j=1:size(X,2)]...)
        @test th_cheb2leg(X, 2) ≈ vcat([permutedims(cheb2leg(X[k,:])) for k=1:size(X,1)]...)
        @test th_cheb2leg(X) ≈ th_cheb2leg(th_cheb2leg(X, 1), 2)
    end
end

