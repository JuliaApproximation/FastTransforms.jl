using FastTransforms, Test
import FastTransforms: th_leg2cheb, th_cheb2leg, th_ultra2ultra,th_jac2jac

@testset "ToeplitzHankel" begin
    @test th_leg2cheb([1.0,2,3,4,5]) ≈ leg2cheb([1.0,2,3,4,5])
    @test th_cheb2leg([1.0,2,3,4,5]) ≈ cheb2leg([1.0,2,3,4,5])
    @test th_leg2chebu([1.0,2,3,4,5]) ≈ ultra2ultra([1.0,2,3,4,5], 0.5, 1.0)
    @test th_ultra2ultra([1.0,2,3,4,5],0.1, 0.2) ≈ ultra2ultra([1.0,2,3,4,5], 0.1, 0.2)
    @test_broken th_jac2jac([1.0,2,3,4,5],0.1, 0.2,0.1,0.4) ≈ jac2jac([1.0,2,3,4,5], 0.1, 0.2,0.1,0.4)
    @test_broken th_jac2jac([1.0,2,3,4,5],0.1, 0.2,0.3,0.2) ≈ jac2jac([1.0,2,3,4,5], 0.1, 0.2,0.3,0.2)

    X = randn(5,4)
    @test th_leg2cheb(X, 1) ≈ hcat([leg2cheb(X[:,j]) for j=1:size(X,2)]...)
    @test th_leg2cheb(X, 2) ≈ vcat([leg2cheb(X[k,:])' for k=1:size(X,1)]...)
    @test th_leg2cheb(X) ≈ th_leg2cheb(th_leg2cheb(X, 1), 2)

    @test th_cheb2leg(X, 1) ≈ hcat([cheb2leg(X[:,j]) for j=1:size(X,2)]...)
    @test th_cheb2leg(X, 2) ≈ vcat([cheb2leg(X[k,:])' for k=1:size(X,1)]...)
    @test th_cheb2leg(X) ≈ th_cheb2leg(th_cheb2leg(X, 1), 2)
end

