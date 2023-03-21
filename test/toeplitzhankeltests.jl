using FastTransforms, Test

@testset "ToeplitzHankel" begin
    @test FastTransforms.th_leg2cheb([1.0,2,3,4,5]) ≈ leg2cheb([1.0,2,3,4,5])
    @test FastTransforms.th_cheb2leg([1.0,2,3,4,5]) ≈ cheb2leg([1.0,2,3,4,5])
    @test FastTransforms.th_leg2chebu([1.0,2,3,4,5]) ≈ ultra2ultra([1.0,2,3,4,5], 0.5, 1.0)
    @test FastTransforms.th_ultra2ultra([1.0,2,3,4,5],0.1, 0.2) ≈ ultra2ultra([1.0,2,3,4,5], 0.1, 0.2)
    @test_broken FastTransforms.th_jac2jac([1.0,2,3,4,5],0.1, 0.2,0.1,0.4) ≈ jac2jac([1.0,2,3,4,5], 0.1, 0.2,0.1,0.4)
    @test_broken FastTransforms.th_jac2jac([1.0,2,3,4,5],0.1, 0.2,0.3,0.2) ≈ jac2jac([1.0,2,3,4,5], 0.1, 0.2,0.3,0.2)

    X = randn(5,4)
    @test FastTransforms.th_leg2cheb(X, 1) ≈ hcat([leg2cheb(X[:,j]) for j=1:size(X,2)]...)
    @test FastTransforms.th_leg2cheb(X, 2) ≈ vcat([leg2cheb(X[k,:])' for k=1:size(X,1)]...)
    
end

