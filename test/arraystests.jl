using FastTransforms, Test
import FastTransforms: ArrayPlan

@testset "Array transform"  begin
    c = randn(5,20,10)
    F = plan_cheb2leg(c)
    FT = ArrayPlan(F, c)

    f = similar(c);
    for k in axes(c,3)
        f[:,:,k] = (F*c[:,:,k])
    end
    @test f ≈ FT*c
    @test c ≈ FT\f

    F = plan_cheb2leg(Vector{Float64}(axes(c,2)))
    FT = ArrayPlan(F, c, (2,))
    for k in axes(c,3)
        f[:,:,k] = (F*c[:,:,k]')'
    end
    @test f ≈ FT*c
    @test c ≈ FT\f
end