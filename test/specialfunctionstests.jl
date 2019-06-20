using FastTransforms
using Compat.Test
import FastTransforms.pochhammer

const sqrtpi = 1.772453850905516027298

@testset "Pochhammer" begin
    @test pochhammer(2,3) == 24
    @test pochhammer(0.5,3) == 0.5*1.5*2.5
    @test pochhammer(0.5,0.5) == 1/sqrtpi
    @test pochhammer(0,1) == 0
    @test pochhammer(-1,2) == 0
    @test pochhammer(-5,3) == -60
    @test pochhammer(-1,-0.5) == 0
    @test 1.0/pochhammer(-0.5,-0.5) == 0
    @test pochhammer(-1+0im,-1) == -0.5
end
