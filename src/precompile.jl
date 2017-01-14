function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    precompile(FastTransforms.pochhammer, (Float64, Int64,))
end
