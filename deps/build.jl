if Sys.isapple()
    const libfasttransforms = joinpath(dirname(@__DIR__), "deps", "libfasttransforms.dylib")
    download("https://github.com/MikaelSlevinsky/FastTransforms/releases/download/v0.2.3/libfasttransforms.dylib", libfasttransforms)
elseif Sys.islinux()
    const libfasttransforms = joinpath(dirname(@__DIR__), "deps", "libfasttransforms.so")
    download("https://github.com/MikaelSlevinsky/FastTransforms/releases/download/v0.2.3/libfasttransforms.so", libfasttransforms)
else
    @warn "FastTransforms is not properly installed."
end
