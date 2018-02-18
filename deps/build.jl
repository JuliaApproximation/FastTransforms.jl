p = pwd()
cd(Pkg.dir("FastTransforms/deps/"))
run(`make`)
cd(p)
