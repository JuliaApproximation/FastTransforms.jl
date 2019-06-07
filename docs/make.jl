using Documenter, FastTransforms

makedocs(modules=[FastTransforms],
			doctest = true,
			clean = true,
			format = :html,
			sitename = "FastTransforms.jl",
			authors = "Richard Mikael Slevinsky",
			pages = Any[
					"Home" => "index.md"
					]
			)


deploydocs(
    repo   = "github.com/JuliaApproximation/FastTransforms.jl.git",
    latest = "master",
    julia  = "0.6",
    osname = "linux",
    target = "build",
    deps   = Deps.pip("pygments", "mkdocs", "python-markdown-math"),
    make   = nothing
    )
