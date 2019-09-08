using Documenter, FastTransforms

makedocs(
			doctest = false,
			format = Documenter.HTML(),
			sitename = "FastTransforms.jl",
			authors = "Richard Mikael Slevinsky",
			pages = Any[
					"Home" => "index.md"
					]
			)


deploydocs(
    repo   = "github.com/JuliaApproximation/FastTransforms.jl.git",
    )
