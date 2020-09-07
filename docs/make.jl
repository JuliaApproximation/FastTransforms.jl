using Documenter, FastTransforms, Literate

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR   = joinpath(@__DIR__, "src/generated")

examples = [
	"sphere.jl",
]

for example in examples
	example_filepath = joinpath(EXAMPLES_DIR, example)
	Literate.markdown(example_filepath, OUTPUT_DIR; execute=true)
end

makedocs(
			doctest = false,
			format = Documenter.HTML(),
			sitename = "FastTransforms.jl",
			authors = "Richard Mikael Slevinsky",
			pages = Any[
					"Home" => "index.md",
					"Examples" => [
        				"generated/sphere.md",
        				],
					]
			)


deploydocs(
    repo   = "github.com/JuliaApproximation/FastTransforms.jl.git",
    )
