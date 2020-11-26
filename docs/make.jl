using Documenter, FastTransforms, Literate, Plots

plotlyjs()

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR   = joinpath(@__DIR__, "src/generated")

examples = [
    "chebyshev.jl",
    "disk.jl",
    "nonlocaldiffusion.jl",
    "padua.jl",
    "sphere.jl",
    "spinweighted.jl",
    "triangle.jl",
]

function uncomment_objects(str)
    str = replace(str, "###```@raw" => "```\n\n```@raw")
    str = replace(str, "###<object" => "<object")
    str = replace(str, "###```\n```" => "```")
    str
end

for example in examples
    example_filepath = joinpath(EXAMPLES_DIR, example)
    Literate.markdown(example_filepath, OUTPUT_DIR; execute=true, postprocess = uncomment_objects)
end

makedocs(
            doctest = false,
            format = Documenter.HTML(),
            sitename = "FastTransforms.jl",
            authors = "Richard Mikael Slevinsky",
            pages = Any[
                    "Home" => "index.md",
                    "Examples" => [
                        "generated/chebyshev.md",
                        "generated/disk.md",
                        "generated/nonlocaldiffusion.md",
                        "generated/padua.md",
                        "generated/sphere.md",
                        "generated/spinweighted.md",
                        "generated/triangle.md",
                        ],
                    ]
        )


deploydocs(
    repo   = "github.com/JuliaApproximation/FastTransforms.jl.git",
    )
