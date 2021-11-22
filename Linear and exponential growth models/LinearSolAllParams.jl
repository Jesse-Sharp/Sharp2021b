#=
    LinearSolAllParams.jl
    Here we solve the linear model analytically to obtain the solution at a specified timepoint, t
=#

function LinearSolAllParams(θ,t)
    (a,C0) = θ
    Linear = a*t+C0
end
