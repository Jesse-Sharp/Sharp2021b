#=
    ExponentialSolAllParams.jl
    Here we solve the exponential model analytically to obtain the solution at a specified timepoint, t
=#

function ExponentialSolAllParams(θ,t)
    (a,C0) = θ
    Exponential = C0*exp(a*t)
end
