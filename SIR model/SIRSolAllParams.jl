#=
    SIRSolAllParams.jl
    Here we solve the SIR model numerically to obtain the solution at a specified timepoint, t
=#

function SIRSolAllParams(θ,t)
    p = θ[1:2]
    IC = θ[3:5]
    prob = ODEProblem(SIRODE!,IC,(0.0,t),p)
    sol = solve(prob)
    return sol
end
