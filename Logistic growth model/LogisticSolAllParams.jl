#=
    LogisticSolAllParams.jl
    Here we solve the logistic model analytically to obtain the solution at a specified timepoint, t
=#

function LogisticSolAllParams(θ,t)
    (r,C0,K) = θ
    Logistic = K*C0./(C0.+(K-C0)*exp.(-r*t))
end
