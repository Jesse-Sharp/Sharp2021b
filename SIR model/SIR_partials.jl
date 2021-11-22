#=
    SIR_partials.jl
    Here we compute the partial derivatives of the SIR model with respect to the parameters using forward mode automatic differentiation
=#

function dModeldParams(θ,t)
    SIRsol = ϕ -> SIRSolAllParams([ϕ;θ[3:5]],t).u[end,:][1]
    dModeldθ = ForwardDiff.jacobian(SIRsol,θ[1:2])
    return dModeldθ
end
