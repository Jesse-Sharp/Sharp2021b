#=
    FisherSIR_obsIonly.jl
    Fisher information for the SIR model where only I is observed
=#

#With unknown parameters θ = (β,γ)
function Fisher_βγ_obsIonly(θ,σ,T,NperT)
    J = SIR_Jacobian_βγ_obsIonly(θ,T)
    FIM = J'*FIM_Obs_Process_SIR_obsIonly(σ,T,NperT)*J
    return FIM
end

#With unknown parameters θ = (β,σ)
function Fisher_βσ_obsIonly(θ,σ,T,NperT)
    J = SIR_Jacobian_βσ_obsIonly(θ,T)
    FIM = J'*FIM_Obs_Process_SIR_obsIonly(σ,T,NperT)*J
    return FIM
end
