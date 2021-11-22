#=
    Linear_partials.jl
    Here we compute the partial derivatives of the linear model with respect to the parameters (determined analytically)
=#

#Partial derivatives of the linear model with respect to a
function dCda(θ,t)
    dCda = t
    return dCda
end

#Partial derivatives of the linear model with respect to C(0)
function dCdC0(θ,t)
    dCdC0 = 1
    return dCdC0
end
