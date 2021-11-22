#=
    Exponential_partials.jl
    Here we compute the partial derivatives of the exponential model with respect to the parameters (determined analytically)
=#

#Partial derivatives of the exponential model with respect to a
function dCda(θ,t)
    (a,C0) = θ
    dCda = t*C0*exp(a*t)
    return dCda
end

#Partial derivatives of the exponential model with respect to C(0)
function dCdC0(θ,t)
    (a,C0) = θ
    dCdC0 = exp(a*t)
    return dCdC0
end
