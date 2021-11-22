#=
    Logistic_partials.jl
    Here we compute the partial derivatives of the logistic model with respect to the parameters (determined analytically)
=#

#Partial derivatives of the logistic model with respect to r
function dLdr(θ,t)
    (r,C0,K) = θ
    dLdr = (C0*K*t*(K-C0)*exp(-r*t))/((K-C0)*exp(-r*t)+C0)^2
    return dLdr
end

#Partial derivatives of the logistic model with respect to the initial condition
function dLdC0(θ,t)
    (r,C0,K) = θ
    dLdC0 = (K^2*exp(r*t))/(C0*(exp(r*t)-1)+K)^2
    return dLdC0
end

#Partial derivatives of the logistic model with respect to K
function dLdK(θ,t)
    (r,C0,K) = θ
    dLdK = (C0^2*exp(r*t)*(exp(r*t)-1))/(C0*(exp(r*t)-1)+K)^2
    return dLdK
end
