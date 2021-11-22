#=
    InfoGeo2D.jl
    Contains functions relating to information geometry in 2D (Christoffel symbols, Geodesic equations, Geodesic boundary value problems)
=#

#Christoffel Symbol of the second kind
#P: point, n: order of finite difference approximation, Fisher: Fisher information matrix function
function Christoffel2D(P,n,Fisher)
    #Construct univariate versions of the fisher information for differentiation using finite differences
    fisha = z -> Fisher([z,P[2]])
    fishb = z -> Fisher([P[1],z])
    ChSym = zeros(2,2,2) #initialise Christoffel symbol
    FIM = Fisher(P)
    IFIM = inv(FIM)
    dFIM = zeros(2,2,2)
    #Finite difference approximations of the derivatives of the Fisher information with respect to the unknown parameters
    dFIM[:,:,1] = central_fdm(n, 1)(fisha,P[1])
    dFIM[:,:,2] = central_fdm(n, 1)(fishb,P[2])
    #Compute Christoffel symbol of the second kind (as per https://link.springer.com/book/10.1007/978-3-319-07779-6)
    for k in 1:2
        for l in 1:2
            for i in 1:2
                ChSym[k,l,i] = (1/2)*(IFIM[i,1]*(dFIM[1,k,l]+dFIM[1,l,k]-dFIM[k,l,1])+IFIM[i,2]*(dFIM[2,k,l]+dFIM[2,l,k]-dFIM[k,l,2]))
            end
        end
    end
    return ChSym
end

#Christoffel Symbol of the first kind
#P: point, n: order of finite difference approximation, Fisher: Fisher information matrix function
function Christoffel2D_first(P,n,Fisher)
    ChSym = Christoffel2D(P,n,Fisher)
    FIM = Fisher(P)
    ChSymF = zeros(2,2,2)
    for i in 1:2
        for j in 1:2
            for k in 1:2
                ChSymF[i,j,k] = sum(FIM[k,m]*ChSym[i,j,m] for m in 1:2) #the metric is used to convert from Christoffel symbols of the second kind to Christoffel symbols of the first kind
            end
        end
    end
    return ChSymF
end
##

#Right hand side for first order geodesic ODE system
function ODEproblem_geodesic2D!(du,u,p,t,Fisher)
    ChS = Christoffel2D([u[1],u[2]],5,Fisher) #Compute Christoffel Symbols using 5th order finite differences
    du[1] = u[3] #position in θ₁
    du[2] = u[4] #position in θ₂
    du[3] = -(ChS[1,1,1]*u[3]^2+ChS[1,2,1]*u[3]*u[4]+ChS[2,1,1]*u[4]*u[3]+ChS[2,2,1]*u[4]^2) #velocity in θ₁
    du[4] = -(ChS[1,1,2]*u[3]^2+ChS[1,2,2]*u[3]*u[4]+ChS[2,1,2]*u[4]*u[3]+ChS[2,2,2]*u[4]^2) #velcoity in θ₂
    du[5] =  sqrt([u[3],u[4]]' * Fisher([u[1],u[2]]) * [u[3],u[4]]) #geodesic length
end

#Solve geodesic ODE from a point (e.g. the MLE) with a specified velocity, to a target geodesic length
function SingleGeodesic2D(MLE::AbstractArray,velocity,ODEmethod,targetLength,Fisher,cb=cb_single2D)
    t0 = 0.0 #initial time
    tN = 5.0 #final time is arbitrary provided that it is sufficiently large that the geodesic reaches the specified geodesic length
    tRange = range(t0,stop=tN,length=101)
    #let u = θ₁, v = dθ₁/dt, p = θ₂, q = dθ₂/dt,
    x_0 = MLE[1] #Initial position of geodesic in θ₁
    y_0 = MLE[2] #Initial position of geodesic in θ₂
    u_0 = velocity[1] #Initial velocity of geodesic in θ₁
    v_0 = velocity[2] #Initial velcoity of geodesic in θ₂
    l_0 = 0 #Initial geodesic length
    u0 = [x_0,y_0,u_0,v_0,l_0]
    sol = solve(ODEProblem((du,u,p,t) -> ODEproblem_geodesic2D!(du,u,p,t,Fisher),u0,(t0,tN)),p=targetLength,method=ODEmethod,callback=cb, reltol=1e-5, abstol=1e-5)
    return sol
end

##
#Callback conditions for ODEs - terminate the solver if the geodesic reaches the targetLength, or if the parameters leave the physically realistic region (for example below, if parameters become negative)
function conditionstop_shooting1(u,t,integrator) # Event occurs when event_f(u,t) == 0
u[1]  #terminate geodesic if parameter would become negative
end
function conditionstop_shooting2(u,t,integrator) # Event occurs when event_f(u,t) == 0
u[2]  #terminate geodesic if parameter would become negative
end
function conditionstop_shooting3(u,t,integrator) # Event occurs when event_f(u,t) == 0
targetLength = integrator.p #Access targetLength passed as a parameter from ODE integrator
u[5] - targetLength #terminate geodesic when we reach target length
end
function affectstop!(integrator)
  terminate!(integrator)
end
#Create callbacks (combine the condition that triggers the event, with the event; in this case affectstop!(), which we define to terminate the integrator)
cbstopshooting = ContinuousCallback(conditionstop_shooting1,affectstop!, interp_points=0)
cbstopshooting2 = ContinuousCallback(conditionstop_shooting2,affectstop!, interp_points=0)
cbstopshooting3 = ContinuousCallback(conditionstop_shooting3,affectstop!, interp_points=0)
#Combine callbacks
cb_single2D = CallbackSet(cbstopshooting,cbstopshooting2,cbstopshooting3) #stop at target length
cb_shooting2D = CallbackSet(cbstopshooting,cbstopshooting2) #don't stop at target length (for solving boundary value problems, e.g. in hypothesis testing)

##
#Riemann Tensor of the first kind, Ricci Tensor, Ricci Scalar (scalar curvature)
#P: point, n: order of finite difference approximation, Fisher: Fisher information matrix function
function RiemannTensor(P,n,Fisher)
    #Compute Christoffel symbols of the second kind
    ChSym = Christoffel2D(P,n,Fisher)
    #Compute Christoffel symbols of the first kind
    ChSymF = Christoffel2D_first(P,n,Fisher)
    #Construct univariate versions of the Christoffel symbols for differentiation using finite differences
    ChSymFa = z -> Christoffel2D_first([z,P[2]],n,Fisher)
    ChSymFb = z -> Christoffel2D_first([P[1],z],n,Fisher)
    #Finite difference approximations of the derivatives of the Christoffel symbols with respect to the unknown parameters
    dChSym = zeros(2,2,2,2)
    dChSym[:,:,:,1] = central_fdm(n, 1)(ChSymFa,P[1])
    dChSym[:,:,:,2] = central_fdm(n, 1)(ChSymFb,P[2])
    Riem = zeros(2,2,2,2)
    #Compute Riemann Tnesor of the first kind (as per Introduction to Tensor Calculus and Continuum Mechanics https://vtk.ugent.be/w/images/9/96/Wiskundige_ingenieurstechnieken_-_Introduction_to_Tensor_Calculus.pdf)
    for i in 1:2
        for j in 1:2
            for k in 1:2
                for l in 1:2
                    Riem[i,j,k,l] = dChSym[j,l,i,k] - dChSym[j,k,i,l] + sum(ChSymF[i,l,r]*ChSym[j,k,r] - ChSymF[i,k,r]*ChSym[j,l,r] for r in 1:2)
                end
            end
        end
    end
    IFIM = inv(Fisher(P))
    Ricci = zeros(2,2)
    #The inverse of the metric is used to convert from the Riemann tensor of the first kind to the Riemann tensor of the second kind
    #The Ricci tensor is obtained by contracting the contravariant index with the third covariant index of the Riemann tensor of the second kind.
    for i in 1:2
        for j in 1:2
            Ricci[i,j] = sum(IFIM[a,b]*Riem[a,i,b,j] for a in 1:2, b in 1:2)
        end
    end
    #The scalar curvature is obtained via contraction of the Ricci tensor
    Scalar = sum(IFIM[a,b]*Ricci[a,b]  for a in 1:2, b in 1:2)
    return Riem, Ricci, Scalar
end


##Additional functions for geodesics

#Solve several geodesic BVPs in parallel (requires using .Threads) from a point (e.g MLE) to specific Targets
#Targets could be (e.g.) points on a confidence region.
function GeoDiscPar2D(MLE,Targets,Fisher)
    Geods = [[i, Array{Any,1}(undef,1)] for i = 1:size(Targets)[1]]
    Guesses = Targets.-MLE' #initial guess required for shooting method
@threads for i = 1:size(Targets)[1]
    Geods[i][2] = GeodesicBVPShooting2D(MLE,Targets[i,:],Guesses[i,:],Fisher)
end
return Geods
end

#Solve geodesic between points (start) and (target) with shooting method; requires initial (guesses) of velocity components.
#Used for (e.g.) computing geodesic distances between points for hypothesis testing.
function GeodesicBVPShooting2D(start,target,guesses,Fisher)
    t0 = 0.0
    tN = 0.05 #Will be adjusted by solver to satisfy BVP constraints
    #let u = θ₁, v = dθ₁/dt, p = θ₂, q = dθ₂/dt,
    x_0 = start[1] #Initial position of geodesic in θ₁
    y_0 = start[2] #Initial position of geodesic in θ₂
    u_0 = guesses[1] #Initial velocity of geodesic in θ₁
    v_0 = guesses[2] #Initial velcoity of geodesic in θ₂
    l_0 = 0 #Initial length of geodesic
    u0 = [x_0,y_0,u_0,v_0,l_0]
    bvp2D = TwoPointBVProblem((du,u,p,t) -> ODEproblem_geodesic2D!(du,u,p,t,Fisher), bvpresidual2D!, u0,(t0,tN),(start,target))
    sol = solve(bvp2D, Shooting(Tsit5()),callback=cb_shooting2D)
    return sol
end

#Track the residuals for the BVP (have we reached the target?), BVP is solved when all residuals evaluate to (approximately) 0.
function bvpresidual2D!(residual,u,p,t)
    start = p[1]
    target = p[2]
    #Require geodesic to start at the specified starting point
    residual[1] = u[1][1] - start[1]
    residual[2] = u[1][2] - start[2]
    #Require geodesic to end at the specified target point
    residual[3] = u[end][1] - target[1]
    residual[4] = u[end][2] - target[2]
    residual[5] = u[1][5] #For BVP solver syntax; evaluates to 0 residual always as initial geodesic length is 0.
end
