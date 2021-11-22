#=
    EquidistantCircumferencePoints.jl
    Generates N equidistant points on the circumference of a unit circle
=#

function EquidistantCircumferencePoints(N)
RadialPosFun = α -> [cos(α);sin(α)]
RadialPos = zeros(N,2)
for i in 1:N
    RadialPos[i,:] = RadialPosFun((i/N)*(2*pi))
end
return RadialPos
end
