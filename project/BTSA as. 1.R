#-------------------#
# assignment 1 BTSA #
#-------------------#

# import data
yt = read.delim("C:/Users/julia/OneDrive/Desktop/Statistics notes/w50y2024/data.txt", header = FALSE)
yt = as.numeric(data$V1)
plot(yt, type = 'l', col = "blue", main = 'Plot of data',
     xlab = "Index", ylab = "Value", 
     main = "Plot of Data")

# Case 1: Conditional likelihood
set.seed(2024)
p=8
y=rev(yt[(p+1):T]) # response
X=t(matrix(yt[rev(rep((1:p),T-p)+rep((0:(T-p-1)),rep(p,T-p)))],p,T-p));
XtX=t(X)%*%X
XtX_inv=solve(XtX)
phi_MLE=XtX_inv%*%t(X)%*%y # MLE for phi
s2=sum((y - X%*%phi_MLE)^2)/(length(y) - p) #unbiased estimate for v

cat("\n MLE of conditional likelihood for phi: ", phi_MLE, "\n",
    "Estimate for v: ", s2, "\n")

################################################################################
##  AR(2) case 
### Posterior inference, conditional likelihood + reference prior via 
### direct sampling                 
################################################################################

n_sample=500 # posterior sample size
library(MASS)

## step 1: sample v from inverse gamma distribution
v_sample=1/rgamma(n_sample, (T-2*p)/2, sum((y-X%*%phi_MLE)^2)/2)

## step 2: sample phi conditional on v from normal distribution
phi_sample=matrix(0, nrow = n_sample, ncol = p)
for(i in 1:n_sample){
  phi_sample[i, ]=mvrnorm(1,phi_MLE,Sigma=v_sample[i]*XtX_inv)
}

#posterior means
apply(phi_sample, 2, mean)
# [1]  1.605355159 -0.885135355 -0.004239663  0.007291180 
# -0.026931834  0.011587524 -0.021160654  0.011207555

round(apply(phi_sample, 2, mean),3)

#rounded variance estimate
round(mean(v_sample), 3)

# question 5
# Assume the folloing AR coefficients for an AR(8)
phi= apply(phi_sample, 2, mean)
roots=1/polyroot(c(1, -phi)) # compute reciprocal characteristic roots
r=Mod(roots) # compute moduli of reciprocal roots
lambda=2*pi/Arg(roots) # compute periods of reciprocal roots
# print results modulus and frequency by decreasing order
print(cbind(r, abs(lambda))[order(r, decreasing=TRUE), ][c(2,4,6,8),]) 

#             r             
# [1,] 0.9632800 1.179690e+01
# [2,] 0.5057569 5.799588e+00
# [3,] 0.4720255 3.019354e+00
# [4,] 0.4280434 1.580598e+16

# moduli
round(r, 3)

# periods
round(lambda, 3)

#-----#
# end #
#-----#
