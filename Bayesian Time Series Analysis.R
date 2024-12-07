#---------------------------------------------#
# Bayesian time series modelling and analysis #
#---------------------------------------------#

# Differencing and filtering using moving averages ----------------------------#
# Load the CO2 dataset in R
data(co2) 

# Take first differences to remove the trend 
co2_1stdiff=diff(co2,differences=1)

# Filter via moving averages to remove the seasonality 
co2_ma=filter(co2,filter=c(1/24,rep(1/12,11),1/24),sides=2)

par(mfrow=c(3,1), cex.lab=1.2,cex.main=1.2)
plot(co2) # plot the original data 
plot(co2_1stdiff) # plot the first differences (removes trend, highlights seasonality)
plot(co2_ma) # plot the filtered series via moving averages (removes the seasonality, highlights the trend)

# end -------------------------------------------- ----------------------------#

# Simulate data from a white noise process ------------------------------------#

# Simulate data with no temporal structure (white noise)
#
set.seed(2021)
T=200
t =1:T
y_white_noise=rnorm(T, mean=0, sd=1)
#
# Define a time series object in R: 
# Assume the data correspond to annual observations starting in January 1960 
#
yt=ts(y_white_noise, start=c(1960), frequency=1)
#
# plot the simulated time series, their sample ACF and their sample PACF
#
par(mfrow = c(1, 3), cex.lab = 1.3, cex.main = 1.3)
yt=ts(y_white_noise, start=c(1960), frequency=1)
plot(yt, type = 'l', col='red', xlab = 'time (t)', ylab = "Y(t)")
acf(yt, lag.max = 20, xlab = "lag",
    ylab = "Sample ACF",ylim=c(-1,1),main="")
pacf(yt, lag.max = 20,xlab = "lag",
     ylab = "Sample PACF",ylim=c(-1,1),main="")

# end -------------------------------------------------------------------------#

# Sample data from AR(1) processes --------------------------------------------#

# sample data from 2 ar(1) processes and plot their ACF and PACF functions
#
set.seed(2024)
T=500 # number of time points
#
# sample data from an ar(1) with ar coefficient phi = 0.9 and variance 1
#
v=1.0 # innovation variance
sd=sqrt(v) #innovation stantard deviation
phi1=0.9 # ar coefficient
yt1=arima.sim(n = T, model = list(ar = phi1), sd = sd)
#
# sample data from an ar(1) with ar coefficient phi = -0.9 and variance 1
#
phi2=-0.9 # ar coefficient
yt2=arima.sim(n = T, model = list(ar = phi2), sd = sd)

par(mfrow = c(2, 1), cex.lab = 1.3)
plot(yt1,main=expression(phi==0.9))
plot(yt2,main=expression(phi==-0.9))

par(mfrow = c(3, 2), cex.lab = 1.3)
lag.max=50 # max lag
#
## plot true ACFs for both processes
#
cov_0=sd^2/(1-phi1^2) # compute auto-covariance at h=0
cov_h=phi1^(0:lag.max)*cov_0 # compute auto-covariance at h
plot(0:lag.max, cov_h/cov_0, pch = 1, type = 'h', col = 'red',
     ylab = "true ACF", xlab = "Lag",ylim=c(-1,1), main=expression(phi==0.9))

cov_0=sd^2/(1-phi2^2) # compute auto-covariance at h=0
cov_h=phi2^(0:lag.max)*cov_0 # compute auto-covariance at h
# Plot autocorrelation function (ACF)
plot(0:lag.max, cov_h/cov_0, pch = 1, type = 'h', col = 'red',
     ylab = "true ACF", xlab = "Lag",ylim=c(-1,1),main=expression(phi==-0.9))

## plot sample ACFs for both processes
#
acf(yt1, lag.max = lag.max, type = "correlation", ylab = "sample ACF",
    lty = 1, ylim = c(-1, 1), main = " ")
acf(yt2, lag.max = lag.max, type = "correlation", ylab = "sample ACF",
    lty = 1, ylim = c(-1, 1), main = " ")
## plot sample PACFs for both processes
#
pacf(yt1, lag.ma = lag.max, ylab = "sample PACF", ylim=c(-1,1),main="")
pacf(yt2, lag.ma = lag.max, ylab = "sample PACF", ylim=c(-1,1),main="")

# end -------------------------------------------------------------------------#

# MLE for the AR(1), examples -------------------------------------------------#

set.seed(2024)
phi=0.9 # ar coefficient
v=1
sd=sqrt(v) # innovation standard deviation
T=500 # number of time points
yt=arima.sim(n = T, model = list(ar = phi), sd = sd) 

## Case 1: Conditional likelihood
y=as.matrix(yt[2:T]) # response
X=as.matrix(yt[1:(T-1)]) # design matrix
phi_MLE=as.numeric((t(X)%*%y)/sum(X^2)) # MLE for phi
s2=sum((y - phi_MLE*X)^2)/(length(y) - 1) # Unbiased estimate for v 
v_MLE=s2*(length(y)-1)/(length(y)) # MLE for v

cat("\n MLE of conditional likelihood for phi: ", phi_MLE, "\n",
    "MLE for the variance v: ", v_MLE, "\n", 
    "Estimate s2 for the variance v: ", s2, "\n")

phi=0.9 # ar coefficient
v=1
sd=sqrt(v) # innovation standard deviation
T=500 # number of time points
yt=arima.sim(n = T, model = list(ar = phi), sd = sd) 

#Using conditional sum of squares, equivalent to conditional likelihood 
arima_CSS=arima(yt,order=c(1,0,0),method="CSS",n.cond=1,include.mean=FALSE)
cat("AR estimates with conditional sum of squares (CSS) for phi and v:", arima_CSS$coef,arima_CSS$sigma2,
    "\n")

#Uses ML with full likelihood 
arima_ML=arima(yt,order=c(1,0,0),method="ML",include.mean=FALSE)
cat("AR estimates with full likelihood for phi and v:", arima_ML$coef,arima_ML$sigma2,
    "\n")

#Default: uses conditional sum of squares to find the starting point for ML and 
#         then uses ML 
arima_CSS_ML=arima(yt,order=c(1,0,0),method="CSS-ML",n.cond=1,include.mean=FALSE)
cat("AR estimates with CSS to find starting point for ML for phi and v:", 
    arima_CSS_ML$coef,arima_CSS_ML$sigma2,"\n")

phi=0.9 # ar coefficient
v=1
sd=sqrt(v) # innovation standard deviation
T=500 # number of time points
yt=arima.sim(n = T, model = list(ar = phi), sd = sd) 

## MLE, full likelihood AR(1) with v=1 assumed known 
# log likelihood function
log_p <- function(phi, yt){
  0.5*(log(1-phi^2) - sum((yt[2:T] - phi*yt[1:(T-1)])^2) - yt[1]^2*(1-phi^2))
}

# Use a built-in optimization method to obtain maximum likelihood estimates
result =optimize(log_p, c(-1, 1), tol = 0.0001, maximum = TRUE, yt = yt)
cat("\n MLE of full likelihood for phi: ", result$maximum)

# end -------------------------------------------------------------------------#

# AR(1) Bayesian inference, conditional likelihood example --------------------#

set.seed(2024)
phi=0.9 # ar coefficient
sd=1 # innovation standard deviation
T=200 # number of time points
yt=arima.sim(n = T, model = list(ar = phi), sd = sd) # sample stationary AR(1) process

y=as.matrix(yt[2:T]) # response
X=as.matrix(yt[1:(T-1)]) # design matrix
phi_MLE=as.numeric((t(X)%*%y)/sum(X^2)) # MLE for phi
s2=sum((y - phi_MLE*X)^2)/(length(y) - 1) # Unbiased estimate for v
v_MLE=s2*(length(y)-1)/(length(y)) # MLE for v 

print(c(phi_MLE,s2))

n_sample=3000   # posterior sample size

## step 1: sample posterior distribution of v from inverse gamma distribution
v_sample=1/rgamma(n_sample, (T-2)/2, sum((yt[2:T] - phi_MLE*yt[1:(T-1)])^2)/2)

## step 2: sample posterior distribution of phi from normal distribution
phi_sample=rep(0,n_sample)
for (i in 1:n_sample){
  phi_sample[i]=rnorm(1, mean = phi_MLE, sd=sqrt(v_sample[i]/sum(yt[1:(T-1)]^2)))}

## plot histogram of posterior samples of phi and v
par(mfrow = c(1, 2), cex.lab = 1.3)
hist(phi_sample, xlab = bquote(phi), 
     main = bquote("Posterior for "~phi),xlim=c(0.75,1.05), col='lightblue')
abline(v = phi, col = 'red')
hist(v_sample, xlab = bquote(v), col='lightblue', main = bquote("Posterior for "~v))
abline(v = sd, col = 'red')

# end -------------------------------------------------------------------------#

# Computing the roots of the AR polynomial ------------------------------------#

# Assume the folloing AR coefficients for an AR(8)
phi=c(0.27, 0.07, -0.13, -0.15, -0.11, -0.15, -0.23, -0.14)
roots=1/polyroot(c(1, -phi)) # compute reciprocal characteristic roots
r=Mod(roots) # compute moduli of reciprocal roots
lambda=2*pi/Arg(roots) # compute periods of reciprocal roots
# print results modulus and frequency by decreasing order
print(cbind(r, abs(lambda))[order(r, decreasing=TRUE), ][c(2,4,6,8),]) 

# end -------------------------------------------------------------------------#





















