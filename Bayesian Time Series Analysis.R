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

# Maximum likelihood estimation, AR(p), conditional likelihood-----------------#

set.seed(2021)
# Simulate 300 observations from an AR(2) with one pair of complex-valued reciprocal roots 
r=0.95
lambda=12 
phi=numeric(2) 
phi[1]=2*r*cos(2*pi/lambda) 
phi[2]=-r^2
sd=1 # innovation standard deviation
T=300 # number of time points
# generate stationary AR(2) process
yt=arima.sim(n = T, model = list(ar = phi), sd = sd) 

## Compute the MLE for phi and the unbiased estimator for v using the conditional likelihood
p=2
y=rev(yt[(p+1):T]) # response
X=t(matrix(yt[rev(rep((1:p),T-p)+rep((0:(T-p-1)),rep(p,T-p)))],p,T-p));
XtX=t(X)%*%X
XtX_inv=solve(XtX)
phi_MLE=XtX_inv%*%t(X)%*%y # MLE for phi
s2=sum((y - X%*%phi_MLE)^2)/(length(y) - p) #unbiased estimate for v

cat("\n MLE of conditional likelihood for phi: ", phi_MLE, "\n",
    "Estimate for v: ", s2, "\n")

# end -------------------------------------------------------------------------#

# Bayesian inference, AR(p), conditional likelihood ---------------------------#

# Simulate 300 observations from an AR(2) with one pair of complex-valued roots 
set.seed(2021)
r=0.95
lambda=12 
phi=numeric(2) 
phi[1]=2*r*cos(2*pi/lambda) 
phi[2]=-r^2
sd=1 # innovation standard deviation
T=300 # number of time points
# generate stationary AR(2) process
yt=arima.sim(n = T, model = list(ar = phi), sd = sd) 
par(mfrow=c(1,1))
plot(yt)

## Compute the MLE of phi and the unbiased estimator of v using the conditional likelihood
p=2
y=rev(yt[(p+1):T]) # response
X=t(matrix(yt[rev(rep((1:p),T-p)+rep((0:(T-p-1)),rep(p,T-p)))],p,T-p));
XtX=t(X)%*%X
XtX_inv=solve(XtX)
phi_MLE=XtX_inv%*%t(X)%*%y # MLE for phi
s2=sum((y - X%*%phi_MLE)^2)/(length(y) - p) #unbiased estimate for v

################################################################################
### Posterior inference, conditional likelihood + reference prior via 
### direct sampling                 
################################################################################

n_sample=1000 # posterior sample size
library(MASS)

## step 1: sample v from inverse gamma distribution
v_sample=1/rgamma(n_sample, (T-2*p)/2, sum((y-X%*%phi_MLE)^2)/2)

## step 2: sample phi conditional on v from normal distribution
phi_sample=matrix(0, nrow = n_sample, ncol = p)
for(i in 1:n_sample){
  phi_sample[i, ]=mvrnorm(1,phi_MLE,Sigma=v_sample[i]*XtX_inv)
}

# end -------------------------------------------------------------------------#

# Smoothing in the NDLM, Example ----------------------------------------------#

#################################################
##### Univariate DLM: Known, constant variances
#################################################
set_up_dlm_matrices <- function(FF, GG, VV, WW){
  return(list(FF=FF, GG=GG, VV=VV, WW=WW))
}

set_up_initial_states <- function(m0, C0){
  return(list(m0=m0, C0=C0))
}

### forward update equations ###
forward_filter <- function(data, matrices, initial_states){
  ## retrieve dataset
  yt <- data$yt
  T <- length(yt)
  
  ## retrieve a set of quadruples 
  # FF, GG, VV, WW are scalar
  FF <- matrices$FF  
  GG <- matrices$GG
  VV <- matrices$VV
  WW <- matrices$WW
  
  ## retrieve initial states
  m0 <- initial_states$m0
  C0 <- initial_states$C0
  
  ## create placeholder for results
  d <- dim(GG)[1]
  at <- matrix(NA, nrow=T, ncol=d)
  Rt <- array(NA, dim=c(d, d, T))
  ft <- numeric(T)
  Qt <- numeric(T)
  mt <- matrix(NA, nrow=T, ncol=d)
  Ct <- array(NA, dim=c(d, d, T))
  et <- numeric(T)
  
  
  for(i in 1:T){
    # moments of priors at t
    if(i == 1){
      at[i, ] <- GG %*% t(m0)
      Rt[, , i] <- GG %*% C0 %*% t(GG) + WW
      Rt[,,i] <- 0.5*Rt[,,i]+0.5*t(Rt[,,i]) 
    }else{
      at[i, ] <- GG %*% t(mt[i-1, , drop=FALSE])
      Rt[, , i] <- GG %*% Ct[, , i-1] %*% t(GG) + WW
      Rt[,,i] <- 0.5*Rt[,,i]+0.5*t(Rt[,,i]) 
    }
    
    # moments of one-step forecast:
    ft[i] <- t(FF) %*% (at[i, ]) 
    Qt[i] <- t(FF) %*% Rt[, , i] %*% FF + VV
    
    # moments of posterior at t:
    At <- Rt[, , i] %*% FF / Qt[i]
    et[i] <- yt[i] - ft[i]
    mt[i, ] <- at[i, ] + t(At) * et[i]
    Ct[, , i] <- Rt[, , i] - Qt[i] * At %*% t(At)
    Ct[,,i] <- 0.5*Ct[,,i] + 0.5*t(Ct[,,i]) 
  }
  cat("Forward filtering is completed!") # indicator of completion
  return(list(mt = mt, Ct = Ct, at = at, Rt = Rt, 
              ft = ft, Qt = Qt))
}


forecast_function <- function(posterior_states, k, matrices){
  
  ## retrieve matrices
  FF <- matrices$FF
  GG <- matrices$GG
  WW <- matrices$WW
  VV <- matrices$VV
  mt <- posterior_states$mt
  Ct <- posterior_states$Ct
  
  ## set up matrices
  T <- dim(mt)[1] # time points
  d <- dim(mt)[2] # dimension of state parameter vector
  
  ## placeholder for results
  at <- matrix(NA, nrow = k, ncol = d)
  Rt <- array(NA, dim=c(d, d, k))
  ft <- numeric(k)
  Qt <- numeric(k)
  
  
  for(i in 1:k){
    ## moments of state distribution
    if(i == 1){
      at[i, ] <- GG %*% t(mt[T, , drop=FALSE])
      Rt[, , i] <- GG %*% Ct[, , T] %*% t(GG) + WW
      Rt[,,i] <- 0.5*Rt[,,i]+0.5*t(Rt[,,i]) 
    }else{
      at[i, ] <- GG %*% t(at[i-1, , drop=FALSE])
      Rt[, , i] <- GG %*% Rt[, , i-1] %*% t(GG) + WW
      Rt[,,i] <- 0.5*Rt[,,i]+0.5*t(Rt[,,i]) 
    }
    
    ## moments of forecast distribution
    ft[i] <- t(FF) %*% t(at[i, , drop=FALSE])
    Qt[i] <- t(FF) %*% Rt[, , i] %*% FF + VV
  }
  cat("Forecasting is completed!") # indicator of completion
  return(list(at=at, Rt=Rt, ft=ft, Qt=Qt))
}

## obtain 95% credible interval
get_credible_interval <- function(mu, sigma2, 
                                  quantile = c(0.025, 0.975)){
  z_quantile <- qnorm(quantile)
  bound <- matrix(0, nrow=length(mu), ncol=2)
  bound[, 1] <- mu + z_quantile[1]*sqrt(as.numeric(sigma2)) # lower bound
  bound[, 2] <- mu + z_quantile[2]*sqrt(as.numeric(sigma2)) # upper bound
  return(bound)
}

### smoothing equations ###
backward_smoothing <- function(data, matrices, 
                               posterior_states){
  ## retrieve data 
  yt <- data$yt
  T <- length(yt) 
  
  ## retrieve matrices
  FF <- matrices$FF
  GG <- matrices$GG
  
  ## retrieve matrices
  mt <- posterior_states$mt
  Ct <- posterior_states$Ct
  at <- posterior_states$at
  Rt <- posterior_states$Rt
  
  ## create placeholder for posterior moments 
  mnt <- matrix(NA, nrow = dim(mt)[1], ncol = dim(mt)[2])
  Cnt <- array(NA, dim = dim(Ct))
  fnt <- numeric(T)
  Qnt <- numeric(T)
  for(i in T:1){
    # moments for the distributions of the state vector given D_T
    if(i == T){
      mnt[i, ] <- mt[i, ]
      Cnt[, , i] <- Ct[, , i]
      Cnt[, , i] <- 0.5*Cnt[, , i] + 0.5*t(Cnt[, , i]) 
    }else{
      inv_Rtp1<-solve(Rt[,,i+1])
      Bt <- Ct[, , i] %*% t(GG) %*% inv_Rtp1
      mnt[i, ] <- mt[i, ] + Bt %*% (mnt[i+1, ] - at[i+1, ])
      Cnt[, , i] <- Ct[, , i] + Bt %*% (Cnt[, , i + 1] - Rt[, , i+1]) %*% t(Bt)
      Cnt[,,i] <- 0.5*Cnt[,,i] + 0.5*t(Cnt[,,i]) 
    }
    # moments for the smoothed distribution of the mean response of the series
    fnt[i] <- t(FF) %*% t(mnt[i, , drop=FALSE])
    Qnt[i] <- t(FF) %*% t(Cnt[, , i]) %*% FF
  }
  cat("Backward smoothing is completed!")
  return(list(mnt = mnt, Cnt = Cnt, fnt=fnt, Qnt=Qnt))
}
####################### Example: Lake Huron Data ######################
plot(LakeHuron,main="Lake Huron Data",ylab="level in feet") 
# 98 observations total 
k=4
T=length(LakeHuron)-k # We take the first 94 observations 
#  as our data
ts_data=LakeHuron[1:T]
ts_validation_data <- LakeHuron[(T+1):98]

data <- list(yt = ts_data)

## set up matrices
FF <- as.matrix(1)
GG <- as.matrix(1)
VV <- as.matrix(1)
WW <- as.matrix(1)
m0 <- as.matrix(570)
C0 <- as.matrix(1e4)

## wrap up all matrices and initial values
matrices <- set_up_dlm_matrices(FF,GG,VV,WW)
initial_states <- set_up_initial_states(m0, C0)

## filtering
results_filtered <- forward_filter(data, matrices, 
                                   initial_states)
ci_filtered<-get_credible_interval(results_filtered$mt,
                                   results_filtered$Ct)
## smoothing
results_smoothed <- backward_smoothing(data, matrices, 
                                       results_filtered)
ci_smoothed <- get_credible_interval(results_smoothed$mnt, 
                                     results_smoothed$Cnt)


index=seq(1875, 1972, length.out = length(LakeHuron))
index_filt=index[1:T]

plot(index, LakeHuron, main = "Lake Huron Level ",type='l',
     xlab="time",ylab="level in feet",lty=3,ylim=c(575,583))
points(index,LakeHuron,pch=20)

lines(index_filt, results_filtered$mt, type='l', 
      col='red',lwd=2)
lines(index_filt, ci_filtered[,1], type='l', col='red',lty=2)
lines(index_filt, ci_filtered[,2], type='l', col='red',lty=2)

lines(index_filt, results_smoothed$mnt, type='l', 
      col='blue',lwd=2)
lines(index_filt, ci_smoothed[,1], type='l', col='blue',lty=2)
lines(index_filt, ci_smoothed[,2], type='l', col='blue',lty=2)

legend('bottomleft', legend=c("filtered","smoothed"),
       col = c("red", "blue"), lty=c(1, 1))


# end -------------------------------------------------------------------------#

# NDLM, Unknown Observational Variance Example --------------------------------#

# create list for matrices
set_up_dlm_matrices_unknown_v <- function(Ft, Gt, Wt_star){
  if(!is.array(Gt)){
    Stop("Gt and Ft should be array")
  }
  if(missing(Wt_star)){
    return(list(Ft=Ft, Gt=Gt))
  }else{
    return(list(Ft=Ft, Gt=Gt, Wt_star=Wt_star))
  }
}


## create list for initial states
set_up_initial_states_unknown_v <- function(m0, C0_star, n0, S0){
  return(list(m0=m0, C0_star=C0_star, n0=n0, S0=S0))
}

forward_filter_unknown_v <- function(data, matrices, 
                                     initial_states, delta){
  ## retrieve dataset
  yt <- data$yt
  T<- length(yt)
  
  ## retrieve matrices
  Ft <- matrices$Ft
  Gt <- matrices$Gt
  if(missing(delta)){
    Wt_star <- matrices$Wt_star
  }
  
  ## retrieve initial state
  m0 <- initial_states$m0
  C0_star <- initial_states$C0_star
  n0 <- initial_states$n0
  S0 <- initial_states$S0
  C0 <- S0*C0_star
  
  ## create placeholder for results
  d <- dim(Gt)[1]
  at <- matrix(0, nrow=T, ncol=d)
  Rt <- array(0, dim=c(d, d, T))
  ft <- numeric(T)
  Qt <- numeric(T)
  mt <- matrix(0, nrow=T, ncol=d)
  Ct <- array(0, dim=c(d, d, T))
  et <- numeric(T)
  nt <- numeric(T)
  St <- numeric(T)
  dt <- numeric(T)
  
  # moments of priors at t
  for(i in 1:T){
    if(i == 1){
      at[i, ] <- Gt[, , i] %*% m0
      Pt <- Gt[, , i] %*% C0 %*% t(Gt[, , i])
      Pt <- 0.5*Pt + 0.5*t(Pt)
      if(missing(delta)){
        Wt <- Wt_star[, , i]*S0
        Rt[, , i] <- Pt + Wt
        Rt[,,i] <- 0.5*Rt[,,i]+0.5*t(Rt[,,i])
      }else{
        Rt[, , i] <- Pt/delta
        Rt[,,i] <- 0.5*Rt[,,i]+0.5*t(Rt[,,i])
      }
      
    }else{
      at[i, ] <- Gt[, , i] %*% t(mt[i-1, , drop=FALSE])
      Pt <- Gt[, , i] %*% Ct[, , i-1] %*% t(Gt[, , i])
      if(missing(delta)){
        Wt <- Wt_star[, , i] * St[i-1]
        Rt[, , i] <- Pt + Wt
        Rt[,,i]=0.5*Rt[,,i]+0.5*t(Rt[,,i])
      }else{
        Rt[, , i] <- Pt/delta
        Rt[,,i] <- 0.5*Rt[,,i]+0.5*t(Rt[,,i])
      }
    }
    
    # moments of one-step forecast:
    ft[i] <- t(Ft[, , i]) %*% t(at[i, , drop=FALSE]) 
    Qt[i] <- t(Ft[, , i]) %*% Rt[, , i] %*% Ft[, , i] + 
      ifelse(i==1, S0, St[i-1])
    et[i] <- yt[i] - ft[i]
    
    nt[i] <- ifelse(i==1, n0, nt[i-1]) + 1
    St[i] <- ifelse(i==1, S0, 
                    St[i-1])*(1 + 1/nt[i]*(et[i]^2/Qt[i]-1))
    
    # moments of posterior at t:
    At <- Rt[, , i] %*% Ft[, , i] / Qt[i]
    
    mt[i, ] <- at[i, ] + t(At) * et[i]
    Ct[, , i] <- St[i]/ifelse(i==1, S0, 
                              St[i-1])*(Rt[, , i] - Qt[i] * At %*% t(At))
    Ct[,,i] <- 0.5*Ct[,,i]+0.5*t(Ct[,,i])
  }
  cat("Forward filtering is completed!\n")
  return(list(mt = mt, Ct = Ct,  at = at, Rt = Rt, 
              ft = ft, Qt = Qt,  et = et, 
              nt = nt, St = St))
}

### smoothing function ###
backward_smoothing_unknown_v <- function(data, matrices, 
                                         posterior_states,delta){
  ## retrieve data 
  yt <- data$yt
  T <- length(yt) 
  
  ## retrieve matrices
  Ft <- matrices$Ft
  Gt <- matrices$Gt
  
  ## retrieve matrices
  mt <- posterior_states$mt
  Ct <- posterior_states$Ct
  Rt <- posterior_states$Rt
  nt <- posterior_states$nt
  St <- posterior_states$St
  at <- posterior_states$at
  
  ## create placeholder for posterior moments 
  mnt <- matrix(NA, nrow = dim(mt)[1], ncol = dim(mt)[2])
  Cnt <- array(NA, dim = dim(Ct))
  fnt <- numeric(T)
  Qnt <- numeric(T)
  
  for(i in T:1){
    if(i == T){
      mnt[i, ] <- mt[i, ]
      Cnt[, , i] <- Ct[, , i]
    }else{
      if(missing(delta)){
        inv_Rtp1 <- chol2inv(chol(Rt[, , i+1]))
        Bt <- Ct[, , i] %*% t(Gt[, , i+1]) %*% inv_Rtp1
        mnt[i, ] <- mt[i, ] + Bt %*% (mnt[i+1, ] - at[i+1, ])
        Cnt[, , i] <- Ct[, , i] + Bt %*% (Cnt[, , i+1] - 
                                            Rt[, , i+1]) %*% t(Bt)
        Cnt[,,i] <- 0.5*Cnt[,,i]+0.5*t(Cnt[,,i])
      }else{
        inv_Gt <- solve(Gt[, , i+1])
        mnt[i, ] <- (1-delta)*mt[i, ] + 
          delta*inv_Gt %*% t(mnt[i+1, ,drop=FALSE])
        Cnt[, , i] <- (1-delta)*Ct[, , i] + 
          delta^2*inv_Gt %*% Cnt[, , i + 1]  %*% t(inv_Gt)
        Cnt[,,i] <- 0.5*Cnt[,,i]+0.5*t(Cnt[,,i])
      }
    }
    fnt[i] <- t(Ft[, , i]) %*% t(mnt[i, , drop=FALSE])
    Qnt[i] <- t(Ft[, , i]) %*% t(Cnt[, , i]) %*% Ft[, , i]
  }
  for(i in 1:T){
    Cnt[,,i]=St[T]*Cnt[,,i]/St[i] 
    Qnt[i]=St[T]*Qnt[i]/St[i]
  }
  cat("Backward smoothing is completed!\n")
  return(list(mnt = mnt, Cnt = Cnt, fnt=fnt, Qnt=Qnt))
}

## Forecast Distribution for k step
forecast_function_unknown_v <- function(posterior_states, k, 
                                        matrices, delta){
  
  ## retrieve matrices
  Ft <- matrices$Ft
  Gt <- matrices$Gt
  if(missing(delta)){
    Wt_star <- matrices$Wt_star
  }
  
  mt <- posterior_states$mt
  Ct <- posterior_states$Ct
  St <- posterior_states$St
  at <- posterior_states$at
  
  ## set up matrices
  T <- dim(mt)[1] # time points
  d <- dim(mt)[2] # dimension of state parameter vector
  
  ## placeholder for results
  at <- matrix(NA, nrow = k, ncol = d)
  Rt <- array(NA, dim=c(d, d, k))
  ft <- numeric(k)
  Qt <- numeric(k)
  
  for(i in 1:k){
    ## moments of state distribution
    if(i == 1){
      at[i, ] <- Gt[, , T+i] %*% t(mt[T, , drop=FALSE])
      
      if(missing(delta)){
        Rt[, , i] <- Gt[, , T+i] %*% Ct[, , T] %*% 
          t(Gt[, , T+i]) + St[T]*Wt_star[, , T+i]
      }else{
        Rt[, , i] <- Gt[, , T+i] %*% Ct[, , T] %*% 
          t(Gt[, , T+i])/delta
      }
      Rt[,,i] <- 0.5*Rt[,,i]+0.5*t(Rt[,,i])
      
    }else{
      at[i, ] <- Gt[, , T+i] %*% t(at[i-1, , drop=FALSE])
      if(missing(delta)){
        Rt[, , i] <- Gt[, , T+i] %*% Rt[, , i-1] %*% 
          t(Gt[, , T+i]) + St[T]*Wt_star[, , T + i]
      }else{
        Rt[, , i] <- Gt[, , T+i] %*% Rt[, , i-1] %*% 
          t(Gt[, , T+i])/delta
      }
      Rt[,,i] <- 0.5*Rt[,,i]+0.5*t(Rt[,,i])
    }
    
    
    ## moments of forecast distribution
    ft[i] <- t(Ft[, , T+i]) %*% t(at[i, , drop=FALSE])
    Qt[i] <- t(Ft[, , T+i]) %*% Rt[, , i] %*% Ft[, , T+i] + 
      St[T]
  }
  cat("Forecasting is completed!\n") # indicator of completion
  return(list(at=at, Rt=Rt, ft=ft, Qt=Qt))
}

## obtain 95% credible interval
get_credible_interval_unknown_v <- function(ft, Qt, nt, 
                                            quantile = c(0.025, 0.975)){
  bound <- matrix(0, nrow=length(ft), ncol=2)
  
  if ((length(nt)==1)){
    for (t in 1:length(ft)){
      t_quantile <- qt(quantile[1], df = nt)
      bound[t, 1] <- ft[t] + t_quantile*sqrt(as.numeric(Qt[t])) 
      
      # upper bound of 95% credible interval
      t_quantile <- qt(quantile[2], df = nt)
      bound[t, 2] <- ft[t] + 
        t_quantile*sqrt(as.numeric(Qt[t]))}
  }else{
    # lower bound of 95% credible interval
    for (t in 1:length(ft)){
      t_quantile <- qt(quantile[1], df = nt[t])
      bound[t, 1] <- ft[t] + 
        t_quantile*sqrt(as.numeric(Qt[t])) 
      
      # upper bound of 95% credible interval
      t_quantile <- qt(quantile[2], df = nt[t])
      bound[t, 2] <- ft[t] + 
        t_quantile*sqrt(as.numeric(Qt[t]))}
  }
  return(bound)
  
}



## Example: Nile River Level (in 10^8 m^3), 1871-1970 
## Model: First order polynomial DLM
plot(Nile) 
n=length(Nile) #n=100 observations 
k=5
T=n-k
data_T=Nile[1:T]
test_data=Nile[(T+1):n]
data=list(yt = data_T)


## set up matrices for first order polynomial model 
Ft=array(1, dim = c(1, 1, n))
Gt=array(1, dim = c(1, 1, n))
Wt_star=array(1, dim = c(1, 1, n))
m0=as.matrix(800)
C0_star=as.matrix(10)
n0=1
S0=10

## wrap up all matrices and initial values
matrices = set_up_dlm_matrices_unknown_v(Ft, Gt, Wt_star)
initial_states = set_up_initial_states_unknown_v(m0, 
                                                 C0_star, n0, S0)

## filtering 
results_filtered = forward_filter_unknown_v(data, matrices, 
                                            initial_states)
ci_filtered=get_credible_interval_unknown_v(results_filtered$mt, 
                                            results_filtered$Ct, 
                                            results_filtered$nt)

## smoothing
results_smoothed=backward_smoothing_unknown_v(data, matrices, 
                                              results_filtered)
ci_smoothed=get_credible_interval_unknown_v(results_smoothed$mnt, 
                                            results_smoothed$Cnt, 
                                            results_filtered$nt[T])

## one-step ahead forecasting
results_forecast=forecast_function_unknown_v(results_filtered, 
                                             k,  matrices)
ci_forecast=get_credible_interval_unknown_v(results_forecast$ft, 
                                            results_forecast$Qt, 
                                            results_filtered$nt[T])


## plot results
index=seq(1871, 1970, length.out = length(Nile))
index_filt=index[1:T]
index_forecast=index[(T+1):(T+k)]

plot(index, Nile, main = "Nile River Level ",type='l',
     xlab="time",ylab="feet",lty=3,ylim=c(400,1500))
points(index,Nile,pch=20)

lines(index_filt,results_filtered$mt, type='l', col='red',lwd=2)
lines(index_filt,ci_filtered[, 1], type='l', col='red', lty=2)
lines(index_filt,ci_filtered[, 2], type='l', col='red', lty=2)
lines(index_filt,results_smoothed$mnt, type='l', col='blue',lwd=2)
lines(index_filt, ci_smoothed[, 1], type='l', col='blue', lty=2)
lines(index_filt, ci_smoothed[, 2], type='l', col='blue', lty=2)

lines(index_forecast, results_forecast$ft, type='l', 
      col='green',lwd=2)
lines(index_forecast, ci_forecast[, 1], type='l', 
      col='green', lty=2)
lines(index_forecast, ci_forecast[, 2], type='l', 
      col='green', lty=2)

# end -------------------------------------------------------------------------#





