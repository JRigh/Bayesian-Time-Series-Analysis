#----------------------------------------#
# Bayesian Time Series Analysis: project #
#----------------------------------------#

install.packages("gtrendsR")
library(gtrendsR)

res = gtrends("timeseries", geo = c("us"), time ="today 12−m")
plot(res)