### ### ### ### ### ### ### ### ### ### ### ### ### ### 
##R CODE COVERING BASICS TO MACHILE LEARNING##
### ### ### ### ### ### ### ### ### ### ### ### ### ### 

####DATA CLEANING AND STRUCTURING####

##1. DATA EXPLORATION

#to find variable class
class(var)

#to convert data types
as.character(var) #to character
as.numeric(var) #to numeric
as.logical(var) #to binary
as.factor(var) #to categorical
as.integer(var) #to numeric int

#to check if data is of specific type
is.character(var)
is.numeric(var)
is.logical(var)
is.factor(var)
is.integer(var)

### ### ### ### ### ### ### ### ### ### ### ### ### ### 

##VECTORS##
##can be of 1 type of class, or different classes

#example of 1 type
vec=c(1,2,3,4)
#example of different types
vec=c(TRUE, 1,2,"Text")

vec*2 ##will multiply each value in vector by 2
vec +2 ##will add each value in the vector by 2
vec >2 ##will compare each value in the vector to 2 and output is a logical T/F
vec%*%vec ##inner multiplication (one value output)
vec%o%vec ##outer multiplication (each specific value multiplied)

##data type for vectors: coercion priority

##if vector has numerics and characters-->class is character
vec1=c(1, 45, "Nikhil")
class(vec1)
##if vector has numerics, logicals and characters--> class is character
vec2=c(2, TRUE, "Nikhil")
class(vec2)
##if vector has numerics and logicals --> class is numeric
vec3=c(2, 45, TRUE)
class(vec3)
##if vector has logicals and characters--> class is character
vec4=c(TRUE, FALSE, "Nikhil")
class(vec4)

##Functions to explore a vector
vector=c(1,2,3,5)
class(vector) #to find the datatype of the vector
length(vector) #to find the length of the vector
max(vector) #find the max value of the vector
which.max(vector) #find the location of the max value of the vector

### ### ### ### ### ### ### ### ### ### ### ### ### ### 

##MATRICES##

#Matrices can only have 1 type of data class
m = matrix(1:8,nrow=2,ncol=4) ##this matrix has values 1-8 (ascending order), spread across 2 rows and 4 columns

m2 = matrix(8:1,nrow=2,ncol=4) ##similar to matrix above, but in descending order (8:1)

#multiplication of matrices
m * m2 #multiplies each value of matrix by its corresponding position in the other matrix

### ### ### ### ### ### ### ### ### ### ### ### ### ### 

##DATAFRAMES##
df= data.frame(id = 1:10, #id is a number from 1-10
               gender=sample(c('Male','Female'), size = 10,replace = T), #gender comes from a vector of males/females, sampled 10x, with replacement
               attended=sample(c(T,F), size = 10,replace=T), #attended comes from a vector of T/F, sampled 10x with replacement
               score=sample(x = 1:100, size = 10,replace = T), #score is a value 1-100, sampled 10x, with replacement
               stringsAsFactors= T) #strings are treated as factors


##Functions to explore a dataframe
#example of df
id = 1:10
age = round(runif(n=10,min = 25,max = 35))
gender = sample(c('Male','Female'),size=10,replace=T)
income = round(runif(n = 10,min = 50000,max = 80000)/1000)*1000
df = data.frame(id,age,gender,income)
                
head(df) #shows the top 6 rows of the dataframe (including col names)
tail(df) #shows the last 6 rows of the dataframe (including col names)
nrow(df) #the number of rows in a df
names(df) #shows the column names
str(df) #shows the data structure of each variable in the df
summary(df) #shows a summary of the data's min,max median and quartile values

### ### ### ### ### ### ### ### ### ### ### ### ### ### 

##SUBSETTING##

#Subsetting for Vectors

#indexing starts from 1
vec=c(1,2,3,4)

vec[1]

#Subsetting for Dataframes
df[c(1,2),4] #c(1,2) is the number of rows; 4 is the column

#use the function "subset" to subset a dataframe
#example
subset(df, gender=="Female") #from the dataframe, pick only rows with gender as female

##LISTS##
lst[[1]] # use this to call c(1,2)
lst[[1]][1] # use this to call the first element of c(1,2)

### ### ### ### ### ### ### ### ### ### ### ### ### ### 

##LOGICAL TESTS##
x=1:7

x<5 #logical test (returns T/F for each value of x if logical condition is met)
sum(x<5) #gets the sum of all values given that they meet the logical test above 

#logical test for ranges
x<5 & x>3 #logical test(returns t/f for each value of x if logical condition is met)
sum(x<5 & x>3) #gets the sum of all values given that they meet the logical tests above

4 %in% x #logical test %in%-means value is in range of x (returns t/f if value meets condition)
sum(4 %in% x) #get sum of values if it meets condition above

### ### ### ### ### ### ### ### ### ### ### ### ### ### 

##READ DATA INTO R##
read.table() #read table
read.csv() #read csv
read.delim() #read delimited file

#To read csv with filepath directly set in the function
data = read.csv('c:/myfiles/thatAwesomeClass/data.csv')

#to read csv after having set the file directory
setwd('c:/myfiles/thatAwesomeClass/')
data = read.csv('data.csv')

### ### ### ### ### ### ### ### ### ### ### ### ### ### 
### ### ### ### ### ### ### ### ### ### ### ### ### ### 

####DATA TRANSFORMATION AND TIDYING####

##2. DATA STRUCTURING

##Wide to tall data
library(tidyr)

gdp_tall = #new dataframe
  gdp %>%  #original dataframe piped in to gather function
  gather('Year','GDP',2:3)

##Tall to wide data
gdp_wide = #new dataframe
  gdp_tall %>% #original tall dataframe piped in to spread function
  spread('Year','GDP')

#Gather data and then show the top 6 rows
data %>% #df
  gather('State','Number_of_Violent_Crimes',2:52)%>% #piped in gather function
  head() #show top 6 rows

### ### ### ### ### ### ### ### ### ### ### ### ### ### 

#Tidying and visualizing data
library(tidyr); library(dplyr); library(ggplot2)

data %>% #df
  gather('State','Number_of_Violent_Crimes',2:52)%>% #piped in gather function
  group_by(State)%>% #group by column State
  summarize(AverageViolentCrime = mean(Number_of_Violent_Crimes,na.rm=T))%>% #average violent crime col as a mean of no of violent crimes col
  ggplot(aes(x=reorder(State,X = AverageViolentCrime), y=AverageViolentCrime,size=AverageViolentCrime,color=AverageViolentCrime))+
  geom_point()+scale_color_continuous(low='white',high='red')+xlab('State')+ylab('Crime')+
  coord_flip() #plot visualization

##Variable rescaling
coupons_scaled = apply(coupons,2,scale)
apply(coupons_scaled,2,function(x) round(mean(x),2)) #mean of variable after rescaling
apply(coupons_scaled,2,sd) #sd of variable after rescaling

#Rescaling and averaging variables using dplyr
library(dplyr)
coupons_scaled_dplyr = 
  coupons %>%
  mutate(c1_scaled=scale(c1),
         c2_scaled=scale(c2),
         c3_scaled=scale(c3),
         c4_scaled=scale(c4),
         c5_scaled=scale(c5))%>%
  select(c1_scaled:c5_scaled)%>%
  rowwise()%>%
  mutate(coupons_avg = mean(c1_scaled:c5_scaled))
coupons_scaled_dplyr

### ### ### ### ### ### ### ### ### ### ### ### ### ### 

##Descriptive Measures for Numeric variables##

#Mean: average of all values

#approach 1#
mean(mtcars$mpg)

#approach 2#
library(dplyr)
mtcars%>%
  summarize(mean(mpg))

#Median: middle observation of sorted obs

#approach 1#
median(mtcars$mpg)

#approach 2#
library(dplyr)
mtcars%>%
  summarize(median(mpg))

#Mode: Most frequent value (R doesn't have a built in function, so a new function has to be set up)
calculate_mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

calculate_mode(mtcars$mpg)

#using dplyr
mtcars%>%
  summarize(calculate_mode(mpg))

#Range: Difference between maximum and minimum

#approach 1#
range(mtcars$mpg)

#approach 2#
max(mtcars$mpg) - min(mtcars$mpg)

#approach 3#
# Using dplyr
mtcars%>%
  summarize(range = max(mpg) - min(mpg))

#5th Percentile: Value such that 5% values are below it

#approach 1#
quantile(mtcars$mpg,0.05)

#approach 2#
# Using dplyr
mtcars%>%
  summarize(quantile(mpg,0.05))

#Interquartile range: Difference between values at 25th and 75th percentiles

#approach 1#
quantile(mtcars$mpg,0.75) - quantile(mtcars$mpg,0.25)

#approach 2#
IQR(mtcars$mpg)

#approach 3#
# Using dplyr
mtcars%>%
  summarize(IQR(mpg))

#Variance: Average of squared deviations from mean

#approach 1#
var(mtcars$mpg)

#approach 2#
# Using dplyr
mtcars%>%
  summarize(var(mpg))

#Standard deviation: Square root of variance

#approach 1#
sd(mtcars$mpg)

#approach 2#
# Using dplyr
mtcars%>%
  summarize(sd(mpg))

### ### ### ### ### ### ### ### ### ### ### ### ### ### 

##Descriptive Measures for Categorical variables##

##Compare categories by count

#approach 1#
table(mtcars$cyl)

#approach 2#
mtcars%>%
  group_by(cyl)%>%
  summarize(cyl_count = n())

#find descriptive measure of cat. variables

#approach 1# 
tapply(mtcars$mpg,mtcars$cyl,'mean')
tapply(mtcars$mpg,mtcars$cyl,'median')

#approach 2#
#using dplyr#
mtcars%>%
  group_by(cyl)%>%
  summarize(avg_mpg = mean(mpg))

mtcars%>%
  group_by(cyl)%>%
  summarize(avg_mpg = median(mpg))


### ### ### ### ### ### ### ### ### ### ### ### ### ### 

##Missing Values##

#ignore missing values (NAs)#
mtcars_dropped_missing = mtcars[!is.na(mtcars$mpg),] #create new df with na's dropped

#Use functions that contain arguments to ignore missing values
mean(mtcars$mpg,na.rm = T)

#Impute Missing Values using mice
install.packages('mice')
library(mice)

mtcars_mice = complete(mice(mtcars)) #new df completed of missing values

#Impute missing values using caret
library(caret)
mtcars_caret = predict(preProcess(mtcars,method = 'medianImpute'),newdata = mtcars)
mtcars_caret$mpg

#Results of imputation
data.frame(original = mtcars_original$mpg,
           missing = mtcars$mpg,
           mice_impute = mtcars_mice$mpg,
           caret_impute = mtcars_caret$mpg)


### ### ### ### ### ### ### ### ### ### ### ### ### ### 

##Visualization##

library(ggplot2)

##Main components of ggplot2 * Data * Aesthetic mapping (aes) +
##Describes how variables are mapped onto graphical attributes + 
##Visual attribute of data including x-y axes, color, fill, shape, and alpha. 
##* Geometric objects (geom) + Determines how values are rendered graphically, 
##as bars (geom_bar), scatterplot (geom_point), line (geom_line), etc.

##Histogram##
ggplot(data=mpg,aes(x=hwy))+
  geom_histogram()

#histogram using aesthetics
ggplot(data=mpg,aes(x=hwy,fill=factor(year)))+
  geom_histogram()

##Frequency Polygons##
ggplot(data=mpg,aes(x=hwy,color=factor(year)))+
  geom_freqpoly(size=1.2)

#with binwidth
ggplot(data=mpg,aes(x=hwy,color=factor(year)))+
  geom_freqpoly(size=1.2, bins=10)

##Density Curve##
ggplot(data=mpg,aes(x=hwy,color=factor(drv)))+
  geom_freqpoly(size=1.2)

#with geom_density
ggplot(data=mpg,aes(x=hwy,color=factor(drv)))+
  geom_density(size=1.2)

##Boxplots##
ggplot(data=mpg,aes(x="",y=hwy))+
  geom_boxplot()

#factor by specific variable (in this case, year)
ggplot(data=mpg,aes(x=factor(year),y=hwy))+
  geom_boxplot(outlier.color='red')+
  geom_text(data=mpg[mpg$hwy>quantile(mpg$hwy,0.99),], aes(label=manufacturer),nudge_x = 0.2)

##QQ Plot##
qqnorm(mpg$hwy)
qqline(mpg$hwy)

##Bar Charts##
ggplot(data=mpg,aes(x=factor(year),y=hwy))+
  geom_bar(stat = 'summary',fun.y='median')

#factored by specific variable (in this case, year)
ggplot(data=mpg,aes(x=factor(cyl),fill=factor(year),y=hwy))+
  geom_bar(stat = 'summary',fun.y='mean')

#dodging
ggplot(data=mpg,aes(x=factor(cyl),fill=factor(year),y=hwy))+
  geom_bar(stat = 'summary',fun.y='mean',position='dodge')

#dropping a specific value within variable
ggplot(data=mpg[!mpg$cyl==5,],aes(x=factor(cyl),fill=factor(year),y=hwy))+
  geom_bar(stat = 'summary',fun.y='mean',position='dodge')

#Grid positioning
set.seed(100)
#all 5 charts
data = data.frame(hi_lo=sample(c('high','low'),20,replace = T),y=1:20,on_off=rep(c('on','off'),10))
g1 = ggplot(data=data,aes(x=hi_lo,y=y,fill=on_off))+
  geom_bar(stat = 'summary',fun.y=mean,position='stack')+
  ggtitle(label='position = stack')+guides(fill=F)
g2 = ggplot(data=data,aes(x=hi_lo,y=y,fill=on_off))+
  geom_bar(stat = 'summary',fun.y=mean,position='dodge')+
  ggtitle(label='position = dodge')+guides(fill=F)
g3 = ggplot(data=data,aes(x=hi_lo,y=y,fill=on_off))+
  geom_bar(stat = 'summary',fun.y=mean,position='fill')+
  ggtitle(label='position = fill')+guides(fill=F)
g4 = ggplot(data=data,aes(x=hi_lo,y=y,fill=on_off))+
  geom_bar(stat = 'summary',fun.y=mean,position='jitter')+
  ggtitle(label='position = jitter')+guides(fill=F)
g5 = ggplot(data=data,aes(x=hi_lo,y=y,fill=on_off))+
  geom_bar(stat = 'summary',fun.y=mean,position='identity')+
  ggtitle(label='position = identity')+guides(fill=F)

#arranging charts in a grid
library(gridExtra)
grid.arrange(g1, g2, g3, g4, g5)

##Scatterplot##
ggplot(data=mpg,aes(x=displ,y=cty))+
  geom_point()

#classification by color
ggplot(data=mpg,aes(x=displ,y=cty,color=class))+
  geom_point()

#classification by size
ggplot(data=mpg,aes(x=displ,y=cty,size=class))+
  geom_point()

#classification by shape
ggplot(data=mpg,aes(x=displ,y=cty,shape=class))+
  geom_point()

#classification by alpha
ggplot(data=mpg,aes(x=displ,y=cty,alpha=class))+
  geom_point()

##Facets##
ggplot(data=mpg,aes(x=displ,y=hwy))+
  geom_point()+
  facet_grid(.~cyl) #faceted base on cyl type vertically

ggplot(data=mpg,aes(x=displ,y=hwy))+
  geom_point()+
  facet_grid(cyl~.) #faceted base on cyl type horizontally

#facet_wrap: will organize charts left to right and 
#then start again in the next line from left to right.
ggplot(data=mpg,aes(x=displ,y=hwy))+
  geom_point()+
  facet_wrap(~cyl)

#two variable facets
ggplot(data=mpg,aes(x=displ,y=hwy))+
  geom_point()+
  facet_grid(cyl~year)

##scatterplot and line graph
ggplot(data=mpg,aes(x=displ,y=hwy))+
  geom_point()+
  geom_smooth(method='lm')

ggplot(data=mpg,aes(x=displ,y=hwy))+
  geom_point()+
  geom_smooth(method='lm')+
  geom_hline(yintercept=mean(mpg$hwy))






####SAMPLE SPLITTING####

#Simple Random Sampling#
set.seed(100)
split = sample(x = nrow(diamonds),size = 0.7*nrow(diamonds)) #Split 70-30

train = diamonds[split,]
test = diamonds[-split,]

#find nr of rows for train set
nrow(train)

#find nr of rows for test set
nrow(test)

#difference between means of train and test outcome variable
mean(train$price) - mean(test$price)

##Stratified random sampling (numeric outcome)#
library(caret)

set.seed(100)
split = createDataPartition(y = diamonds$price,p = 0.7,list = F, groups = 50)

train = diamonds[split,]
test = diamonds[-split,]

##Stratified random sampling (categorical outcome)#
set.seed(100)
split = createDataPartition(y = diamonds$price_hilo,p = 0.7,list = F)
train = diamonds[split,]
test = diamonds[-split,]

#for categorical value, look at value counts
table(train$price_hilo); table(test$price_hilo)

#counts of proportions
prop.table(rbind(train = table(train$price_hilo), 
                 test = table(test$price_hilo)),margin = 1)

##Stratified Random Sampling (with a categorical outcome) using caTools
library(caTools)
set.seed(100)
split = sample.split(Y = diamonds$price_hilo, SplitRatio = 0.7)

#The only difference is that sample.split() will generate a logical, not a vector of numbers.
table(split)

train = diamonds[split,]
test = diamonds[!split,]



####LINEAR REGRESSION MODELING####
##Modeling Wages  ##

## Read Data
# Be sure to add your complete file path or setwd to where the data file is
wages = read.csv('wages.csv')
# Source: Simulated dataset based on a real dataset in Data Analysis using Regression and Multilevel/Hierarchical Models by Andrew Gelman and Jennifer Hill


### ### ### ## ## ### ### ###
##### Explore and Clean Data
### ### ### ## ## ### ### ###

str(wages)
head(wages)
summary(wages)

# Examine distribution of Earn
ggplot(data=wages,aes(x=earn))+
  geom_histogram(binwidth=5000,fill='cadetblue')

# remove negative earn
sum(wages$earn<0)
wages = wages[wages$earn>=0,]

# Examine outliers
ggplot(data=wages,aes(x='',y=earn))+
  geom_boxplot(outlier.color='red',outlier.alpha=0.5, fill='cadetblue')+
  geom_text(aes(x='',y=median(wages$earn),label=median(wages$earn)),size=3,hjust=11)+
  xlab(label = '')



#### ### ### ### ## ## ### ### ###
###### Model 1: Simple Regression
# earn = f(age)
### ### ### ## ## ### ### ###

# Scatterplot
library(ggplot2)
# Is there a discernible trend
ggplot(data=wages,aes(x=age,y=earn))+
  geom_point()+
  coord_cartesian(ylim=c(0,200000))

# Is there a correlation?
cor(wages$age,wages$earn)


ggplot(data=wages,aes(x=age,y=earn))+
  geom_point()+
  geom_smooth(method='lm',size=1.3,color='steelblue3')+
  coord_cartesian(ylim=c(0,200000))

# Estimate
model1 = lm(earn~age,data=wages)
paste('earn','=',round(coef(model1)[1],0),'+',round(coef(model1)[2],0),'age')

# Prediction
anova(model1)
summary(model1)

pred = predict(model1)
data.frame(earn = wages$earn[100:109], prediction = pred[100:109])

sse = sum((pred - wages$earn)^2)
sst = sum((mean(wages$earn)-wages$earn)^2)
model1_r2 = 1 - sse/sst; model1_r2
sse1 = sum((pred-wages$earn)^2); sse1
rmse1 = sqrt(mean((pred-wages$earn)^2)); rmse1

# Inference
# Does age influence earn?
# What would be the wage of a 35 year old person
model1$coef[1]+ model1$coef[2]*35  # doing it manually 
predict(model1,newdata=data.frame(age=35)) # or using the predict function

# How much more would a 45 year old than a 35 year old
# Generally speaking, what would a 10 year increase in age do to one's wage

### ### ### ## ## ### ### ###
###### Model 2: Simple Regression
# earn = f(height)
### ### ### ## ## ### ### ###

## Visualize
ggplot(aes(x=height,y=earn),data=wages)+
  geom_point()+
  geom_smooth(method="lm",size=1.3,color='steelblue3',se=FALSE)+
  coord_cartesian(ylim = c(0,200000))

# Estimate Model
model2 = lm(earn~height,data=wages)
paste('earn','=',round(coef(model2)[1],0),'+',round(coef(model2)[2],0),'height')

# Predict
summary(model2)

pred = predict(model2)
data.frame(earn = wages$earn[100:109], prediction = pred[100:109])

sse2 = sum((pred - wages$earn)^2)
sst2 = sum((mean(wages$earn)-wages$earn)^2)
model2_r2 = 1 - sse2/sst2; model2_r2

rmse2 = sqrt(mean((pred-wages$earn)^2)); rmse2

## Inference
# Does height influence earn?
# What impact will a two inch increase in height have on your wage?
2 * coef(model2)[2]
# How much will a six foot person earn (all else being equal)?
predict(model2,newdata=data.frame(height=72))

### ### ### ## ## ### ### ###
###### Model 3: Simple Regression (Categorical Predictor)
# earn = f(sex)
### ### ### ## ## ### ### ###

# Simple Regression with a Categorical Variable
# When exploring the relationship with a categorical variable, a scatterplot is not an option. Bar chart makes more sense.
ggplot(data=wages,aes(x=sex,y=earn,fill=sex))+
  geom_bar(stat='summary',fun.y='mean',position='dodge')+
  guides(fill=F)

# bar chart with error bars
wages%>%
  group_by(sex)%>%
  summarise(meanEarn=mean(earn),earnLow=mean(earn)-1.96*sd(earn)/sqrt(n()),earnHigh=mean(earn)+1.96*sd(earn)/sqrt(n()))%>%
  ungroup()%>%
  ggplot(aes(x=sex,y=meanEarn))+
  geom_errorbar(aes(ymin=earnLow,ymax=earnHigh),size=1.1)+
  geom_line(aes(x=sex,y=meanEarn,group=1),linetype=3)+
  geom_bar(data=wages,aes(x=sex,y=earn,alpha=0.2,fill=sex),stat='summary',fun.y='mean',position='dodge')+
  guides(fill=F,alpha=F)


# Estimate
### Now, construct and interpret a simple regression using sex  as IV
## Note sex is a categorical variable
model3 = lm(earn~sex,data=wages)
class(wages$sex) # factor
levels(wages$sex)
table(wages$sex) # for an unordered factor, levels are in alphabetical order

# Prediction
summary(model3)
pred = predict(model3)
data.frame(earn = wages$earn[100:109], prediction = pred[100:109])

sse3 = sum((pred - wages$earn)^2)
sst3 = sum((mean(wages$earn)-wages$earn)^2)
model3_r2 = 1 - sse3/sst3; model3_r2

rmse3 = sqrt(mean((pred-wages$earn)^2)); rmse3

# Inference
#Do females earn more than males? What is the difference?

#View results using anova table
anova(linearModel3)
# visualizing data can show things that the indices may not reveal
#Dig deeper by plotting a density plot. What do you find?
ggplot(data=wages,aes(x=earn,color=sex))+
  geom_density(size=1.1)


### ### ### ## ## ### ### ###
###### Model 4: Simple Regression (categorical predictor)
# earn = f(race)
### ### ### ## ## ### ### ###


## Examine relationship between race and earn
# Estimate Model
ggplot(data=wages,aes(x=race,y=earn,fill=race))+
  geom_bar(stat='summary',fun.y='mean',position='dodge')+
  guides(fill=F)
class(wages$race)
levels(wages$race)

table(wages$race)

model4 = lm(earn~race,wages)

# Prediction
summary(model4)

pred = predict(model4)
data.frame(earn = wages$earn[100:109], prediction = pred[100:109])

sse4 = sum((pred - wages$earn)^2)
sst4 = sum((mean(wages$earn)-wages$earn)^2)
model4_r2 = 1 - sse4/sst4; model4_r2

rmse4 = sqrt(mean((pred-wages$earn)^2)); rmse4

# Inference
# Does race influence earn?
# On average, how much do those with race "african-american" earn 
# IF race influenced earn, who would you say earns more, whites or asian and what is the difference?

### ### ### ## ## ### ### ###
###### Model 5: Multiple Regression
# earn = f(height, sex)
### ### ### ## ## ### ### ###

# Estimate
model5 = lm(earn~height+sex,data=wages)

# Prediction
summary(model5)

pred = predict(model5)

sse5 = sum((pred - wages$earn)^2)
sst5 = sum((mean(wages$earn)-wages$earn)^2)
model5_r2 = 1 - sse5/sst5; model5_r2

rmse5 = sqrt(mean((pred-wages$earn)^2)); rmse5

# How do the prediction metrics compare to those for simple regressions using just height or sex. 

# Inference
# Does height have an effect on earn. Does sex have an effect on earn
# What is the effect of a 4 inch height difference on earn, while holding sex constant. 
# Which variable is a stronger predictor
library(lm.beta)
lm.beta(model5)

### ### ### ## ## ### ### ###
###### Model 6: Multiple Regression
# earn = earn = f(height, sex, race, ed, age)
### ### ### ## ## ### ### ###

# Estimate
model6 = lm(earn~height+sex+race+ed+age,data=wages)
# Prediction
summary(model6)

pred = predict(model6)

sse6 = sum((pred - wages$earn)^2)
sst6 = sum((mean(wages$earn)-wages$earn)^2)
model6_r2 = 1 - sse6/sst6; model6_r2

rmse6 = sqrt(mean((pred-wages$earn)^2)); rmse6

# Inference
# Interpret individual Coefficients
# sometimes we want to compare effect of variables
library(lm.beta)
lm.beta(model6)
# Based on the model, how much will a 22 year old, 64 inches tall, White Female with 16 yrs of ed earn?

### ### ### ## ## ### ### ###
###### Model 7: Multiple Regression (with Interaction)
# earn = f(age, sex, age*sex)
### ### ### ## ## ### ### ###

# Estimate
model7 = lm(earn~age+sex+age:sex,data=wages)

# Prediction
summary(model7)

pred = predict(model7)

sse7 = sum((pred - wages$earn)^2)
sst7 = sum((mean(wages$earn)-wages$earn)^2)
model7_r2 = 1 - sse7/sst7; model7_r2

rmse7 = sqrt(mean((pred-wages$earn)^2)); rmse7

# Inference
# Do age and sex interact in influencing earn?
# Statisticians recommend not interpreting the main effects when there is an interaction. Let's see why.
ggplot(aes(x=age,y=earn),data=wages)+
  geom_point(aes(color=sex))+
  geom_smooth(aes(color=sex),method="lm",size=1.1,se=F)+
  coord_cartesian(ylim = c(1,200000))
#  Age is positively related to earn BUT only for males.  



### ### ### ## ## ### ### ###
###### Model 8: Non-linear Regression
# earn = f(age, age^2)
### ### ### ## ## ### ### ###

# Estimate
## Nonlinear (Polynomial) regression
model8 = lm(earn~poly(age,2),data=wages)

# Prediction
# How does this R2 compare to that for just model1
summary(model8)

pred = predict(model8)
sse8 = sum((pred - wages$earn)^2)
sst8 = sum((mean(wages$earn)-wages$earn)^2)
model8_r2 = 1 - sse8/sst8; model8_r2
rmse8 = sqrt(mean((pred-wages$earn)^2)); rmse8

ggplot(aes(x=age,y=earn),data=wages)+
  geom_point()+
  geom_smooth(method="lm", formula=y~poly(x,2),size=1.3,se=FALSE, color='steelblue3')+
  coord_cartesian(ylim = c(1,200000))

ggplot(data=wages,aes(x=age,y=earn))+
  geom_point()+
  geom_smooth(method='lm',size=1.3,color='steelblue3', se=FALSE)+
  coord_cartesian(ylim=c(0,200000))


### ### ### ## ## ### ### ###
###### Model 9: Multiple Regression
# Estimate on train sample, evaluate on test sample
# earn = f(height, sex, race, ed, age)
### ### ### ## ## ### ### ###


# Split Sample
set.seed(100) # set seed to ensure results can be replicated
split = sample(1:nrow(wages),0.7*nrow(wages))
train = wages[split,]
test = wages[-split,]
nrow(train)
nrow(test)

# Estimate model on train sample
model9 = lm(earn~height+sex+race+ed+age,data=train)


# Prediction

# in-sample metrics
summary(model9)

pred = predict(model9)
sse9 = sum((pred - train$earn)^2)
sst9 = sum((mean(train$earn)-train$earn)^2)
model9_r2 = 1 - sse9/sst9; model9_r2
rmse9 = sqrt(mean((pred-train$earn)^2)); rmse9

# out of sample metrics
pred = predict(model9, newdata=test)
sse9_test = sum((pred - test$earn)^2)
sst9_test = sum((mean(train$earn)-test$earn)^2)
model9_r2_test = 1 - sse9_test/sst9_test; model9_r2_test
rmse9_test = sqrt(mean((pred-test$earn)^2)); rmse9_test


####  Results
model = c('model1', 'model2', 'model3', 'model4', 'model5', 'model6', 'model7', 'model8', 'model9')
sse = c(sse1, sse2, sse3, sse4, sse5, sse6, sse7, sse8, sse9)
rmse = c(rmse1, rmse2, rmse3, rmse4, rmse5, rmse6, rmse7, rmse8, rmse9)
r2 = c(model1_r2, model2_r2, model3_r2, model4_r2, model5_r2, model6_r2, model7_r2, model8_r2, model9_r2)
results = data.frame(model, sse, rmse, r2)

results
results%>%
  gather(key = metric, value = values,2:4)%>%
  ggplot(aes(x=model, y=values))+
  geom_bar(stat='identity', fill='cadetblue')+
  facet_grid(metric~., scales = 'free_y')

####LOGISTIC REGRESSION MODELING####

#what is mean of startprice for devices sold
tapply(train$startprice,train$sold,mean)

#visualize start price
ggplot(data=train,aes(x=factor(sold),y=startprice,fill=factor(sold)))+
  geom_bar(stat='summary',fun.y='mean')

#what is mean of biddable for devices sold
tapply(train$sold,train$biddable,mean)

#visualize biddable
ggplot(data=train,aes(x=biddable,y=sold,fill=biddable))+
  geom_bar(stat='summary',fun.y='mean')

##One variable model##
model1 = glm(sold~startprice,data=train,family='binomial')
summary(model1)

summary(model1)$coef[2] # coefficient for startprice

exp(summary(model1)$coef[2])

#Given below is the percent increase in likelihood an iPad being sold with a $1 increase in price
100*(exp(summary(model1)$coef[2])-1)

#What is the probability of an iPad priced at $200 selling?
#approach 1#
exp(model1$coef[1] + model1$coef[2]*200)/(1+exp(model1$coef[1] + model1$coef[2]*200))
#approach 2#
predict(model1,newdata=data.frame(startprice=200),type='response') 

##Model 2##
model2 = glm(sold~storage,data=train,family='binomial')
summary(model2) 

#How many times better is the chance of selling a 16/32/64GB iPad relative to 128GB
exp(summary(model2)$coef[2]) 

#what % is the chance of selling a 16/32/64 GB better than 128GB
100*(exp(summary(model2)$coef[2])-1)

##Performance Assessment##
model3 = glm(sold~startprice+biddable+condition+cellular+carrier+
               color+storage+productline+noDescription+upperCaseDescription+
               startprice_99end,data=train,family='binomial')

pred = predict(model3,type='response')

ggplot(data=data.frame(pred),aes(x=pred))+
  geom_histogram(fill='steelblue3')

table(as.numeric(pred>0.5)) # how many predictions for sold

ct = table(train$sold,pred>0.5);ct # classification table

accuracy = sum(ct[1,1],ct[2,2])/nrow(train); accuracy #accuracy

specificity = ct[1,1]/sum(ct[1,1],ct[1,2]); specificity #specificity

sensitivity = ct[2,2]/sum(ct[2,1],ct[2,2]); sensitivity #sensitivity

#comparison to baseline model

t = table(train$sold)
baseline = max(t[1],t[2])/nrow(train); baseline #baseline

#baseline for test sample
baseline = table(test$sold)[1]/nrow(test); baseline

##ROC Curve##
library(ROCR)
ROCRpred = prediction(pred,test$sold)
as.numeric(performance(ROCRpred,"auc")@y.values) # auc measure

ROCRperf = performance(ROCRpred,"tpr","fpr")
plot(ROCRperf) # basic plot

plot(ROCRperf,xlab="1 - Specificity",ylab="Sensitivity") # relabeled axes

plot(ROCRperf,colorize=TRUE) # color coded ROC curve
plot(ROCRperf,colorize=TRUE,print.cutoffs.at=seq(0,1,0.2),text.adj=c(-0.3,2)) # color coded and annotated ROC curve

#ROC for baseline model#
baselinePred = pred*0
ROCRpred = prediction(baselinePred,test$sold)
as.numeric(performance(ROCRpred,"auc")@y.values) # auc measure

## construct plot
ROCRperf = performance(ROCRpred,"tpr","fpr")
plot(ROCRperf) # basic plot
plot(ROCRperf,xlab="1 - Specificity",ylab="Sensitivity") # relabeled axes





####FEATURE SELECTION####

#split data#
library(caret)
set.seed(100)
split = createDataPartition(y=wine$quality,p = 0.7,list = F,groups = 100)
train = wine[split,]
test = wine[-split,]

#correlation of variables
cor(train[,-12])

#round the corr values
round(cor(train[,-12]), 2)*100

#visualizing correlation
library(tidyr); library(dplyr); library(ggplot2)
corMatrix = as.data.frame(cor(train[,-12]))
corMatrix$var1 = rownames(corMatrix)
corMatrix %>%
  gather(key=var2,value=r,1:11)%>%
  ggplot(aes(x=var1,y=var2,fill=r))+
  geom_tile()+
  geom_text(aes(label=round(r,2)),size=3)+
  scale_fill_gradient2(low = 'red',high='green',mid = 'white')+
  theme(axis.text.x=element_text(angle=90))

##correlation matrix
library(corrplot)
corrplot(cor(train[,-12]),method = 'square',type = 'lower',diag = F)

#Model with train data#
model = lm(quality~.,train)
summary(model)

#calculate VIF
#Threat of collinearity can also come from linear relationships between sets of variables. 
#One way to assess the threat of multicollinearity in a linear regression is to compute 
#the Variance Inflating Factor (VIF). 1<VIF<Inf. 
#VIF>10 indicates seious multicollinearity. VIF>5 may warrant examination.

library(car)
vif(model)

#Subsetting#
# install.packages('leaps')
library(leaps)
subsets = regsubsets(quality~.,data=train, nvmax=11)
summary(subsets)

names(summary(subsets))

subsets_measures = data.frame(model=1:length(summary(subsets)$cp),cp=summary(subsets)$cp,
                              bic=summary(subsets)$bic, adjr2=summary(subsets)$adjr2)
subsets_measures

#visualization

library(ggplot2)
library(tidyr)
subsets_measures %>%
  gather(key = type, value=value, 2:4)%>%
  ggplot(aes(x=model,y=value))+
  geom_line()+
  geom_point()+
  facet_grid(type~.,scales='free_y')

#Regsubsets solution with lowest cp
which.min(summary(subsets)$cp)

#coefficients for regsubsets solution with lowest cp
coef(subsets,which.min(summary(subsets)$cp))

##Stepwise function##

##Forward stepwise##
start_mod = lm(quality~1,data=train)
empty_mod = lm(quality~1,data=train)
full_mod = lm(quality~.,data=train)
forwardStepwise = step(start_mod,scope=list(upper=full_mod,lower=empty_mod),direction='forward')

##Backward stepwise##
start_mod = lm(quality~.,data=train)
empty_mod = lm(quality~1,data=train)
full_mod = lm(quality~.,data=train)
backwardStepwise = step(start_mod,scope=list(upper=full_mod,lower=empty_mod),direction='backward')

##Hybrid stepwise##
start_mod = lm(quality~1,data=train)
empty_mod = lm(quality~1,data=train)
full_mod = lm(quality~.,data=train)
hybridStepwise = step(start_mod,scope=list(upper=full_mod,lower=empty_mod),direction='both')

##Ridge##
#install.packages('glmnet')
library(glmnet)
x = model.matrix(quality~.-1,data=train)
y = train$quality
ridgeModel = glmnet(x,y,alpha=0)
ridgeModel

plot(ridgeModel,xvar='lambda',label=T)

cv.ridge = cv.glmnet(x,y,alpha=0) # default is 10-fold cross validation
plot(cv.ridge)

##Lasso##
lassoModel = glmnet(x,y, alpha=1) # Note default for alpha is 1 which corresponds to Lasso
lassoModel

plot(lassoModel,xvar='lambda',label=T)

plot(lassoModel,xvar='dev',label=T)

cv.lasso = cv.glmnet(x,y,alpha=1) # 10-fold cross-validation
plot(cv.lasso)


####DECISION TREES####
library(rpart)
library(rpart.plot)

##Regression Tree##

tree1 = rpart(sold~startprice,data=train) # default method='anova' used

rpart.plot(tree1) #plot the tree
summary(tree1)

##Prediction for regression tree##
pred = predict(tree1,newdata=test)

##Classification Tree##
tree1 = rpart(sold~startprice,data=train,method='class')

rpart.plot(tree1)
summary(tree1)

##Complex Tree##
tree1Complex = rpart(sold~startprice,data=train,cp=0.005) # default method='anova' used
rpart.plot(tree1Complex)

#Predictions for classification tree#
pred = predict(tree1,newdata=test,type='class')
ct = table(test$sold,pred); ct

accuracy = sum(ct[1,1],ct[2,2])/nrow(test); accuracy ##accuracy test
specificity = ct[1,1]/sum(ct[1,1],ct[1,2]); specificity ##specificity test
sensitivity = ct[2,2]/sum(ct[2,1],ct[2,2]); sensitivity ##sensitivity test

##Complex trees with parameters##
tree3Complex = rpart(sold~startprice+biddable+condition+cellular+carrier+color+storage+
                       productline+noDescription+upperCaseDescription+startprice_99end,
                     data=train,method='class',control=rpart.control(minbucket = 25))

##ROC curve##
library(ROCR)
ROCRpred = prediction(pred,test$sold)
as.numeric(performance(ROCRpred,"auc")@y.values) # auc measure

## construct plot
ROCRperf = performance(ROCRpred,"tpr","fpr")
plot(ROCRperf) # basic plot




####ADVANCED DECISION TREES####

#find RMSE of tree
tree = rpart(earn~.,data=train)
predTree = predict(tree,newdata=test)
rmseTree = sqrt(mean((predTree-test$earn)^2)); rmseTree

##maximal tree
maximalTree = rpart(earn~.,data=train,control=rpart.control(minbucket=1))
predMaximalTree = predict(maximalTree,newdata=test)
rmseMaximalTree = sqrt(mean((predMaximalTree-test$earn)^2)); rmseMaximalTree

#10-fold validation of tree
library(caret)
trControl = trainControl(method="cv",number = 10)
tuneGrid = expand.grid(.cp = seq(0.001,0.1,0.001))
set.seed(100)
cvModel = train(earn~.,data=train,method="rpart",
                trControl = trControl,tuneGrid = tuneGrid)
cvModel$bestTune

treeCV = rpart(earn~.,data=train,
               control=rpart.control(cp = cvModel$bestTune))
predTreeCV = predict(treeCV,newdata=test)
rmseCV = sqrt(mean((predTreeCV-test$earn)^2)); rmseCV

##Bagging##
library(randomForest)
set.seed(100)
bag = randomForest(earn~.,data=train,mtry = ncol(train)-1,ntree=1000)
predBag = predict(bag,newdata=test)
rmseBag = sqrt(mean((predBag-test$earn)^2)); rmseBag

plot(bag)

varImpPlot(bag); importance(bag)  ## see variable importance

getTree(bag,k=100)   # View Tree 100

hist(treesize(bag))  # size of trees constructed 

##Randomforest##
# install.packages('raondomForest')
library(randomForest)
set.seed(100)
forest = randomForest(earn~.,data=train,ntree = 1000)
predForest = predict(forest,newdata=test)
rmseForest = sqrt(mean((predForest-test$earn)^2)); rmseForest

names(forest)

summary(forest)
plot(forest)

varImpPlot(forest); importance(forest)  ## see variable importance

getTree(forest,k=100)   # View Tree 100

hist(treesize(forest))  # size of trees constructed 

##Random forest with cross validation##
trControl=trainControl(method="cv",number=10)
tuneGrid = expand.grid(mtry=1:5)
set.seed(100)
cvForest = train(earn~.,data=train,
                 method="rf",ntree=1000,trControl=trControl,tuneGrid=tuneGrid )
cvForest  # best mtry was 2

set.seed(100)
forest = randomForest(earn~.,data=train,ntree = 100,mtry=2)
predForest = predict(forest,newdata=test)
rmseForest = sqrt(mean((predForest-test$earn)^2)); rmseForest

##Boosting##
library(gbm)

set.seed(100)
boost = gbm(earn~.,data=train,distribution="gaussian",
            n.trees = 100000,interaction.depth = 3,shrinkage = 0.001)
predBoostTrain = predict(boost,n.trees = 100000)
rmseBoostTrain = sqrt(mean((predBoostTrain-train$earn)^2)); rmseBoostTrain

summary(boost) ##to see variable ranking

#predict on test data
predBoost = predict(boost,newdata=test,n.trees = 10000)
rmseBoost = sqrt(mean((predBoost-test$earn)^2)); rmseBoost

##Boosting with cross-validation##
set.seed(100)
trControl=trainControl(method="cv",number=10)
tuneGrid=  expand.grid(n.trees = 1000, interaction.depth = c(1,2),
                       shrinkage = (1:100)*0.001,n.minobsinnode=5)
cvBoost = train(earn~.,data=train,method="gbm", 
                trControl=trControl, tuneGrid=tuneGrid)

boostCV = gbm(earn~.,data=train,distribution="gaussian",
              n.trees=cvBoost$bestTune$n.trees,
              interaction.depth=cvBoost$bestTune$interaction.depth,
              shrinkage=cvBoost$bestTune$shrinkage,
              n.minobsinnode = cvBoost$bestTune$n.minobsinnode)
predBoostCV = predict(boostCV,test,n.trees=1000)
rmseBoostCV = sqrt(mean((predBoostCV-test$earn)^2)); rmseBoostCV

##10-fold Cross-Validation to tune tree complexity##
library(caret)
trControl = trainControl(method="cv",number=10) #10-fold cross validation
tuneGrid = expand.grid(.cp=seq(0,0.1,0.001))   

set.seed(100)
trainCV = train(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume,data=train,
                method="rpart", trControl=trControl,tuneGrid=tuneGrid)

head(trainCV$results) # first few cv results

plot(trainCV)

trainCV$bestTune

#Applying tree model with optimal complexity value to test sample
treeCV = rpart(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume,data=train,
               method="class", control=rpart.control(cp=trainCV$bestTune))
predCV = predict(treeCV,newdata=test,type="class")
ct = table(test$Direction,predCV); ct

(ct[1,1]+ct[2,2])/nrow(test) # Accuracy


####SUPPORT VECTOR MACHINES####

### ### ### ### ### ### ### ### ### ### ### ### ### ### 

##Linearly Separable##

#visualize data
ggplot(data,aes(x=x1,y=x2,color=y))+
  geom_point()+
  guides(color=F)

#find a linear classifier or a hyperplane
ggplot(data,aes(x=x1,y=x2,color=y))+
  geom_point()+
  guides(color=F)+
  geom_abline(slope = 1,intercept = 0,color='cadetblue', size=1)

# Maximum Margin Classifier
ggplot(data,aes(x=x1,y=x2,color=y))+
  geom_point()+
  guides(color=F)+
  geom_abline(slope = 1,intercept = 0,color='cadetblue', size=1)+
  geom_abline(slope = 1,intercept = -0.2,color='rosybrown', size=1)+
  geom_abline(slope = 1,intercept = 0.2,color='rosybrown', size=1)

### ### ### ### ### ### ### ### ### ### ### ### ### ### 

##Non-linearly Separable##

ggplot(data,aes(x=x1,y=x2,color=y))+
  geom_point()+
  guides(color=F)+
  geom_abline(slope = 1,intercept = 0,color='cadetblue', size=1)+
  geom_abline(slope = 1,intercept = -0.2,color='rosybrown', size=1)+
  geom_abline(slope = 1,intercept = 0.2,color='rosybrown', size=1)

##Support Vector Machines- Linear Models##
set.seed(0617)
split = sample(1:nrow(data),0.7*nrow(data))
train = data[split,]
test = data[-split,]

#Fit an SVM model using the default cost of 1
library(e1071)
svmLinear = svm(y~.,train,kernel='linear',scale=F,type='C-classification') 
# if outcome is a factor, default type='C-classification'
summary(svmLinear)

#plot the decision boundary
beta = t(svmLinear$coefs) %*% svmLinear$SV
slope = -beta[1]/beta[2]
intercept = svmLinear$rho/beta[2]

ggplot(train,aes(x=x1,y=x2,color=y))+
  geom_point()+
  guides(color=F)+
  geom_abline(slope = slope,intercept = intercept,color='cadetblue', size=1)

##plot with margins
ggplot(train,aes(x=x1,y=x2,color=y))+
  geom_point()+
  guides(color=F)+
  geom_abline(slope = slope,intercept = intercept,color='cadetblue', size=1)+
  geom_abline(slope = slope,intercept = intercept-1/beta[2],color='rosybrown', size=1)+  
  geom_abline(slope = slope,intercept = intercept+1/beta[2],color='rosybrown', size=1)

##Plot within svm library
plot(svmLinear, train)

#prediction on training set
pred = predict(svmLinear)
table(pred,train$y)

#prediction on testing set
pred = predict(svmLinear,newdata=test)
table(pred,test$y)


##SVM: Cost = 100##
library(e1071)
svmLinear = svm(y~.,train,kernel='linear',scale=F,type='C-classification',cost=100) # if outcome is a factor, default type='C-classification'
beta = t(svmLinear$coefs) %*% svmLinear$SV
slope = -beta[1]/beta[2]
intercept = svmLinear$rho/beta[2]

ggplot(train,aes(x=x1,y=x2,color=y))+
  geom_point()+
  guides(color=F)+
  geom_abline(slope = slope,intercept = intercept,color='cadetblue', size=1)

summary(svmLinear)


##SVM: Tune for best cost##
svmTune = tune(method = svm,y~.,data=train,kernel='linear', type='C-classification', scale=F, ranges = list(cost=c(0.01,0.1,1, 10, 100)))
svmTune$best.model

#predict best tune on testing data
pred = predict(svmTune$best.model,newdata=test)
table(pred,test$y)

#plot for best cost tune
plot(svmTune$best.model,test)

### ### ### ### ### ### ### ### ### ### ### ### ### ### 
##SVM- Polynomials##

library(e1071)
svmLinear = svm(y~.,data = train, kernel='linear',scale=F,type='C-classification')
pred = predict(svmLinear)
mean(pred==train$y)

pred = predict(svmLinear,newdata=test)
mean(pred==test$y)

plot(svmLinear,train)

##SVM with polynomial kernel##
svmPoly = svm(y~.,data = train, kernel='polynomial',scale=F,type='C-classification',degree=2)
pred = predict(svmPoly)
mean(pred==train$y)

pred = predict(svmPoly,newdata=test)
mean(pred==test$y)

##tuning the parameters of polynomial model
tune_svmPoly = tune(method = svm,y~.,data = train,kernel='polynomial',
                    ranges= list(degree=c(2,3), cost = c(0.01, 0.1, 1), gamma=c(0,1,10), coef0=c(0,0.1,1,10)))
summary(tune_svmPoly)

pred = predict(tune_svmPoly$best.model)
mean(pred==train$y)

pred = predict(tune_svmPoly$best.model,newdata=test)
mean(pred==test$y)

plot(tune_svmPoly$best.model,train)

### ### ### ### ### ### ### ### ### ### ### ### ### ### 
##SVM- Radial Basis Function##

svmRadial = svm(y~.,data = train, kernel='radial',scale=F,type='C-classification')
pred = predict(svmRadial)
mean(pred==train$y)

pred = predict(svmRadial,newdata=test)
mean(pred==test$y)

plot(svmRadial,train)

##model tuning##
tune_svmRadial = tune(method='svm',y~.,data=train,kernel='radial', type='C-classification',
                      ranges=list(cost=c(0.1,10,100), gamma=c(1,10), coef0 = c(0.1,1,10)))
summary(tune_svmRadial$best.model)

pred = predict(tune_svmRadial$best.model)
mean(pred==train$y)

pred = predict(tune_svmRadial$best.model,newdata=test)
mean(pred==test$y)



