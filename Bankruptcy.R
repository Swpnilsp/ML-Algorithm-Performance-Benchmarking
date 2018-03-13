### Project- Comparative study of prediction accuracy of machine learning aglorithms
## In the project, following alogrithms are compared for accuracy- 
# Logistic Regression, decision tree, 
#generalized additive models, Random Forest, Neural Network, Discriminant Analysis and XGBoost

library(rpart)
library(caret)
library(tidyverse)
library(data.table)
library(GGally)
library(corrplot)
library(verification)
library(ROCR)
library(maptree)
library(glmnet)
library(gridExtra)
library(randomForest)
library(mgcv)
library(nnet)
library(pROC)
library(gbm)
library(e1071)
library(xgboost)
setwd("~/Study/MS-Bana/DM 2/Case Study 1")
bankruptcy<-fread('bankruptcy.csv')

## Getting the sense of the data
colSums(is.na(bankruptcy))
str(bankruptcy)
head(bankruptcy)

### There are no missing values in the data, all columns are integer or numeric except for CUSIP
### We change DLRSN to factor 
bankruptcy$DLRSN<-as.factor(bankruptcy$DLRSN)
summary(bankruptcy)

## Checking CUSIP
sum(duplicated(bankruptcy$CUSIP))
## CUSIP Uniquely identifies each row, we drop the cusip column and use it as rownames
rownames(bankruptcy)<-bankruptcy$CUSIP
bankruptcy$CUSIP<-NULL
head(bankruptcy)

### Dividing test and train dataset
set.seed(222)
indexes<-sample(nrow(bankruptcy),0.8*nrow(bankruptcy),replace = F)
train<-bankruptcy[indexes,]
test<-bankruptcy[-indexes,]
dim(train)
dim(test)



### Exploratory data analysis-
summary(train)
cormat<-cor(train[,-'DLRSN'])
corrplot(cormat,method='number')

## Strong correlation between R6 and R1, R8 and R3--
## We need to take this into consideration while building models
cor.test(train$R1,train$R6)
cor.test(train$R3,train$R8)
p1<-ggplot(data = train,aes(R3,R8,col=DLRSN))+geom_point(alpha=0.5)
p2<-ggplot(data = train,aes(R1,R6,col=DLRSN))+geom_point(alpha=0.5)
grid.arrange(p1,p2)

## Now we plot each feature against DLRSN
p1<-ggplot(data = train,aes(x = DLRSN,y = R1,fill=DLRSN))+geom_boxplot()
p2<-ggplot(data = train,aes(x = DLRSN,y = R2,fill=DLRSN))+geom_boxplot()
p3<-ggplot(data = train,aes(x = DLRSN,y = R3,fill=DLRSN))+geom_boxplot()
p4<-ggplot(data = train,aes(x = DLRSN,y = R4,fill=DLRSN))+geom_boxplot()
p5<-ggplot(data = train,aes(x = DLRSN,y = R5,fill=DLRSN))+geom_boxplot()
p6<-ggplot(data = train,aes(x = DLRSN,y = R6,fill=DLRSN))+geom_boxplot()
p7<-ggplot(data = train,aes(x = DLRSN,y = R7,fill=DLRSN))+geom_boxplot()
p8<-ggplot(data = train,aes(x = DLRSN,y = R8,fill=DLRSN))+geom_boxplot()
p9<-ggplot(data = train,aes(x = DLRSN,y = R9,fill=DLRSN))+geom_boxplot()
p10<-ggplot(data = train,aes(x = DLRSN,y = R10,fill=DLRSN))+geom_boxplot()
p11<-ggplot(data = train,aes(x = DLRSN,y = FYEAR,fill=DLRSN))+geom_boxplot()

grid.arrange(p1,p2,p3,p4,p5,p6,nrow=3)
grid.arrange(p7,p8,p9,p10,p11,nrow=3)
rm(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11)

## It can be seen that all the values of DLRSN in the year 1999 were 0.
table(train$FYEAR,train$DLRSN) 

## this is an intersting find, let's not use YEAR for the prediction

## For the prediction, we define the cost function as we assign the cost of 15 when a bad observation is predicted as good
## and weight 1, when a good obesrvation is predicted as bad
## Cost function
cost1 <- function(actual, predicted) {
  weight1 = 15
  weight0 = 1
  c1 = (actual == 1) & (predicted < cutoff)  #logical vector - true if actual bad but predicted good
  c0 = (actual == 0) & (predicted > cutoff)  #logical vecotr - true if actual good but predicted bad
  return(mean(weight1 * c1 + weight0 * c0))
}

## Area under the ROC Curve used as cost function. We will need this later

## Cost function
cost2 <- function(actual, predicted) {
return(auc(roc(actual,predicted))[1])
}

## Prob thresholds to be used for ROC Curve
thresh<-seq(0,1,0.001)
### Logistic Regression-
full.log.probit<-glm(data = train,DLRSN~.,family = binomial(link=probit))

## the error we get because of the fact that all values of DLRSN are 0 for the year 1999. We need to remove YEAR
full.log.probit<-glm(data = train[,-'FYEAR'],DLRSN~.,family = binomial(link=probit))
summary(full.log.probit)
full.log.probit.prediction<-predict(full.log.probit,type = "response")
roc.plot(x = train$DLRSN == "1", pred = full.log.probit.prediction,thresholds = thresh)$roc.vol
## We can see that r4 and r5 are not significant

## Lasso Variable selection- We need to select variables that are most important
## We need to standardize all the variables before we can go for LASSO variable selection
X<-scale(train[,-c('DLRSN','FYEAR')])
X<-as.matrix(X)
Y<- as.matrix(train[,'DLRSN'])
lasso.fit<- glmnet(x=X, y=Y, family = "binomial", alpha = 1)
plot(lasso.fit, xvar = "lambda")

## We need to decide optimum value of lambda using Cross Validation, we go for 10 fold cross validation
cv.lasso<- cv.glmnet(x=X, y=Y,family = "binomial", alpha = 1, nfolds = 10)
plot(cv.lasso)

## In the above graph, the left vertical line is the value of lambda that gives gives smallest cross-validation error
## And the right is the value of lambda with CV error within 1 standard deviation of smallest CV error
cv.lasso$lambda.min
cv.lasso$lambda.1se

## we decide to go for lambda=0.007
# Checking the coefficients
coef(lasso.fit, s=cv.lasso$lambda.1se)
coef(lasso.fit, s=cv.lasso$lambda.mi)

## Predictions using, s=cv.lasso$lambda.1se
pred.lasso<- predict(lasso.fit, newx = X, s=cv.lasso$lambda.1se,type = 'response')

roc.plot(x = train$DLRSN == "1", pred = pred.lasso,thresholds = thresh)$roc.vol


#### Classification Tree- Now we try to fill the classification tree for the data.
full.rpart<-rpart(data = train[,-'FYEAR'],DLRSN~.,method = 'class')
plot(full.rpart)
text(full.rpart)
plotcp(full.rpart)
printcp(full.rpart)
rpart.prediction<-predict(full.rpart,type = 'prob')
roc.plot(x = train$DLRSN == "1", pred = rpart.prediction[,2],thresholds = thresh)$roc.vol

### Random Forest
full.randomForest<-randomForest(data=train[,-'FYEAR'],DLRSN~.,ntree=1000)
plot(full.randomForest)
rf.predicted<-predict(full.randomForest,type = 'prob')
roc.plot(x = train$DLRSN == "1", pred = rf.predicted[,2],thresholds = thresh)$roc.vol
varImpPlot(full.randomForest)

### GAM- Generalized additive models
# For generalized additive models
full.gam<-gam(data=train,DLRSN~s(R1)+s(R2)+s(R3)+s(R4)+s(R5)+s(R6)+s(R7)+s(R8)+
                s(R9)+s(R10),family = 'binomial')
summary(full.gam)
full.gam.prediction<-predict(full.gam,type = 'response')

# we can remove smoothing for non significant features
gam.reduced<-gam(data=train,DLRSN~R1+s(R2)+s(R3)+s(R6)+R7+s(R8)+
                s(R9)+s(R10),family = 'binomial')
summary(gam.reduced)
gam.reduced.prediction<-predict(gam.reduced,type = 'response')

## Comparing AUC for both
roc.plot(x = train$DLRSN=="1",pred = cbind(full.gam.prediction,gam.reduced.prediction),thresholds = thresh,
         legend = TRUE,leg.text = c("Full GAM","Reduced GAM"))$roc.vol

## We can see, they have almost same ROC

## Neural Networks- 
# for Neural Nets, we need to standardise all numeric variables 
train.std<-train
summary(train.std)
for (i in 3:12)
{
  train.std[,i]<-scale(train.std[,..i])
}
summary(train.std)

## for neural nets, we have tunning paramets such as number of hidden layers and weight decay-
# For now, we just consider the training data for cross validation, 5 fold cross validation
avgTrainROC<-NULL
avgTestROC<-NULL
for ( j in 1:10 )
{
  trainRoc<-NULL
  testRoc<-NULL
  for ( i in 1:5)
  {
    set.seed(22*i)
    flags<-sample(nrow(train.std),0.8*nrow(train.std),replace = F)
    nnet.train<-train.std[flags,]
    nnet.test<-train.std[-flags,]
    model<-nnet(data=nnet.train[,-c('FYEAR')],DLRSN~.,size=j,lineout=F,decay=0,maxit=10000)
    train.pred<-predict(model)
    test.pred<-predict(model,nnet.test)
    trainRoc[i]<-cost2(nnet.train$DLRSN,as.numeric(train.pred))
    testRoc[i]<-cost2(nnet.test$DLRSN,as.numeric(test.pred))
  }
  avgTrainROC[j]<-mean(trainRoc)
  avgTestROC[j]<-mean(testRoc)
}

ggplot(data = NULL,aes(x = 1:10,y = avgTrainROC,col='Train'))+geom_line()+
  geom_line(aes(y=avgTestROC,col='Test'))+labs(x="Hidden Layers",y='Average AUC')+
  scale_x_continuous(limits = c(1,10),breaks =seq(1,10,1) )

h<-which(avgTestROC==max(avgTestROC))
h
## As we can see, the AUC for train keeps increasing as we add more hidden layers but the test AUC starts
# decreasing after 8th layer. So we finalize 8 hidden layers.

## Now, we need to decide, weight decay,

avgTrainROC<-NULL
avgTestROC<-NULL
d<-NULL
for ( j in 1:30 )
{
  trainRoc<-NULL
  testRoc<-NULL
  wt<-j/1000
  d[j]<-wt
  for ( i in 1:5)
  {
    set.seed(22*i)
    flags<-sample(nrow(train.std),0.8*nrow(train.std),replace = F)
    nnet.train<-train.std[flags,]
    nnet.test<-train.std[-flags,]
    model<-nnet(data=nnet.train[,-c('FYEAR')],DLRSN~.,size=h,lineout=F,decay=wt,maxit=10000)
    train.pred<-predict(model)
    test.pred<-predict(model,nnet.test)
    trainRoc[i]<-cost2(nnet.train$DLRSN,as.numeric(train.pred))
    testRoc[i]<-cost2(nnet.test$DLRSN,as.numeric(test.pred))
  }
  avgTrainROC[j]<-mean(trainRoc)
  avgTestROC[j]<-mean(testRoc)
}

ggplot(data = NULL,aes(x = d,y = avgTrainROC,col='Train'))+geom_line()+
  geom_line(aes(y=avgTestROC,col='Test'))+labs(x="Weight Decay",y='Average AUC')+
  scale_x_continuous(limits = c(0,0.03),breaks =seq(0,0.03,0.005) )

## Finding out value of weight decay for which test error was minimum
d[which(avgTestROC==max(avgTestROC))]
dcay<-d[which(avgTestROC==max(avgTestROC))]

## We decide to build model with wight decay=0.027 and number of hiddne layers=3
nnet.model<-nnet(data=train[,-'FYEAR'],DLRSN~.,size=3,decay=dcay,lineout=F,maxit=10000)
nnet.prediction<-predict(nnet.model)
roc.plot(x=train$DLRSN=="1",pred=nnet.prediction,thresholds = thresh)$roc.vol


### Linear Discriminant Analysis 
model.lda<-lda(data=train[,-'FYEAR'],DLRSN~.)
lda.predicted<-predict(model.lda)$posterior[,2]
roc.plot(x=train$DLRSN=="1",pred=lda.predicted,thresholds = thresh)$roc.vol





##### Boosting algorithms-
# GBM- Gradient Boosting Machine 
# we need to tune depth of trees for better prediction and avoiding overfitting. We go for a 5 fold cross validation
avgAUC<-NULL
for (i in 2:9)
{
  model<-gbm(data=train[,-'FYEAR'],as.character(DLRSN)~.,distribution = "bernoulli",n.trees = 5000,
                 interaction.depth = i,cv.folds = 5)
  model.prediction<-predict(model,newdata = train[,-'FYEAR'],n.trees = 5000,type = 'response')
  avgAUC[i]<-cost2(train$DLRSN,model.prediction)
}

ggplot(data = NULL,aes(x = 2:9,y = avgAUC[2:9]))+geom_line()

gbm.model<-gbm(data=train[,-'FYEAR'],as.character(DLRSN)~.,distribution = "bernoulli",n.trees = 5000,
           interaction.depth = 8)
gbm.model.prediction<-predict(gbm.model,newdata = train[,-'FYEAR'],n.trees = 5000,type = 'response')

roc.plot(x=train$DLRSN=="1",pred=gbm.model.prediction,thresholds = thresh)$roc.vol

## Checking performance of test data
gbm.test.prediction<-predict(gbm.model,newdata = test[,-'FYEAR'],n.trees = 5000,type = 'response')
roc.plot(x=test$DLRSN=="1",pred=gbm.test.prediction,thresholds = thresh)$roc.vol


### XGBoost- eXtreme Gradient Boosting 

## We need to create matrix for test and train. On-hot encoding is not required as there are no important factors

##We will not touch test data yet. for tunning, we divide training dataset into two parts, xtrain and xtest. 
set.seed(100)
flag<-sample(nrow(train),0.8*nrow(train),replace = F)
xtrain<-train[flag,]
xtest<-train[-flag,]
train_mat<-sparse.model.matrix(data = xtrain[,-'FYEAR'],DLRSN~.-1)
head(train_mat)
test_mat<-sparse.model.matrix(data = xtest[,-'FYEAR'],DLRSN~.-1)
head(test_mat)
train_label<-as.numeric(xtrain$DLRSN)-1
test_label<-as.numeric(xtest$DLRSN)-1

# We need to conver data to DMatrix form
train_dMatrix<-xgb.DMatrix(data = as.matrix(train_mat),label=train_label)
test_dMatrix<-xgb.DMatrix(data = as.matrix(test_mat),label=test_label)


## Modeling
params <- list("objective" = "reg:logistic",
                   "eval_metric" = "auc")
watchlist <- list(train = train_dMatrix, test = test_dMatrix)

# eXtreme Gradient Boosting Model
xgb_model <- xgb.train(params = params,
                       data = train_dMatrix,
                       nrounds = 2000,
                       watchlist = watchlist,
                       eta = 0.02,
                       max.depth = 4,
                       gamma = 0,
                       subsample = 1,
                       colsample_bytree = 1,
                       missing = NA,
                       seed = 222)

tunning<-as.data.frame(xgb_model$evaluation_log)
ggplot(data = NULL,aes(x = tunning$iter,y = tunning$train_auc,col='train'))+geom_line()+
  geom_line(aes(y = tunning$test_auc,col='test'))

## As we can see, test AUC decreases after some time
# Optimum number of rounds

rounds<-which(tunning$test_auc==max(tunning$test_auc))

xgb_model <- xgb.train(params = params,
                       data = train_dMatrix,
                       nrounds = rounds[1],
                       watchlist = watchlist,
                       eta = 0.02,
                       max.depth = 4,
                       gamma = 0,
                       subsample = 1,
                       colsample_bytree = 1,
                       missing = NA,
                       seed = 222)

### Training prediction-
train_matrix<-sparse.model.matrix(data = train[,-'FYEAR'],DLRSN~.-1)
train_label<-as.numeric(train$DLRSN)-1
train_matrix<-xgb.DMatrix(data = as.matrix(train_matrix),label=train_label)

xgb_prediction.train<-predict(xgb_model, newdata = train_matrix)
## Prediction on test data-
# creating test Matrix
test_matrix<-sparse.model.matrix(data = test[,-'FYEAR'],DLRSN~.-1)
test_label<-as.numeric(test$DLRSN)-1
test_matrix<-xgb.DMatrix(data = as.matrix(test_matrix),label=test_label)

xgb_prediction<-predict(xgb_model, newdata = test_matrix)

## AUC-
roc.plot(x = test$DLRSN=="1",pred = xgb_prediction,thresholds = thresh)$roc.vol

# Feature importance
imp <- xgb.importance(colnames(train_dMatrix), model = xgb_model)
print(imp)
xgb.plot.importance(imp)



######

### Comparing Performance of all models on training data
roc.plot(x=train$DLRSN=="1",pred=cbind(full.log.probit.prediction,pred.lasso,
                                       rpart.prediction[,2],rf.predicted[,2],full.gam.prediction,nnet.prediction,
                                       lda.predicted,gbm.model.prediction,xgb_prediction.train),legend = T,
                                        leg.text = c("Logistic","Lasso","DecisionTree",
                                         "RandomForest","G. Additive Model","NeuralNets","LDA",
                                         "GBM","XGB"),thresholds = thresh)$roc.vol

## Comparison of model AUC for test data

X<-scale(test[,-c('DLRSN','FYEAR')])
X<-as.matrix(X)

logit.test.pred<-predict(full.log.probit,test,type = 'response')
lasso.test.pred<-predict(lasso.fit, newx = X, s=cv.lasso$lambda.1se,type = 'response')
rpart.test.pred<-predict(full.rpart,test,type = 'prob')
rf.test.pred<-predict(full.randomForest,test,type = 'prob')[,2]
gam.test.pred<-predict(full.gam,test,type = 'response')
nnet.test.pred<-predict(nnet.model,test)
lda.test.pred<-predict(model.lda,test)$posterior[,2]

roc.plot(x=test$DLRSN=="1",pred=cbind(logit.test.pred,lasso.test.pred,
                                      rpart.test.pred[,2],rf.test.pred,gam.test.pred,nnet.test.pred,
                                      lda.test.pred,gbm.test.prediction,xgb_prediction),legend = T,
         leg.text = c("Logistic","Lasso","DecisionTree",
                      "RandomForest","G. Additive Model","NeuralNets","LDA","GBM","XGB"),thresholds = thresh)$roc.vol

models<-c('Logistic Reg',"Lasso Reg","DecisionTree","RandomForest","Additive Model","Neural Net","LDA","GBM","XGB")
TrainAuc<-c(cost2(train$DLRSN,as.numeric(full.log.probit.prediction)),cost2(train$DLRSN,as.numeric(pred.lasso)),
            cost2(train$DLRSN,as.numeric(rpart.prediction[,2])),cost2(train$DLRSN,as.numeric(rf.predicted[,2])),
            cost2(train$DLRSN,as.numeric(full.gam.prediction)),cost2(train$DLRSN,as.numeric(nnet.prediction)),
            cost2(train$DLRSN,as.numeric(lda.predicted)),cost2(train$DLRSN,as.numeric(gbm.model.prediction)),
            cost2(train$DLRSN,as.numeric(xgb_prediction.train)))

TestAuc<-c(cost2(test$DLRSN,as.numeric(logit.test.pred)),cost2(test$DLRSN,as.numeric(lasso.test.pred)),
           cost2(test$DLRSN,as.numeric(rpart.test.pred[,2])),cost2(test$DLRSN,as.numeric(rf.test.pred)),
           cost2(test$DLRSN,as.numeric(gam.test.pred)),cost2(test$DLRSN,as.numeric(nnet.test.pred)),
           cost2(test$DLRSN,as.numeric(lda.test.pred)),cost2(test$DLRSN,as.numeric(gbm.test.prediction)),
           cost2(test$DLRSN,as.numeric(xgb_prediction)))

results<-as.data.frame(cbind(models,TrainAuc,TestAuc))
results<-results%>%arrange(desc(TestAuc))
results

### As we can see, GBM has the highest AUC for test data.

### Defining optimum cutoff probability for minimizing the cost function-
probs<-seq(0,1,0.001)
cost<-NULL
for (i in 1:1000)
{
  cutoff<-probs[i]
  predicted<-ifelse(gbm.model.prediction>cutoff,1,0)
  cost[i]<-cost1(train$DLRSN,predicted)
}
plot(1:1000,cost)

cutoffProb<-probs[which(cost==min(cost))]
cutoffProb
predicted<-ifelse(rf.test.pred>cutoffProb,1,0)
cm<-confusionMatrix(predicted,test$DLRSN)
cm[2]
cm[3]$overall[1]
cost1(test$DLRSN,predicted)

## Note that the accuracy may not be high for the test set, however, the cost is lowest. 
## If we were to go for accuracy, we can define new cost function as-

cost3 <- function(actual, predicted) {
m<-mean(actual==predicted)
return(m)
}


### Defining optimum cutoff probability for minimizing the cost function-
probs<-seq(0,1,0.001)
cost<-NULL
for (i in 1:1000)
{
  cutoff<-probs[i]
  predicted<-ifelse(gbm.model.prediction>cutoff,1,0)
  cost[i]<-cost3(train$DLRSN,predicted)
}
plot(1:1000,cost)



cutoffProb<-probs[which(cost==max(cost))]
cutoffProb
predicted<-ifelse(rf.test.pred>cutoffProb,1,0)
cm<-confusionMatrix(predicted,test$DLRSN)
cm[2]
cm[3]$overall[1]
cost1(test$DLRSN,predicted)

## We have achieved almost 89% accuracy! 