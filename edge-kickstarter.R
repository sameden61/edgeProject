# Project Kickstarter #

# ----- Libraries --------
library(tidyverse)
library(skimr)
library(caret)
library(ROCR)
library(caTools)
library(xgboost)
library(dplyr)
library(rpart)
library(rpart.plot)
library(doParallel)
library(randomForest)

##############################################################################
##################                                    ########################
##################             Read me                ########################
##################                                    ########################
##############################################################################

# Please name models log.model.restrict, cart.model.cv, etc. to make the comparision easier
# Please label confusion matrix outputs as tpr.log.xxx, fpr.cart.cv, etc. to make comparison easier
# If you're working in a branch, you can use 'git merge master' to pull in any changes from the Master to your branch 


##############################################################################
##################                                    ########################
##################        Global data setup           ########################
##################                                    ########################
##############################################################################

# Read in data
data <- readRDS("kickstarterData.rds")

# Remove 'outcome' variables, so that dependent variables are in the campaigner's control
data <- dplyr::select(data,-NumBackers, -AvgBackerPledge, -PledgedUSD, -StaffPick)

# Check the variables
names(data)

# Remove factors that are unique to each row; remove year because it's all the the same
data <- dplyr::select(data,-YearLaunched, -CampaignID, -CampaignName, -Blurb)

# Fix required variable types
data$CharLengthName <- as.numeric(data$CharLengthName)
data$OutcomeBinary <- factor(data$OutcomeBinary)

# Remove variables, which may be added later
# data <- select(data,-HourLaunched, -CategorySub, -QuarterLaunched)
data <- dplyr::select(data,-QuarterLaunched)

# Check for multicollinearity 
names(data)
lapply(data, class)
# Numeric/Integer = CampaignLengthDays, CharLengthName, NumWordsName, CharLengthBlurb, NumWordsBlurb, GoalUSD 
data.corr = dplyr::select(data,-MonthLaunched, -DayOfWeekLaunched, -HourLaunched, -Country, -CategoryParent, -CategorySub, -OutcomeBinary)
df.corr <- round(cor(data.corr),2)
df.corr
# Multicollinear issue found >.70 between CharLengthBlurb + NumWordsBlurb, drop CharLengthBlurb 
data <- dplyr::select(data,-CharLengthBlurb)

# Split train/test 
set.seed(15071)
split = createDataPartition(data$OutcomeBinary,p=0.7,list=FALSE)
data.train = data[split,]
data.test = data[-split,]


# Define loss function
loss.matrix<-  cbind(c(0, 1), c(2, 0))
loss.matrix
threshold <- loss.matrix[1,2]/(loss.matrix[1,2]+loss.matrix[2,1])

# Create a user-defined function (returns a 2-element array)
Loss <- function(data, lev = NULL, model = NULL, ...) {
  c(AvgLoss = mean(data$weights * (data$obs != data$pred)), Accuracy = mean(data$obs == data$pred))
}


####################################################################
##################                                ##################
##################   Logistic Regression Simple   ##################
##################                                ##################
####################################################################

# Logistic Simple 
log.model.simple = glm(OutcomeBinary ~ ., data=data.train, family="binomial") 
options(max.print = 2000) # Just to see everything
summary(log.model.simple)

# Logistic Simple Eval
pred.prob = predict(log.model.simple, newdata=data.test, type="response")

# Full ROC 
rocr.pred <- prediction(pred.prob, data.test$OutcomeBinary)
rocr.pred.df <- data.frame(fpr=slot(performance(rocr.pred, "tpr", "fpr"),"x.values")[[1]],tpr=slot(performance(rocr.pred, "tpr", "fpr"),"y.values")[[1]])

# Plot ROCR results
base <- data.frame(x=c(0,1),y=c(0,1))
perfect <- data.frame(x=c(0,0,1),y=c(0,1,1))
gg.log <- ggplot() +
  geom_line(data=rocr.pred.df,aes(x=fpr,y=tpr,colour='LR'),lwd=1) +
  geom_area(data=rocr.pred.df, aes(x=fpr,y=tpr),fill = "grey60",alpha=0.4,position="identity") +
  geom_line(data=base, aes(x=x, y=y,colour='B'),lwd=1) +
  geom_line(data=perfect, aes(x=x, y=y,colour='P'),lwd=1) +
  xlab("False Positive Rate") +
  ylab("True Positive Rate") +
  theme_bw() +
  xlim(0, 1) +
  ylim(0, 1) +
  scale_color_manual(values = c('LR'='black','B'='blue','P'='red'), name = "", labels = c('LR'="Logistic regression", 'B'="Baseline", "P"="Perfect prediction")) +
  theme(axis.title=element_text(size=18), axis.text=element_text(size=18), legend.text=element_text(size=18))

gg.log

# Chosen to limit the FPR, using ROCR as general guide.  Check back later ***
#threshold = 0.65

# Confusion Matrix 
pred = (pred.prob > threshold)
confusion.matrix = table(data.test$OutcomeBinary, pred)
table(data.test$OutcomeBinary, pred)

# Accuracy  
accuracy.log.simple = sum(diag(confusion.matrix)) / sum(confusion.matrix)
accuracy.log.simple

# True Positive Rate 
truePos.log.simple = confusion.matrix[2,2]/sum(confusion.matrix[2,])
truePos.log.simple

# False Positive Rate 
falsePos.log.simple = confusion.matrix[1,2]/sum(confusion.matrix[1,])
falsePos.log.simple

# AUC
pred.auc <- prediction(pred.prob, data.test$OutcomeBinary)
AUC.log.simple <- data.frame(performance(pred.auc, "auc")@y.values[[1]])  %>% as.numeric()
# AUC.log.simple <- AUC.log.simple$`performance(pred.auc, "auc")@y.values[[1]]`
AUC.log.simple

# Plot AUC results
gg.log <- gg.log +
        annotate("text", label = sprintf("Simple AUC = %.3f", AUC.log.simple), x= 0.6, y=0.1, size = 6, colour = "grey20")
gg.log

importance.log.simple <- as.data.frame(round(varImp(log.model.simple),2))
importance.log.simple <- data.frame(names=rownames(importance.log.simple),imp = importance.log.simple$Overall)
importance.log.simple <- importance.log.simple[order(importance.log.simple$imp, decreasing = T),]
importance.log.simple

####################################################################
##################                                ##################
##################    Logistic Reg Restricted     ##################
##################                                ##################
####################################################################

# Remove insignificant variables
summary(log.simple)
data.train.restrict <- data.train
data.train.restrict <- dplyr::select(data.train.restrict,-CategoryParent, -CharLengthName, -NumWordsBlurb, -CharLengthBlurb)
data.train.restrict <- dplyr::select(data.train.restrict,-GoalUSD)
names(data.train.restrict)

log.model.restrict = glm(OutcomeBinary ~., data=data.train.restrict, family="binomial") 

summary(log.model.restrict)

# Logistic Restricted Eval
pred.prob.restrict = predict(log.model.restrict, newdata=data.test, type="response")

# Full ROC
rocr.pred.restrict <- prediction(pred.prob.restrict, data.test$OutcomeBinary)
rocr.pred.df.restrict <- data.frame(fpr=slot(performance(rocr.pred.restrict, "tpr", "fpr"),"x.values")[[1]],tpr=slot(performance(rocr.pred.restrict, "tpr", "fpr"),"y.values")[[1]])

# Use same threshold, so don't need to replot separately

# Confusion Matrix
pred.restrict = (pred.prob.restrict > threshold)
confusion.matrix = table(data.test$OutcomeBinary, pred.restrict)
table(data.test$OutcomeBinary, pred.restrict)

# Accuracy 
accuracy.log.restrict = sum(diag(confusion.matrix)) / sum(confusion.matrix)
accuracy.log.restrict

# True Positive Rate
truePos.log.restrict = confusion.matrix[2,2]/sum(confusion.matrix[2,])
truePos.log.restrict

# False Positive Rate
falsePos.log.restrict = confusion.matrix[1,2]/sum(confusion.matrix[1,])
falsePos.log.restrict

# AUC
pred.auc.restrict <- prediction(predict(log.model.restrict, newdata=data.test, type="response"), data.test$OutcomeBinary)
AUC.log.restrict <- as.data.frame(performance(pred.auc.restrict, "auc")@y.values[[1]])  %>% as.numeric()
AUC.log.restrict

# Compare model differences
gg.log <- gg.log +
        geom_line(data=rocr.pred.df.restrict,aes(x=fpr,y=tpr,colour='LR2'),lwd=1) +
        geom_area(data=rocr.pred.df.restrict, aes(x=fpr,y=tpr),fill = "lightsteelblue2",alpha=0.4,position="identity") +
        annotate("text", label = sprintf("Restrict AUC = %.3f", AUC.log.restrict), x= 0.6, y=0.2, size = 6, colour = "grey20")+
        scale_color_manual(values = c('LR' = 'grey50', 'LR2'='black','B'='blue','P'='red'), name = "", labels = c('LR'="Old glm",'LR2'="Restricted glm", 'B'="Baseline", "P"="Perfect prediction"))
gg.log

importance.log.restrict <- as.data.frame(round(varImp(log.model.restrict),2))
importance.log.restrict <- data.frame(names=rownames(importance.log.restrict),imp = importance.log.restrict$Overall)
importance.log.restrict <- importance.log.restrict[order(importance.log.restrict$imp, decreasing = T),]
importance.log.restrict

####################################################################
##################                                ##################
##################              CART              ##################
##################                                ##################
####################################################################

# Simple CART 
cart.model = rpart(OutcomeBinary~., data = data.train)
prp(cart.model)

# Cross-validate 
# First change OutcomeBinary back to factor #
data.train$OutcomeBinary = as.factor(data.train$OutcomeBinary)
data.test$OutcomeBinary = as.factor(data.test$OutcomeBinary)

levels(data.train$OutcomeBinary)


cv.trees = train(y = data.train$OutcomeBinary,
                 x = subset(data.train, select=-c(OutcomeBinary)),
                 method = "rpart",
                 weights = ifelse(data.train$OutcomeBinary == "TRUE", loss.matrix[2,1],loss.matrix[1,2] ), # loss function
                 trControl = trainControl(method = "cv", number = 10, summaryFunction = Loss), 
                 metric="AvgLoss", maximize=FALSE,                    
                 tuneGrid = data.frame(.cp = seq(0, 0.04, by=.001)))  

cv.trees
my.best.tree = cv.trees$finalModel
par(mar=c(1,1,1,1))
prp(my.best.tree,digits=2,varlen=-15,tweak=1.2)

# CART my.best.tree eval
# Confusion Matrix 
#pred.prob.cart = predict(my.best.tree, newdata=data.test, type = "class")
#data.test.mm <- as.data.frame(model.matrix(OutcomeBinary~.+0, data=data.test))  

pred.cart = predict(my.best.tree, newdata=data.test, type = "class")
#pred.cart = (pred.prob.cart > threshold)
pred.cart = as.data.frame(pred.cart)
#pred.cart = pred.cart[,2]
confusion.matrix = table(data.test$OutcomeBinary, pred.cart$pred.cart)

table(data.test$OutcomeBinary, pred.cart$pred.cart)

# Accuracy 
accuracy.cart = sum(diag(confusion.matrix)) / sum(confusion.matrix)
accuracy.cart

# True Positive Rate 
truePos.cart = confusion.matrix[2,2]/sum(confusion.matrix[2,])
truePos.cart

# False Positive Rate
falsePos.cart = confusion.matrix[1,2]/sum(confusion.matrix[1,])
falsePos.cart

# AUC
pred.auc.cart <- prediction(predict(my.best.tree, newdata=data.test)[,2], data.test$OutcomeBinary)
AUC.cart <- as.data.frame(performance(pred.auc.cart, "auc")@y.values[[1]]) %>% as.numeric()
AUC.cart

importance.cart <- as.data.frame(round(varImp(my.best.tree),2))
importance.cart <- data.frame(names=rownames(importance.cart),imp = importance.cart$Overall)
importance.cart <- importance.cart[order(importance.cart$imp, decreasing = T),]
importance.cart

####################################################################
##################                                ##################
##################         Random Forest          ##################
##################                                ##################
####################################################################

# RF cannot handle categorical variables with >53 categories :(
# Check str(data.train) to see which is causing issue --> CategorySub
# Try removing CategorySub


#data.train.rf = select(data.train,-CategorySub)

#data.train.rf <- as.data.frame(model.matrix( OutcomeBinary~ CategorySub +0, data = data.test))

data.train.rf <- dummyVars( ~ CategorySub , data = subset(data.train,select =-c(OutcomeBinary)))
data.train.rf <- data.train.rf %>% predict(newdata = data.train) %>% data.frame()
data.train.rf <- subset(data.train, select =-c(CategorySub)) %>% cbind(data.train.rf)

data.test.rf <- dummyVars( ~ CategorySub , data = subset(data.test,select =-c(OutcomeBinary)))
data.test.rf <- data.test.rf %>% predict(newdata = data.test) %>% data.frame()
data.test.rf <- subset(data.test, select =-c(CategorySub)) %>% cbind(data.test.rf)

names(data.train.rf)

# target.rf <- (data.train$OutcomeBinary %>% as.numeric() -1) %>% as.factor()
# head(data.train$OutcomeBinary)
# head(target.rf)

# Cross-validate

#readRDS("rfcv.rda")

evalLossRF <- function(data, lev = NULL, model = NULL, ...) {
  loss <- as.numeric(sum(ifelse(data$obs == "TRUE", 1,2) * (data$obs != data$pred)))
  names(loss) <- "loss"
  loss
}

head(data.train[1:300,]$OutcomeBinary)

cl <- makePSOCKcluster(6)
registerDoParallel(cl)

# Reduced just for speed 
rf.cv = train(y = data.train$OutcomeBinary,
              x = subset(data.train.rf, select = -c(OutcomeBinary)),
              method="rf",
              cutoff=c(1-threshold,threshold),
              nodesize = 25,
              ntrees = 101,
              maxdepth = 4,
              importance = TRUE,
              tuneGrid=data.frame(mtry=12:15),
              metric="loss", maximize=FALSE, 
              trControl= trainControl(method="cv", number = 5, summaryFunction = evalLossRF))

stopCluster(cl)
registerDoSEQ()

varImp(rf.cv$finalModel)

names(rf.cv)


saveRDS(rf.cv, "rfcv.rda")
my.best.rf = rf.cv$finalModel
saveRDS(my.best.rf, "mybestrf.rda")
plot(my.best.rf)

# RF my.best.rf eval
# Confusion Matrix 
pred.rf = predict(my.best.rf, newdata=subset(data.test.rf, select = -c(OutcomeBinary)))
#pred.rf = (pred.prob.rf > threshold)
#pred.rf = data.frame(pred.rf)
head(pred.rf)

length(pred.rf)

#pred.rf = pred.rf[,"TRUE"]
confusion.matrix = table(data.test$OutcomeBinary, pred.rf)

table(data.test$OutcomeBinary, pred.rf)


# Accuracy 
accuracy.rf = sum(diag(confusion.matrix)) / sum(confusion.matrix)
accuracy.rf

# True Positive Rate 
truePos.rf = confusion.matrix[2,2]/sum(confusion.matrix[2,])
truePos.rf

# False Positive Rate
falsePos.rf = confusion.matrix[1,2]/sum(confusion.matrix[1,])
falsePos.rf

# AUC
pred.auc.rf <- prediction(predict(my.best.rf, newdata=data.test.rf, type= "prob")[,2], data.test.rf$OutcomeBinary)
AUC.rf <- as.data.frame(performance(pred.auc.cart, "auc")@y.values[[1]]) %>% as.numeric
AUC.rf


importance.rf <- as.data.frame(round(importance(my.best.rf)[order(-importance(my.best.rf)),],2))
colnames(importance.rf) = 'imp'
importance.rf <- cbind(data.frame(var=rownames(importance.rf)),importance.rf)
importance.rf

####################################################################
##################                                ##################
##################             xgBoost            ##################
##################                                ##################
####################################################################

set.seed(15701)

# target <- data.train$OutcomeBinary
# target <- as.numeric(target) - 1
# 
# head(target)
# head(data.train$OutcomeBinary)
# 
# target.test <- data.test$OutcomeBinary
# target.test <- as.numeric(target.test) - 1
# 
# 
# data.train.xgb <- model.matrix(~ .+0, data = subset(data.train, select=-c(OutcomeBinary)))
# data.test.xgb <- model.matrix(~ .+0, data = subset(data.test, select=-c(OutcomeBinary)))
# 
# data.train.xgb <- xgb.DMatrix(data = as.matrix(data.train.xgb),label = target)
# data.test.xgb <- xgb.DMatrix(data = as.matrix(data.test.xgb),label = target.test)
# 
# attr(data.train.xgb, 'label') <- getinfo(data.train.xgb, 'label')
# 
# 
# evalLoss <- function(preds, data.train.xgb) {
#   labels <- attr(data.train.xgb, "label")
#   err<- as.numeric(sum(ifelse(labels, ifelse(preds>(2/3),0,1),ifelse(preds>(2/3),2,0) )))
#   #err<- as.numeric(sum(labels))
#   #print(sum(labels))
#   return(list(metric = "loss", value = err))
# }
# 
# params = params <- list(
#   objective = "binary:logistic"
#   ,eval_metric = evalLoss
#   ,eta =0.1
# )
# 
# cv.xgb <- xgb.cv(
#   params = params
#   ,data = as.matrix(data.train.rf)
#   ,label = target
#   ,maximize = FALSE
#   ,nrounds = 1000
#   ,nfold = 10
#   ,print_every_n = 10
#   ,early_stopping_rounds = 20
# )
# 
# best.xgb.iter <- cv.xgb$best_iteration
# 
# 
# outcomebinary.cv.xgb <- xgb.train(
#   params = params
#   ,data = data.train.xgb
#   ,nrounds = best.xgb.iter
#   ,watchlist=list(val = data.test.xgb, train = data.train.xgb)
#   ,maximize = FALSE
#   ,early_stopping_rounds = 20
# )
# 
# 
# outcomebinary.cv.xgb

evalLossRF <- function(data, lev = NULL, model = NULL, ...) {
  loss <- as.numeric(sum(ifelse(data$obs == "TRUE", 1,2) * (data$obs != data$pred)))
  names(loss) <- "loss"
  loss
}

xgb.cv = train(y = data.train$OutcomeBinary,
              x = data.train.rf %>% subset(select = -c(OutcomeBinary)) %>% data.matrix(),
              method="xgbTree",
              metric="loss", maximize=FALSE,
              importance = TRUE,
              trControl= trainControl(method="cv", number = 5, summaryFunction = evalLossRF))


xgb.cv
xgb.cv$bestTune

pred.prob.xgb = predict(xgb.cv$finalModel, newdata=data.test.rf %>% subset(select = -c(OutcomeBinary)) %>% data.matrix())


# xgBoost eval
#pred.prob.xgb = predict(outcomebinary.cv.xgb, newdata=data.test.xgb)

head(pred.prob.xgb)
head(target.test)

tail(pred.prob.xgb)
tail(target.test)


# Confusion Matrix
pred.xgb = (pred.prob.xgb < (1-threshold))
confusion.matrix = table(data.test$OutcomeBinary, pred.xgb)
table(data.test$OutcomeBinary, pred.xgb)

# Accuracy
accuracy.xgb = sum(diag(confusion.matrix)) / sum(confusion.matrix)
accuracy.xgb

# True Positive Rate
truePos.xgb = confusion.matrix[2,2]/sum(confusion.matrix[2,])
truePos.xgb

# False Positive Rate 
falsePos.xgb = confusion.matrix[1,2]/sum(confusion.matrix[1,])
falsePos.xgb

# AUC
pred.auc.xgb <- prediction(predict(outcomebinary.cv.xgb, newdata=data.test.xgb), data.test$OutcomeBinary)
AUC.xgb <- as.data.frame(performance(pred.auc.xgb, "auc")@y.values[[1]]) %>% as.numeric()
AUC.xgb

# xgb.cv.model <- xgb.cv$finalModel
importance.xgb <- xgb.importance(model = xgb.cv.model)
# 
# importance.xgb <- as.data.frame(round(importance(xgb.cv$finalModel)[order(-importance(xgb.cv$finalModel)),],2))
# colnames(importance.xgb) = 'imp'
# importance.xgb <- cbind(data.frame(var=rownames(importance.xgb)),importance.xgb)
# importance.xgb


####################################################################
##################                                ##################
##################        Comp Efficiency         ##################
##################                                ##################
####################################################################

# Log Simple #
time.log.simple <- system.time(log.model.simple <- glm(OutcomeBinary ~ ., data=data.train, family="binomial"))

# Log Restricted #
time.log.restrict <- system.time(log.model.restrict <- glm(OutcomeBinary ~., data=data.train.restrict, family="binomial"))

# CART #
time.cart <- system.time(cv.trees <- train(y = data.train$OutcomeBinary,
                             x = subset(data.train, select=-c(OutcomeBinary)),
                             method = "rpart",
                             trControl = trainControl(method = "cv", number = 10), 
                             metric="Accuracy", maximize=TRUE,                    
                             tuneGrid = data.frame(.cp = seq(0, 0.04, by=.001))))

# RF #
# time.rf <- system.time(rf.cv <- train(y = data.train.rf$OutcomeBinary,
#                           x = subset(data.train.rf,select=-c(OutcomeBinary)),
#                           method="rf", nodesize=25, ntree=80, cutoff=c(0.65,0.35),
#                           tuneGrid=data.frame(mtry=1:10),
#                           trControl= trainControl(method="cv",number=5)))

# XGB #
# time.xgb <- system.time(outcomebinary.cv.xgb <- train(y = data.train$OutcomeBinary,
#                                           x = data.matrix(subset(data.train, select=-c(OutcomeBinary))),
#                                           method = "xgbTree", 
#                                           metric="Accuracy",
#                                           trControl = trainControl(method="cv", number=10)))

# Compare results


#Models
model.list <- c("Log Simple", "Log Restrict", "CART", "Random Forest", "XGBoost")
results.accuracy <- c(accuracy.log.simple, accuracy.log.restrict, accuracy.cart, accuracy.rf, accuracy.xgb)
results.tpr <- c(truePos.log.simple,truePos.log.restrict,truePos.cart, truePos.rf, truePos.xgb)
results.fpr <- c(falsePos.log.simple,falsePos.log.restrict,falsePos.cart, falsePos.rf, falsePos.xgb)
results.auc <- c(AUC.log.simple,AUC.log.restrict,AUC.cart, AUC.rf, AUC.xgb)
#results.time <- c(time.log.simple[["elapsed"]],time.log.simple[["elapsed"]],time.cart[["elapsed"]],time.rf[["elapsed"]], time.xgb[["elapsed"]])

all.results <- model.list
all.results <- all.results %>% cbind(results.accuracy %>% round(3))  %>% 
  cbind(results.tpr%>% round(3)) %>%
  cbind(results.fpr%>% round(3)) %>%
  cbind(results.auc%>% round(3)) %>%
  #cbind(results.time%>% round(3)) %>%
  as.data.frame()

col.names <- c("Model","Accuracy","TPR","FPR","AUC")#, "Comp. Time")
colnames(all.results) <- col.names

all.results 

summary(log.model.simple)

#Choose XGBoost
print(outcomebinary.cv.xgb.feature_importances_)

# plot
pyplot.bar(range(len(outcomebinary.cv.xgb.feature_importances_)), outcomebinary.cv.xgb.feature_importances_)
pyplot.show()

