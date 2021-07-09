############################ Problem 1 ############################

# Load the Data
company_data<- read.csv(file.choose())
summary(company_data$Sales)

#categorizing the sales column (target) using cut function
company_data$Sales <- cut(company_data$Sales , breaks = c( -Inf ,8.135, Inf)
                          , labels = c("low" , "high"))


company_data$Sales <- as.factor(company_data$Sales)
str(company_data)

#splitting the data
library(caTools)
set.seed(2)
split <- sample.split(company_data$Sales, SplitRatio = 0.8)
company_train <- subset(company_data, split == TRUE)
company_test <- subset(company_data, split == FALSE)

# check the proportion of class variable
prop.table(table(company_data$Sales))
prop.table(table(company_train$Sales))
prop.table(table(company_test$Sales))

#Training a model on the data
install.packages("C50")
library(C50)

#model 
model1 <- C5.0(company_train[, -1], company_train$Sales)

#plotting the model
windows()
plot(model1)

# Display detailed information about the tree
summary(model1)

#Evaluating model performance
# Test data accuracy
test_res <- predict(model1, company_test)
test_acc <- mean(company_test$Sales == test_res)
test_acc

# cross tabulation of predicted versus actual classes
library(gmodels)
CrossTable(company_test$Sales, test_res, dnn = c('actual sales', 'predicted sales'))

# On Training Dataset
train_res <- predict(model1, company_train)
train_acc <- mean(company_train$Sales == train_res)
train_acc

table(company_train$Sales, train_res)

#train accuracy more than test accuracy hence overfit model

#building the model using train data using pruning technique to overcome overfit
library(rpart)
model2 <- rpart(Sales ~ ., data = company_train,method = 'class',control = rpart.control(cp = 0, maxdepth = 2))

# Plot Decision Tree
install.packages("rpart.plot")
library(rpart.plot)
rpart.plot(model2, box.palette = "auto", digits = 1)

# Test data accuracy
test_res <- predict(model2, company_test , type = 'class')
test_acc <- mean(company_test$Sales == test_res)
test_acc

# cross tabulation of predicted versus actual classes
library(gmodels)
CrossTable(company_test$Sales, test_res, dnn = c('actual sales', 'predicted sales'))

# On Training Dataset
train_res <- predict(model2, company_train , type = 'class')
train_acc <- mean(company_train$Sales == train_res)
train_acc

table(company_train$Sales, train_res)

#both train and test accuracy are alomost same and hence right fit model

#random forest method
install.packages("randomForest")
library(randomForest)

#building random forest model
model3 <- randomForest(Sales ~ ., data = company_train,maxnodes=4, mtree=6)
help(randomForest)
#plotting the model
windows()
plot(model3)

# Test data accuracy
test_res <- predict(model3, company_test , type = 'class')
test_acc<- mean(company_test$Sales == test_res)
test_acc

# cross tabulation of predicted versus actual classes
library(gmodels)
CrossTable(company_test$Sales, test_res, dnn = c('actual sales', 'predicted sales'))

# On Training Dataset
train_res <- predict(model3, company_train , type = 'class')
train_acc <- mean(company_train$Sales == train_res)
train_acc
table(company_train$Sales, train_res)


################################## problem 2 ############################


Diabetes_data<- read.csv(file.choose())
summary(Diabetes_data$ Classvariable)


Diabetes_data$ Classvariable <- as.factor(Diabetes_data$Classvariable)
str(Diabetes_data)

#splitting the data
library(caTools)
set.seed(2)
split <- sample.split(Diabetes_data$Classvariable, SplitRatio = 0.8)
Diabetes_train <- subset(Diabetes_data, split == TRUE)
Diabetes_test <- subset(Diabetes_data, split == FALSE)

# check the proportion of class variable
prop.table(table(Diabetes_data$Classvariable))
prop.table(table(Diabetes_train$Classvariable))
prop.table(table(Diabetes_test$Classvariable))

#Training a model on the data
install.packages("C50")
library(C50)

#model 
model1 <- C5.0(Diabetes_train[, -9], Diabetes_train$Classvariable)

#plotting the model
windows()
plot(model1)

# Display detailed information about the tree
summary(model1)

#Evaluating model performance
# Test data accuracy
test_res <- predict(model1, Diabetes_test)
test_acc <- mean(Diabetes_test$Classvariable == test_res)
test_acc

# cross tabulation of predicted versus actual classes
library(gmodels)
CrossTable(Diabetes_test$Classvariable, test_res, dnn = c('actual Classvariable', 'predicted Classvariable'))

# On Training Dataset
train_res <- predict(model1, Diabetes_train)
train_acc <- mean(Diabetes_train$Classvariable == train_res)
train_acc

table(Diabetes_train$Classvariable, train_res)



#building the model using train data using pruning technique to overcome overfit
library(rpart)
model2 <- rpart(Classvariable ~ ., data = Diabetes_train,method = 'class',control = rpart.control(cp = 0, maxdepth = 2))

# Plot Decision Tree
install.packages("rpart.plot")
library(rpart.plot)
rpart.plot(model2, box.palette = "auto", digits = 1)

# Test data accuracy
test_res <- predict(model2, Diabetes_test , type = 'class')
test_acc <- mean(Diabetes_test$Classvariable == test_res)
test_acc

# cross tabulation of predicted versus actual classes
library(gmodels)
CrossTable(Diabetes_test$Classvariable, test_res, dnn = c('actual Classvariable', 'predicted Classvariable'))

# On Training Dataset
train_res <- predict(model2, company_train , type = 'class')
train_acc <- mean(Diabetes_train$Classvariable == train_res)
train_acc

table(Diabetes_train$Classvariable, train_res)

#both train and test accuracy are alomost same and hence right fit model

#random forest method
install.packages("randomForest")
library(randomForest)

#building random forest model
model3 <- randomForest(Classvariable ~ ., data = Diabetes_train,maxnodes=4, mtree=6)
help(randomForest)
#plotting the model
windows()
plot(model3)

# Test data accuracy
test_res <- predict(model3, Diabetes_test , type = 'class')
test_acc<- mean(Diabetes_test$Classvariable == test_res)
test_acc

# cross tabulation of predicted versus actual classes
library(gmodels)
CrossTable(Diabetes_test$Classvariable, test_res, dnn = c('actual Classvariable', 'predicted Classvariable'))

# On Training Dataset
train_res <- predict(model3, Diabetes_train , type = 'class')
train_acc <- mean(Diabetes_train$Classvariable == train_res)
train_acc
table(Diabetes_train$Classvariable, train_res)


###################################### problem 3#################################

#load company data
fraud_check_data<- read.csv(file.choose())


#categorizing the Taxable.Income  column (target) using cut function
fraud_check_data$Taxable.Income <- cut(fraud_check_data$Taxable.Income , breaks = c( -Inf ,30000, Inf)
                                       , labels = c("Risky" , "Good"))

summary(fraud_check_data$Taxable.Income)

fraud_check_data$Taxable.Income <- as.factor(fraud_check_data$Taxable.Income )
str(fraud_check_data)

# check the proportion of class variable
prop.table(table(fraud_check_data$Taxable.Income))

# #since imbalanced data we upsample
# install.packages("caret")
# library(caret)
# fraud_check <- upSample(fraud_check_data,fraud_check_data$Taxable.Income)

#splitting the data
library(caTools)
set.seed(2)
fraud_check <- fraud_check_data
split <- sample.split(fraud_check$Taxable.Income, SplitRatio = 0.7)
fraud_check_train <- subset(fraud_check, split == TRUE)
fraud_check_test <- subset(fraud_check, split == FALSE)

# check the proportion of class variable
prop.table(table(fraud_check$Taxable.Income))
prop.table(table(fraud_check_train$Taxable.Income))
prop.table(table(fraud_check_test$Taxable.Income))

#Training a model on the data
install.packages("C50")
library(C50)

#model 
model1 <- C5.0(fraud_check_train[, -c(3)], fraud_check_train$Taxable.Income)

#plotting the model
windows()
plot(model1)

# Display detailed information about the tree
summary(model1)

#Evaluating model performance
# Test data accuracy
test_res <- predict(model1, fraud_check_test)
test_acc <- mean(fraud_check_test$Taxable.Income == test_res)
test_acc

# cross tabulation of predicted versus actual classes
library(gmodels)
CrossTable(fraud_check_test$Taxable.Income, test_res, dnn = c('actual default', 'predicted default'))

# On Training Dataset
train_res <- predict(model1, fraud_check_train)
train_acc <- mean(fraud_check_train$Taxable.Income == train_res)
train_acc

table(fraud_check_train$Taxable.Income, train_res)


#building the model using train data using pruning technique
library(rpart)
model2 <- rpart(Taxable.Income ~ ., data = fraud_check_train,method = 'class',control = rpart.control(cp = 0, maxdepth = 4))

# Plot Decision Tree
install.packages("rpart.plot")
library(rpart.plot)
rpart.plot(model2, box.palette = "auto", digits = -1)

# Test data accuracy
test_res <- predict(model2, fraud_check_test , type = 'class')
test_acc <- mean(fraud_check_test$Taxable.Income == test_res)
test_acc

# cross tabulation of predicted versus actual classes
library(gmodels)
CrossTable(fraud_check_test$Taxable.Income, test_res, dnn = c('actual default', 'predicted default'))

# On Training Dataset
train_res <- predict(model2, fraud_check_train , type = 'class')
train_acc <- mean(fraud_check_train$Taxable.Income == train_res)
train_acc

table(fraud_check_train$Taxable.Income, train_res)

#both train and test accuracy are alomost same and hence right fit model

#random forest method
install.packages("randomForest")
library(randomForest)

#building random forest model
model3 <- randomForest(Taxable.Income ~ ., data = fraud_check_train,maxnodes=4, mtree=6)
help(randomForest)
#plotting the model
windows()
plot(model3)

# Test data accuracy
test_res <- predict(model3, fraud_check_test , type = 'class')
test_acc<- mean(fraud_check_test$Taxable.Income == test_res)
test_acc

# cross tabulation of predicted versus actual classes
library(gmodels)
CrossTable(fraud_check_test$Taxable.Income, test_res, dnn = c('actual default', 'predicted default'))

# On Training Dataset
train_res <- predict(model3, fraud_check_train , type = 'class')
train_acc <- mean(fraud_check_train$Taxable.Income == train_res)
train_acc
table(fraud_check_train$Taxable.Income, train_res)

###################################### problem 4 #################################

#load company data
HR_data<- read.csv(file.choose())

library(caTools)
set.seed(0)
split <- sample.split(HR_data$ monthly.income.of.employee, SplitRatio = 0.8)
HR_train <- subset(HR_data, split == TRUE)
HR_test <- subset(HR_data, split == FALSE)

install.packages("rpart")
library(rpart)
model <- rpart( monthly.income.of.employee ~ ., data = HR_train,
                method = 'class', control = rpart.control(cp = 0, maxdepth = 3))

# Plot Decision Tree
library(rpart.plot)
rpart.plot(model, box.palette = "auto", digits = -3)

# Measure the RMSE on Test data
test_pred <- predict(model, newdata = HR_test , type = "vector")

# RMSE
accuracy1 <- sqrt(mean(HR_test$monthly.income.of.employee  - test_pred)^2)
accuracy1

# Measure the RMSE on Train data
train_pred <- predict(model, newdata = HR_train, type = "vector")

# RMSE
accuracy_train <- sqrt(mean(HR_train$monthly.income.of.employee  - train_pred)^2)
accuracy_train

#since both the RMSE of train and test are similar the model is right fit

#Random forest model
library(randomForest)
model <- randomForest(monthly.income.of.employee ~ ., data = HR_train , maxnodes=4, mtree=3)

# Measure the RMSE on Test data hence test accuracy
test_pred <- predict(model, newdata = HR_test)
acc_test <- sqrt(mean(HR_test$monthly.income.of.employee - test_pred)^2)
acc_test

# Measure the RMSE on Train data hence train accuracy
train_pred <- predict(model, newdata = HR_train)
acc_train <- sqrt(mean(HR_train$monthly.income.of.employee - train_pred)^2)
acc_train

#now checking that the candidate claim is genuine or fake
#candidate claims are stored in a dataframe
cols <- c("Position.of.the.employee", "no.of.Years.of.Experience.of.employee","monthly.income.of.employee")
Candidate_claim <- data.frame(a <-  "Region Manager" , b <-  5 , c <-  70000)
colnames(Candidate_claim) <- cols
#binding with test data for prediction
HR_test <- rbind(HR_test, Candidate_claim)

#predicting using model
HR_test$monthly.income.of.employee.pred <- predict(model, newdata = HR_test)

#our predicted salary is (61655.08)
HR_test$monthly.income.of.employee.pred[31]

#since the  predicted salary is almost similar to claimed salary it is said that candidate is genuine 





