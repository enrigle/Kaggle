# Data Dictionary
# 
# Variable	Definition	Key
# survival	Survival	0 = No, 1 = Yes
# pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
# sex	Sex	
# Age	Age in years	
# sibsp	# of siblings / spouses aboard the Titanic	
# parch	# of parents / children aboard the Titanic	
# ticket	Ticket number	
# fare	Passenger fare	
# cabin	Cabin number	
# embarked	Port of Embarkation
rm(list = ls())

library(mice)
library(nnet)
library(randomForest)
library(e1071)
library(dplyr)
library(rpart)
#Loading data
files <- setwd("C:/Users/USUARIO/Documents/Clases/Kaggle_competitions/Titanic/")
train <- read.csv(paste0(files, "/", "train.csv"), stringsAsFactors = FALSE, na.strings = "")
test <- read.csv(paste0(files, "/", "test.csv"), stringsAsFactors = FALSE, na.strings = "")

#merge all data
full <- bind_rows(train, test)

# regex
full$Title <- gsub("^.*, |\\. .*$", "", full$Name)
Mr <- c("Col", "Don", "Jonkheer", "Sir")
Mrs <- c("Dona", "Lady", "Mme", "the Countess")
Miss <- c("Mlle", "Ms")

# normalization of names
full$Title[full$Title %in% Mr] <- "Mr"
full$Title[full$Title %in% Mrs] <- "Mrs"
full$Title[full$Title %in% Miss] <- "Miss"

# create new variables
full$Fsize <- full$SibSp + full$Parch + 1
full$FsizeD[full$Fsize == 1] <- "Singleton"
full$FsizeD[full$Fsize >1 & full$Fsize < 5] <- "Small"
full$FsizeD[full$Fsize > 4] <- "Large"
full$Child <- ifelse(full$Age <= 12, 1, 0)
factor_var <- c("Survived", "Pclass", "Sex", "Embarked",
                "Title", "FsizeD", "Child")

# transformed to factor
full[factor_var] <- lapply(full[factor_var], factor)

res <- data.frame(PassengerId = test[, 1], Survived = 0)

# The mice helps you imputing missing values w/ plausible data values. These plausible values are drawn
# from a distribution specifically designed for e/ missing datapoint. methods(mice) for a list of methods
mice_mod <- mice(full[, c(3, 5 : 8, 10, 12)], method = "rf")
mice_output <- complete(mice_mod)

# head(complete(mice_mod))
# head(complete(mice_mod, 2))

# replace Age, Fare and Embarked 
full[, c(6, 10, 12)] <- mice_output[, c(3, 6, 7)]
full$Embarked <- factor(full$Embarked)
full$Child <- factor(ifelse(full$Age <= 12, 1, 0))

# removing cols Ticket and Cabin
full <- full[, -c(9, 11)]
train <- full[1 : nrow(train), ]
test <- full[(nrow(train) + 1) : nrow(full), -2]

#logistic
logistic_mod <- glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare +
                      Embarked + Title + FsizeD + Child, family = binomial, data = train)
res$Survived <- res$Survived + round(predict(logistic_mod, test, type = "response"), 0)
#nnet
nn_mod <- nnet(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare +
                 Embarked + Title + FsizeD + Child, data = train, size = 10, decay = 0.01)
res$Survived <- res$Survived + as.numeric(predict(nn_mod, test, type = "class"))
#randomForest
rf_mod <- randomForest(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare +
                         Embarked + Title + FsizeD + Child, data = train)
res$Survived <- res$Survived + as.numeric(predict(rf_mod, test))
#SVM
svm_mod <- svm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare +
                 Embarked + Title + FsizeD + Child, data = train)
res$Survived <- res$Survived + as.numeric(predict(svm_mod, test))
#rpart
rpart_mod <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare +
                     Embarked + Title + FsizeD + Child, data = train, method = "class")

rpart_predict <- predict(rpart_mod, newdata = test)
res$Survived <- res$Survived + ifelse(rpart_predict[, 1] < rpart_predict[, 2], 1, 0)
res$Survived <- ifelse(res$Survived > 3, 1, 0)
write.csv(res, file = "Titanic_voting_2.csv", row.names = FALSE)

# confMat <- table(train$Survived, test$Survived)
# accuracy <- sum(diag(confMat))/sum(confMat)




