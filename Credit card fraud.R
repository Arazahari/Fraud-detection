# First we should install o call libraries that we are going to use:
library(tidyverse)
library(tidymodels)
library(dplyr)
library(stringr)
library(tidyr)
library(ggplot2)
library(anytime)
library(smotefamily)
library(performanceEstimation)
library(SmartEDA)
library(splitstackshape)
library(rpart)
library(randomForest)
library(pROC)
library(rsample) 
library(caret)

# Now We need to import our data Train and Test

Train = read.csv('Desktop/fraudTrain.csv')
Test = read.csv('Desktop/fraudTest.csv')

# A glimpse of our data

head(Train)
head(Test)

# Now we need to know a little more about the structure of our datasets

str(Train)
str(Test)

# Preparing data for cleaning
# To do this we need to combine our dfs together, using cbind or rbind method

df = rbind(Train, Test)

percent_selected = 0.05
df = df %>% sample_frac(percent_selected)

num_rows_p <- nrow(percent_df)
num_rows_p
str(df)
summary(df)

# Now we should use Data cleaning methods to prepare our data sets

colSums(is.na(df))
sum(duplicated(df))

# We can split trans_date_trans_time into two different columns using separate()

df = df %>% separate(trans_date_trans_time,
                     c('trans_date','trans_time'))

# We need to drop some of our columns which doesn't provide useful information

df = df %>% select(-c(X, cc_num, lat, long, merch_lat,
                      merch_long,trans_num, unix_time, merchant,
                      zip, first,trans_time,street,trans_num, last))


str(df)

df$is_fraud = as.factor(df$is_fraud)
levels(df$is_fraud) = c("Not_Fraud", "Fraud")


table(df$is_fraud)
str(df)
# Only trans_date and trans_time are know "chr" which should be change to Data format

df$trans_date = anytime(df$trans_date)
df$trans_date = as.Date(df$trans_date)

df$dob = anytime(df$dob)
df$dob = as.Date(df$dob)
str(df)

# We need to change character variables into factor

df$category = as.factor(df$category)
df$gender = as.factor(df$gender)
df$city = as.factor(df$city)
df$state = as.factor(df$state)
df$job = as.factor(df$job)

str(df)

#Now we can visualize our Data
#Used ggplot to show how fraud happened in different categories by gender

ggplot(df, aes(x = category, fill = as.factor(gender))) +
  geom_bar(width = 0.5) +
  scale_fill_manual(values = c("1" = "violetred" , "0" = "darkorange")) +
  ggtitle("Plotting Gender across categories") +
  theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5))


# And in this plot we can see the how often fraud happened by the city

ggplot(df, aes(x= city_pop, fill= as.factor(is_fraud))) + 
  geom_bar(color= "grey20", width = 0.5) +
  scale_fill_manual(values = c("1" = "violetred" , "0" = "darkorange")) +
  ggtitle("Fraud vs City Population") +
  theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5))

set.seed(125)
df_new <- rownames_to_column(df, var = "id") %>% mutate_at(vars(id), as.integer)
train_2 <- df_new %>% stratified(., group = "is_fraud", size = 0.70)
test_2 <- df_new[-train_2$id, ]
prop.table(table(test_2$is_fraud))






#Now we should split our data set 

df_split = initial_split(df, prop = 0.7, strata = is_fraud) 
train_2 = training(df_split)
test_2 = testing(df_split)


#Now we need to start model evaluation
#First we are going with LOGISTIC REGRESSION

fullmod = glm(as.factor(is_fraud) ~ amt + as.factor(gender) + city_pop +
                as.factor(category) +
                as.factor(state), 
              data = train_2, family = "binomial")

pred_glm = predict(fullmod,newdata= test_2,type="response")
roc_glm = roc(test_2$is_fraud, pred_glm)
plot(roc_glm, col="orange")
auc_glm = auc(test_2$is_fraud, pred_glm)
print(auc_glm)
pred_glm_new = ifelse(pred_glm > 0.7,"Fraud","Not_Fraud")
confusionMatrix(as.factor(test_2$is_fraud), as.factor(pred_glm_new))
# Now we are going to try DECESION TREE

train_control <- trainControl(method = "cv",
                              number = 5,
                              classProbs = TRUE,
                              summaryFunction = twoClassSummary)

model <- train(as.factor(is_fraud) ~ amt + as.factor(gender) + city_pop +
                 as.factor(category) +
                 as.factor(state) + as.factor(job), 
               data = train_2, 
               method = "rpart", 
               trControl = train_control,
               metric= "ROC")

pred_y = predict(model ,newdata= test_2,type="prob")
roc_y = roc(test_2$is_fraud, pred_y[,2])
plot(roc_y, col="blue")

auc_y = auc(test_2$is_fraud, pred_y[,2])
print(auc_y)

pred_y_new = ifelse(pred_y[,2] > 0.7,"Fraud","Not_Fraud")
x =confusionMatrix(as.factor(test_2$is_fraud), as.factor(pred_y_new))
x
#RANDOM FOREST
model_rf = randomForest(as.factor(is_fraud) ~ amt + city_pop, 
                        data = train_2,ntree=500, mtry=8)

pred_rf = predict(model_rf ,newdata= test_2,type="prob")

roc_rf = roc(test_2$is_fraud, pred_rf[,2])
plot(roc_rf, col="green")

auc_rf = auc(test_2$is_fraud, pred_rf[,2])
print(auc_rf)

pred_rf_new = ifelse(pred_rf[,2] > 0.7,"Fraud","Not_Fraud")
z =confusionMatrix(as.factor(test_2$is_fraud), as.factor(pred_rf_new))
z
