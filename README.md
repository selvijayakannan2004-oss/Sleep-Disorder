install.packages(c("caret", "e1071", "pROC", "ggplot2", "reshape2", "nnet"), dependencies = TRUE)
# Sleep Disorder Classification using Logistic Regression and SVM
# ---------------------------------------------------------------

library(caret)
library(e1071)
library(pROC)
library(ggplot2)
library(reshape2)
library(nnet)

set.seed(123)

# ---- Example Dataset (Replace this with your CSV) ----
n <- 200
sleep_data <- data.frame(
  Age = round(runif(n, 18, 65)),
  BMI = round(runif(n, 18, 35), 1),
  HeartRate = round(runif(n, 60, 100)),
  StressLevel = round(runif(n, 1, 10)),
  Disorder = factor(sample(c("None", "Insomnia", "Sleep Apnea"), n, replace = TRUE))
)1

# ---- If you have a CSV file, uncomment this line ----
# sleep_data <- read.csv("sleep_data.csv")

# ---- Split the data ----
set.seed(123)
trainIndex <- createDataPartition(sleep_data$Disorder, p = 0.8, list = FALSE)
train <- sleep_data[trainIndex, ]
test <- sleep_data[-trainIndex, ]

# ---- Logistic Regression ----
log_model <- multinom(Disorder ~ Age + BMI + HeartRate + StressLevel, data = train)
log_pred <- predict(log_model, newdata = test)

log_conf <- confusionMatrix(log_pred, test$Disorder)
cat("🔹 Logistic Regression Results:\n")
print(log_conf)

# ---- Support Vector Machine (SVM) ----
svm_model <- svm(Disorder ~ Age + BMI + HeartRate + StressLevel, data = train, kernel = "linear")
svm_pred <- predict(svm_model, newdata = test)

svm_conf <- confusionMatrix(svm_pred, test$Disorder)
cat("\n🔹 SVM Results:\n")
print(svm_conf)

# ---- Compare Accuracy ----
results <- data.frame(
  Model = c("Logistic Regression", "SVM"),
  Accuracy = c(log_conf$overall["Accuracy"], svm_conf$overall["Accuracy"])
)

print(results)

# ---- Visualization ----
ggplot(results, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", width = 0.6) +
  ylim(0, 1) +
  ggtitle("Model Accuracy Comparison") +
  theme_minimal()
