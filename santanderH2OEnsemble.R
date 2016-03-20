source("install.R")
library(h2o)
library(h2oEnsemble)
library(SuperLearner)
library(cvAUC)
library(readr)
library(stringr)
library(caret)

#logloss
MultiLogLoss <- function(act, pred)
{
        eps = 1e-15;
        nr <- nrow(pred)
        pred = matrix(sapply( pred, function(x) max(eps,x)), nrow = nr)      
        pred = matrix(sapply( pred, function(x) min(1-eps,x)), nrow = nr)
        ll = sum(act*log(pred) + (1-act)*log(1-pred))
        ll = ll * -1/(nrow(act))      
        return(ll);
}



##load data frames and feature engineering
train <- read_csv("./data/train.csv")
test <- read_csv("./data/test.csv")

test$TARGET<-NA
test$tt<-'test'
train$tt<-'train'
ttd<-rbind(train,test)

feature.names <- names(ttd)
ttd$TARGET<-as.factor(ttd$TARGET)

findCorrelation(cor(ttd[,1:323]), cutoff = 0.9, verbose = TRUE,names=TRUE)

?findCorrelation

##### Removing constant features
cat("\n## Removing the constants features.\n")
for (f in names(ttd)) {
        if (length(unique(ttd[[f]])) == 1) {
                cat(f, "is constant in train. We delete it.\n")
                ttd[[f]] <- NULL
        }
}

##### Removing identical features
features_pair <- combn(names(ttd), 2, simplify = F)
toRemove <- c()
for(pair in features_pair) {
        f1 <- pair[1]
        f2 <- pair[2]
        
        if (!(f1 %in% toRemove) & !(f2 %in% toRemove)) {
                if (all(ttd[[f1]] == ttd[[f2]])) {
                        cat(f1, "and", f2, "are equals.\n")
                        toRemove <- c(toRemove, f2)
                }
        }
}

feature.names <- setdiff(names(train), toRemove)

train <- train[, feature.names]
test <- test[, feature.names]


##### Removing identical features
features_pair <- combn(names(train), 2, simplify = F)
toRemove <- c()
for(pair in features_pair) {
        f1 <- pair[1]
        f2 <- pair[2]
        
        if (!(f1 %in% toRemove) & !(f2 %in% toRemove)) {
                if (all(train[[f1]] == train[[f2]])) {
                        cat(f1, "and", f2, "are equals.\n")
                        toRemove <- c(toRemove, f2)
                }
        }
}


N <- ncol(ttd)
ttd$napercent <- rowSums(is.na(ttd)) / N
ttd$zeropercent <- rowSums(ttd[,feature.names]== 0) / N








test<-ttd[ttd$tt=='test',]
train<-ttd[ttd$tt=='train',]
train$tt<-NULL
test$tt<-NULL
test$TARGET<-NULL






        
#load data and make some feature eng
h2o.init(nthreads=-1,max_mem_size='30g')

train.hex<-as.h2o(train, destination_frame="train.hex")
test.hex<-as.h2o(test, destination_frame="test.hex")
train.hex$TARGET = as.factor(train.hex$TARGET)


#models to use
h2o.randomForest.1 <<- function(..., ntrees = 300, nbins = 50, sample_rate = 0.85,seed = 123) h2o.randomForest.wrapper(..., ntrees = ntrees, nbins = nbins, sample_rate = sample_rate,seed = seed)
h2o.randomForest.2 <<- function(..., ntrees = 200, nbins = 30, balance_classes = TRUE, seed = 125) h2o.randomForest.wrapper(..., ntrees = ntrees, nbins = nbins, balance_classes = balance_classes, seed = seed)
h2o.gbm.1 <<- function(..., ntrees = 200, col_sample_rate = 0.7, max_depth = 10, seed = 12) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, seed = seed)
h2o.gbm.2 <<- function(..., ntrees = 150, balance_classes = TRUE, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, balance_classes = balance_classes, seed = seed)
h2o.deeplearning.1 <<- function(..., hidden = c(500,500,500), activation = "Rectifier", seed = 17)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.2 <<- function(..., hidden = c(500,500), activation = "RectifierWithDropout", epochs = 50, seed = 247)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.glm.1 <<- function(..., alpha = 1.0) h2o.glm.wrapper(..., alpha = alpha)
h2o.glm.2 <<- function(..., alpha = 0.5) h2o.glm.wrapper(..., alpha = alpha)

## Base learners
learner <<- c("h2o.randomForest.1","h2o.randomForest.2","h2o.gbm.1","h2o.gbm.2",
              "h2o.deeplearning.1","h2o.deeplearning.2","h2o.glm.1","h2o.glm.2")

#training
splits = h2o.splitFrame(train.hex, 0.95, destination_frames=c("trainSplit","testSplit"))
fit = h2o.ensemble(x = 3:ncol(train.hex),
                   y = 2, 
                   training_frame = splits[[1]], 
                   family = "binomial", 
                   learner = learner,
                   metalearner ="h2o.glm.wrapper",
                   cvControl = list(V=6,shuffle = TRUE)
)

## Predict
p = predict(fit, splits[[2]])
labels = as.data.frame(splits[[2]][,"target"])
labels_num = as.data.frame(as.numeric(splits[[2]][,"target"]))

h2oPredRF1  = as.data.frame(p$basepred$h2o.randomForest.1)
h2oPredRF2  = as.data.frame(p$basepred$h2o.randomForest.2)
h2oPredDL1 = as.data.frame(p$basepred$h2o.deeplearning.1)
#h2oPredDL2 = as.data.frame(p$basepred$h2o.deeplearning.2)
#h2oPredGBM1 = as.data.frame(p$basepred$h2o.gbm.1)
h2oPredGBM2 = as.data.frame(p$basepred$h2o.gbm.2)
h2oPredGLM1 = as.data.frame(p$basepred$h2o.glm.1)
#h2oPredGLM2 = as.data.frame(p$basepred$h2o.glm.2)
h2oPredEns = as.data.frame(p$pred$p1)

# Metrics - AUC, Logloss
metrics <- matrix(c(AUC(predictions=h2oPredRF1, labels=labels),
                    AUC(predictions=h2oPredRF2, labels=labels),
                    AUC(predictions=h2oPredDL1, labels=labels),
                    AUC(predictions=h2oPredDL2, labels=labels),
                    AUC(predictions=h2oPredGBM1, labels=labels),
                    AUC(predictions=h2oPredGBM2, labels=labels),
                    AUC(predictions=h2oPredGLM1, labels=labels),
                    AUC(predictions=h2oPredGLM2, labels=labels),
                    AUC(predictions=h2oPredEns, labels=labels),
                    MultiLogLoss(labels_num,h2oPredRF1),
                    MultiLogLoss(labels_num,h2oPredRF2),
                    MultiLogLoss(labels_num,h2oPredDL1),
                    MultiLogLoss(labels_num,h2oPredDL2),
                    MultiLogLoss(labels_num,h2oPredGBM1),
                    MultiLogLoss(labels_num,h2oPredGBM2),
                    MultiLogLoss(labels_num,h2oPredGLM1),
                    MultiLogLoss(labels_num,h2oPredGLM2),
                    MultiLogLoss(labels_num,h2oPredEns)),
                  ncol=9,byrow=TRUE)
colnames(metrics) <- c("RF1","RF2","DL1","DL2","GBM1",
                       "GBM2","GLM1","GLM2","ENSEMBLE")
rownames(metrics) <- c("AUC","Logloss")
metrics <- as.table(metrics)



#predict dataset
pTest <- predict(fit, test.hex)
pEns<-as.data.frame(pTest$pred$p1)
pID<-as.data.frame(test.hex$ID)
submission <- as.data.frame(cbind(pID,pEns))
colnames(submission) <- c("ID", "TARGET")

return(list(submission,metrics))
#write.csv(as.data.frame(submission), file = "./results/submission2.csv", quote = F, row.names = F)










v5part_result<-EnsTrain(v5part.train,v5part.test)
v5part_submission<-v5part_result[[1]]
v5part_result[[2]]




#generate final submission
final_subm<-rbind(v5part_submission,v101part_submission,v101part.na_submission)

write.csv(final_subm, file = "./results/submission3ds.csv", quote = F, row.names = F)

 