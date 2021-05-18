# understand_gbm.R

# For visualization functions of xgboost, you may need to set up some packages.
#install.packages("igraph")
#install.packages("DiagrammeR")

#rds_file <- "C:/RMV2.0 Workshop/deteleme/GBMProject.rds"
rds_file <- "C:/RMV2.0 Workshop/deteleme/Project_05.12.rds"
#rds_file <- "C:/RMV2.0 Workshop/deteleme/Project_05.13.rds"

Project <- readRDS(rds_file)
cat("Your project has the following models.",fill=T)
print(names(Project$model_obj_list$models_list))

#library(xgboost)
res_baseline = Project$model_obj_list$models_list[["Data_pre_2_2.csv"]]
model_raw = res_baseline$gbm_model_raw
model_ser = res_baseline$gbm_model_serialized

gbm_model1=xgboost::xgb.load("C:/RMV2.0 Workshop/deteleme/xgb.model")
gbm_model2=xgboost::xgb.load.raw(model_raw)
gbm_model3=xgboost::xgb.unserialize(model_ser)

#raw <- xgboost::xgb.save.raw(gbm_model1)
#bst <- xgboost::xgb.load.raw(raw)

train = res_baseline$train
variables = res_baseline$variables
print(variables)
train_input <- train[,variables]
y_fit <- predict(gbm_model1, as.matrix(train_input))
# OK, even though the object looks empty, this still works.
y_fit <- predict(gbm_model2, as.matrix(train_input))

# This only works when model was stored with xgb.load(), not load.raw or unserialize
xgboost::xgb.plot.deepness(gbm_model1)

# This doesn't work unless we set the cb.gblinear.history() callback:
#xgboost::xgb.gblinear.history(gbm_model1)
#Error in xgboost::xgb.gblinear.history(gbm_model1) :
#  model must be trained while using the cb.gblinear.history() callback

# A data.table with columns Feature, Gain, Cover, Frequency
importance_matrix <- xgboost::xgb.importance(colnames(as.matrix(train_input)), model = gbm_model1)

xgboost::xgb.plot.importance(
  importance_matrix = importance_matrix,
  rel_to_first = TRUE, xlab = "Relative importance"
)

# An interesting list of the trees, and their nodes and leaves
# In the sample project, there are:
# 400 trees/boosters, and
# 23186-400 = 22786 non-root nodes, and
# on average, 57 non-root nodes per tree
# Would love to visualize one tree, for example
model_dump = xgboost::xgb.dump(gbm_model1, with_stats=T)
head(model_dump)
tail(model_dump)

# Plot all the trees.
# Just kidding, we have 400 trees. Seriously, don't do it.
#xgboost::xgb.plot.tree(model = gbm_model1)

# Plot only the first tree and display the node IDs:
xgboost::xgb.plot.tree(model = gbm_model1, trees = 0, show_node_id = TRUE)
#xgboost::xgb.plot.tree(model = gbm_model1, trees = 1, show_node_id = TRUE)
#xgboost::xgb.plot.tree(model = gbm_model1, trees = 0:2, show_node_id = TRUE)

#"This function tries to capture the complexity of a gradient boosted tree model
#in a cohesive way by compressing an ensemble of trees into a single tree-graph
#representation. The goal is to improve the interpretability of a model generally
#seen as black box."

#p <- xgboost::xgb.plot.multi.trees(model = gbm_model1, feature_names = variables)
#print(p)

# This fails with error message:
#Column 2 ['No'] of item 2 is missing in item 1. Use fill=TRUE to fill with NA (NULL for list columns), or
#use.names=FALSE to ignore column names. use.names='check' (default from v1.12.2) emits this message and
#proceeds as if use.names=FALSE for  backwards compatibility. See news item 5 in v1.12.2 for options to
#control this message.
