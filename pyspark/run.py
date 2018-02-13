# Spark instance imports.
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

# CSV reader.
from get_data import get_data

# For preparing data.
from data_formatter import data_formatter

# Decision tree imports.
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col

# Plotting tools.
import matplotlib.pyplot as plt
import numpy as np

################################################################################
# Depending on whether we're running in pyspark
# or submiting a job, instantiate a spark session.
################################################################################

spark = SparkSession.builder.config("k1", "v1").getOrCreate();

################################################################################
# File I/O.
################################################################################

# Read in the CSVs and perform initial selection.
print "Reading input data"
vbf_data = get_data(spark,"../output/vbf.csv");
ggf_data = get_data(spark,"../output/ggf.csv");

# Split each into training/testing samples.
split_vbf = vbf_data.randomSplit([0.5,0.5],1);
split_ggf = ggf_data.randomSplit([0.5,0.5],1);

# Format the data in a way usable for ML algorithm.
train_vbf = data_formatter(split_vbf[0], 1,["DelEta_jj", "DijetMass", "LeadingJetPt"]);
train_ggf = data_formatter(split_ggf[0], 0,["DelEta_jj", "DijetMass", "LeadingJetPt"]);

# Append the training data together.
train_total = train_vbf.union(train_ggf);

# Create equivalent testing samples.
test_vbf  = data_formatter(split_vbf[1], 1,["DelEta_jj", "DijetMass", "LeadingJetPt"]);
test_ggf  = data_formatter(split_ggf[1], 0,["DelEta_jj", "DijetMass", "LeadingJetPt"]);

# Build vectors of weights for the test events
# for later use in statistical analysis.
weight_vec_vbf  = test_vbf.select("weight").collect()
weight_vec_ggf  = test_ggf.select("weight").collect()
gbt_weights_vbf = []
gbt_weights_ggf = []

for elt in range(0,len(weight_vec_vbf)) :
   gbt_weights_vbf.append(2.0*weight_vec_vbf[elt][0])
for elt in range(0,len(weight_vec_ggf)) :
   gbt_weights_ggf.append(2.0*weight_vec_ggf[elt][0])

################################################################################
# Instantiate machine learning algorithm, then train and test.
################################################################################

# Create classifier (decision tree).
# dt             = DecisionTreeClassifier(labelCol    = "label",
#                                         featuresCol = "features")
# dt_model       = dt.fit(train_total);
# dt_pred_vbf    = dt_model.transform(test_vbf);
# dt_pred_ggf    = dt_model.transform(test_ggf);

# Try a gradient boosted decision tree instead.
gbt          = GBTClassifier(labelCol="label", featuresCol="features", maxIter=10);
gbt_model    = gbt.fit(train_total);
gbt_pred_vbf = gbt_model.transform(test_vbf);
gbt_pred_ggf = gbt_model.transform(test_ggf);

################################################################################
# Evaluate efficacy of algorithm.
################################################################################

# Try extracting probabilities from prediction data frame.
gbt_scores_vbf = [];
gbt_scores_ggf = [];

coll_vbf = gbt_pred_vbf.select(gbt_pred_vbf["probability"]).collect();
coll_ggf = gbt_pred_ggf.select(gbt_pred_ggf["probability"]).collect();

for elt in range(0,len(coll_vbf)) :
   gbt_scores_vbf.append(coll_vbf[elt][0][0])
for elt in range(0,len(coll_ggf)) :
   gbt_scores_ggf.append(coll_ggf[elt][0][0])

# Superimpose histograms of the BDT probability
# (analogous to BDT response?)
bins = np.linspace(0,1,25);

plt.hist(gbt_scores_vbf, bins, alpha=0.5, label='VBF', color='blue', weights = gbt_weights_vbf)
plt.hist(gbt_scores_ggf, bins, alpha=0.5, label='ggF', color='red', weights = gbt_weights_ggf)
plt.legend(loc='upper right')
plt.show()

# TO DO: Need to map the individual events to their event weights,
# so that yield distributions can be drawn.
