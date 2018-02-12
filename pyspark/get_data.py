
# Get relevant data from CSV and put it it a dataframe.
def get_data(spark,filename) :
    in_file = spark.read.csv(filename,header=True,inferSchema=True);
    in_file = in_file.filter((in_file["event_type"]>-1) & (in_file["DijetMass"]>120.0))
    return in_file;
