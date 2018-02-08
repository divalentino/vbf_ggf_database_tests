# vbf_ggf_database_tests
Proof-of-concept project using open-source database software and statistical analysis tools to perform physics analyses.

##################################################
Planned tools:
##################################################

- Apache Spark / python (querying data / ML)
- R (statistical analysis / ML)
- C++ / python / ROOT (physical calculations, e.g. four momenta)
- MadGraph5_aMC@NLO (event generation)
- PYTHIA8 (parton showering)

##################################################
Initial plan for use case:
##################################################

Using multivariate techniques (e.g. boosted decision trees) to classify and identify
vector boson fusion-mediated Higgs production against a ``background" of gluon fusion-
mediated events.

For example:
1. Generate events in MG5, shower in PYTHIA8 to produce ROOT ntuples
2. use ROOT to create CSVs *
3. Import CSVs into Spark
4. Use Spark to create training / testing sets, train multivariate classifier
   (e.g. boosted decision tree)
5. Apply to testing set, read results into R to perform statistical analysis, 
   e.g. significance calculations, negative log-likelihood hypothesis testing

* Not sure if R has physics libraries which can do four momentum calculations.
If so, this would be a viable alternative.