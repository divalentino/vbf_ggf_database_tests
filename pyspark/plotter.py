# Plotting tools.
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

def plot_bdt_scores(gbt_pred_vbf, gbt_pred_ggf, gbt_weights_vbf, gbt_weights_ggf) :

    pp = PdfPages('bdt_score.pdf')
    
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
    bins = np.linspace(0,1,15);

    plt.hist(gbt_scores_vbf, bins, alpha=1.0, label='VBF', color='blue', weights = gbt_weights_vbf)
    plt.hist(gbt_scores_ggf, bins, alpha=0.5, label='ggF', color='red', weights = gbt_weights_ggf)
    plt.legend(loc='upper right')
    plt.savefig(pp,format='pdf')

    pp.close()
    
    #plt.show()

#def plot_sig_scan(score_cuts, sig_values) :
    
