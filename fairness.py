import numpy as np
import pandas as pd

class Fairness(object):
    """
    Compute fairness metrics
    """
    def __init__(self, G, test_nodes_idx, targets, predictions, sens_attr, neptune_run):
        self.sens_attr = sens_attr        
        self.neptune_run = neptune_run
        self.neptune_run["sens_attr"] = self.sens_attr
        self.G = G
        self.test_nodes_idx = test_nodes_idx.cpu().detach().numpy()
        self.true_y = np.asarray(targets) # target variables
        self.pred_y = np.asarray(predictions) # prediction of the classifier
        self.sens_attr_array = self.G.nodes["user"].data[self.sens_attr].cpu().detach().numpy() # sensitive attribute values
        self.sens_attr_values = self.sens_attr_array[self.test_nodes_idx]
        self.s0 = self.sens_attr_values == 0
        self.s1 = self.sens_attr_values == 1
        self.y1_s0 = np.bitwise_and(self.true_y==1, self.s0)
        self.y1_s1 = np.bitwise_and(self.true_y==1, self.s1)
        self.y0_s0 = np.bitwise_and(self.true_y==0, self.s0)
        self.y0_s1 = np.bitwise_and(self.true_y==0, self.s1)

        neptune_run["# pos"] = np.count_nonzero(self.sens_attr_values)
        neptune_run["# neg"] = np.count_nonzero(self.sens_attr_values==0)
    
    def statistical_parity(self):
        ''' P(y^=1|s=0) = P(y^=1|s=1) '''
        # stat_parity = abs(sum(pred_y[s0]) / sum(s0) - sum(pred_y[s1]) / sum(s1))
        stat_parity = sum(self.pred_y[self.s0]) / sum(self.s0) - sum(self.pred_y[self.s1]) / sum(self.s1)
        self.neptune_run["fairness/SPD"] = stat_parity
        print(" Statistical Parity Difference (SPD): {:.4f}".format(stat_parity))

    
    def equal_opportunity(self):
        ''' P(y^=1|y=1,s=0) = P(y^=1|y=1,s=1) '''
        # equal_opp = abs(sum(pred_y[y1_s0]) / sum(y1_s0) - sum(pred_y[y1_s1]) / sum(y1_s1))
        equal_opp = sum(self.pred_y[self.y1_s0]) / sum(self.y1_s0) - sum(self.pred_y[self.y1_s1]) / sum(self.y1_s1)
        self.neptune_run["fairness/EOD"] = equal_opp
        print(" Equal Opportunity Difference (EOD): {:.4f}".format(equal_opp))


    def overall_accuracy_equality(self):
        ''' P(y^=0|y=0,s=0) + P(y^=1|y=1,s=0) = P(y^=0|y=0,s=1) + P(y^=1|y=1,s=1) '''
        oae_s0 = np.count_nonzero(self.pred_y[self.y0_s0]==0) / sum(self.y0_s0) + sum(self.pred_y[self.y1_s0]) / sum(self.y1_s0)
        oae_s1 = np.count_nonzero(self.pred_y[self.y0_s1]==0) / sum(self.y0_s1) + sum(self.pred_y[self.y1_s1]) / sum(self.y1_s1)
        oae_diff = oae_s0 - oae_s1
        self.neptune_run["fairness/OAED"] = oae_diff
        print(" Overall Accuracy Equality Difference (OAED): {:.4f}".format(oae_diff))


    def treatment_equality(self):
        ''' P(y^=1|y=0,s=0) / P(y^=0|y=1,s=0) = P(y^=1|y=0,s=1) / P(y^=0|y=1,s=1) '''
        ''' P(y^=0|y=1,s=0) / P(y^=1|y=0,s=0) = P(y^=0|y=1,s=1) / P(y^=1|y=0,s=1) '''
        # te_s0 = (sum(self.pred_y[self.y0_s0]) / sum(self.y0_s0)) / (np.count_nonzero(self.pred_y[self.y1_s0]==0) / sum(self.y1_s0))
        # te_s1 = (sum(self.pred_y[self.y0_s1]) / sum(self.y0_s1)) / (np.count_nonzero(self.pred_y[self.y1_s1]==0) / sum(self.y1_s1))
        te_s0 = (np.count_nonzero(self.pred_y[self.y1_s0]==0) / sum(self.y1_s0)) / (sum(self.pred_y[self.y0_s0]) / sum(self.y0_s0))
        te_s1 = (np.count_nonzero(self.pred_y[self.y1_s1]==0) / sum(self.y1_s1)) / (sum(self.pred_y[self.y0_s1]) / sum(self.y0_s1))
        te_diff = te_s0 - te_s1
        self.neptune_run["fairness/TED"] = te_diff
        print(" Treatment Equality Difference (TED): {:.4f}".format(te_diff))