'''
Created on 2016-03-16

@author: Niranjan
'''

import numpy as np
import sys
import os
from multiprocessing import Pool
np.random.seed(0)   #Set the seed value to get the same random values each time
class LDA:
    def __init__(self, K, alpha, beta, docs, V, smartinit=True):
        self.K = K
        self.alpha = alpha # parameter of topics prior
        self.beta = beta   # parameter of words prior
        self.docs = docs   # parameter of documents prior
        self.V = V

        self.z_m_n = [] # topics of words of documents
        self.n_m_z = np.zeros((len(self.docs), K)) + alpha     # word count of each document and topic
        self.n_z_t = np.zeros((K, V)) + beta # word count of each topic and vocabulary
        self.n_z = np.zeros(K) + V * beta    # word count of each topic

        self.N = 0
        # enumerate through every document
        for m, doc in enumerate(docs):
            self.N += len(doc)
            z_n = []
            for t in doc:
                if smartinit:
                    p_z = self.n_z_t[:, t] * self.n_m_z[m] / self.n_z
                    z = np.random.multinomial(1, p_z / p_z.sum()).argmax()
                else:
                    z = np.random.randint(0, K)
                z_n.append(z)
                self.n_m_z[m, z] += 1
                self.n_z_t[z, t] += 1
                self.n_z[z] += 1
            self.z_m_n.append(np.array(z_n))

    def inference(self):
        """learning once iteration"""
        for m, doc in enumerate(self.docs):
            z_n = self.z_m_n[m]
            n_m_z = self.n_m_z[m]
            for n, t in enumerate(doc):
                # discount for n-th word t with topic z
                z = z_n[n]
                n_m_z[z] -= 1
                self.n_z_t[z, t] -= 1
                self.n_z[z] -= 1

                # sampling topic new_z for t
                p_z = self.n_z_t[:, t] * n_m_z / self.n_z
                new_z = np.random.multinomial(1, p_z / p_z.sum()).argmax()

                # set z the new topic and increment counters
                z_n[n] = new_z
                n_m_z[new_z] += 1
                self.n_z_t[new_z, t] += 1
                self.n_z[new_z] += 1

    def worddist(self):
        """get topic-word distribution normalize self.n_z_t"""
        return self.n_z_t / self.n_z[:, np.newaxis]

    def perplexity(self, docs=None):
        if docs == None: docs = self.docs
        phi = self.worddist()
        log_per = 0
        N = 0
        Kalpha = self.K * self.alpha
        for m, doc in enumerate(docs):
            # to normalize
            theta = self.n_m_z[m] / (len(self.docs[m]) + Kalpha)
            for w in doc:
                log_per -= np.log(np.inner(phi[:,w], theta))
            N += len(doc)
        return np.exp(log_per / N)
    
    def write_data(self):
        
        tp_cnt_f = open('output/lda_topic_cnt_f.mat','w')
        usr_tp_cnt_f = open('output/lda_usr_tp_cnt_f.mat', 'w')
        itm_tp_cnt_f = open('output/lda_itm_tp_cnt_f.mat','w')
        # word count of each topic
        np.save(tp_cnt_f, self.n_z )
        # word count of each document and topic
        np.save(usr_tp_cnt_f,self.n_m_z)
        # word count of each topic and vocabulary
        np.save(itm_tp_cnt_f,self.n_z_t)
        print 'done writing outputs'
        

def lda_learning(lda, iteration):
    pre_perp = lda.perplexity()
    for i in range(iteration):
        print i
        lda.inference()
        perp = lda.perplexity()
        if pre_perp:
            if pre_perp < perp:
                break
            else:
                pre_perp = perp
    lda.write_data()      
    
def train(data):
    
    data = np.load(data)
    alpha = 0.5
    beta = 0.5
    n_topics = 10
    iteration = 30
    train_data = []
    tot_itms = data.shape[1]
    for row in data:
        n_zero_vals = np.ndarray.tolist(np.nonzero(row)[0])
        train_data.append(n_zero_vals) 
    lda = LDA(n_topics, alpha, beta, train_data, tot_itms)
    lda_learning(lda, iteration)
    
# function that calculates the recommended score
def get_recc_score():
    
    rec_f = open('output/recc_scores_lda.mat','w')
    p_z = np.load('output/lda_topic_cnt_f.mat')
    p_u_z = np.load('output/lda_usr_tp_cnt_f.mat')
    p_i_z = np.load('output/lda_itm_tp_cnt_f.mat').T
        
    t_usrs = p_u_z.shape[0]
    t_itms = p_i_z.shape[0]
    recc_items = np.zeros((t_usrs, t_itms))
    
    # calculate the recommendation score for every user and every item
    for u in range(t_usrs):
        scores = 0
        scores = p_z * p_u_z[u] * p_i_z
        # sum over all topics for every item
        scores = np.sum(scores, axis=1)
        recc_items[u] = scores
    # write the scores
    np.save(rec_f,recc_items)
    print 'finished calculating the recommended scores'

# function to calcualte the recommendation performance of LDA
def calc_LDA_performance(data,topk=2):
    
    precisions = []
    recalls = []
    # get the test data
    test_data = np.load(data)
    # get the calculated recommendation scores
    recc_scores = np.load('output/recc_scores_lda.mat')
    
    for i,j in enumerate(recc_scores):
        
        grn_truth = test_data[i]
        # get the total liked items that were marked off from the training set
        tot_likd_itms = len(np.transpose(np.nonzero(grn_truth)))  
        found = 0
        # sort the recommended scores and take top k
        j = np.argsort(j)[-topk:]
        topk_rec_prjs = j[::-1]
        
        for p in topk_rec_prjs:
            # if the sorted item is in the ground truth
            if (grn_truth[p] == 1):
                found += 1
        prec = found/float(topk)
        rec = found/float(tot_likd_itms)
        precisions.append(prec)
        recalls.append(rec)
    avg_prec = np.average(precisions)
    avg_rec = np.average(recalls)
    print avg_prec
    print avg_rec


if __name__ == "__main__":
    
#     create_data()
    trains = ['input/train1.mat']
    tests = ['input/test1.mat']
    for dat in range(len(trains)):
        train(trains[dat])
        get_recc_score()
        calc_LDA_performance(tests[dat])
        
