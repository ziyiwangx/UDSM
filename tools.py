import numpy as np
import scipy.io as sio 
import scipy.sparse as sp


def load_data(path, dtype=np.float32):
    db = sio.loadmat(path)
    traindata = dtype(db['traindata'])
    testdata = dtype(db['testdata'])
    cateTrainTest = dtype(db['cateTrainTest'])

    mean = np.mean(traindata, axis=0)
    traindata -= mean
    testdata -= mean

    return traindata, testdata, cateTrainTest

def save_sparse_matrix(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape, _type=array.__class__)

def load_sparse_matrix(filename):
    matrix = np.load(filename)

    _type = matrix['_type']
    sparse_matrix = _type.item(0)

    return sparse_matrix((matrix['data'], matrix['indices'],
                                 matrix['indptr']), shape=matrix['shape'])

def binarize_adj(adj):
    adj[adj != 0] = 1
    return adj
        
def renormalize_adj(adj):
    rowsum = np.array(adj.sum(axis=1))
    inv = np.power(rowsum, -0.5).flatten()
    inv[np.isinf(inv)] = 0.
    zdiag = sp.diags(inv)     

    return adj.dot(zdiag).transpose().dot(zdiag)

def sign_dot(data, func):
    return np.sign(np.dot(data, func))


def mAP(cateTrainTest, IX, num_return_NN=None):
    numTrain, numTest = IX.shape

    num_return_NN = numTrain if not num_return_NN else num_return_NN

    apall = np.zeros((numTest, 1))
    yescnt_all = np.zeros((numTest, 1))
    for qid in range(numTest):
        query = IX[:, qid]
        x, p = 0, 0

        for rid in range(num_return_NN):
            if cateTrainTest[query[rid], qid]:
                x += 1
                p += x / (rid * 1.0 + 1.0)

        yescnt_all[qid] = x
        if not p:
            apall[qid] = 0.0
        else:
            apall[qid] = p / (num_return_NN * 1.0)

    return np.mean(apall), apall, yescnt_all


'''    # ------------------ different phase ------------------
def mAP(cateTrainTest, IX, label_phase_num, label_phase, num_return_NN=None):
    numTrain, numTest = IX.shape

    num_return_NN = numTrain if not num_return_NN else num_return_NN

    apall = np.zeros((numTest, 1))
    yescnt_all = np.zeros((numTest, 1))

    # --------------------- phase 1 --------------------#
    apall_phase1 = np.zeros((label_phase_num[0], 1))
    yescnt_all_phase1 = np.zeros((label_phase_num[0], 1))

    for qid_phase1 in range(label_phase_num[0]):
        qid = label_phase[qid_phase1]
        query = IX[:, qid]
        x, p = 0, 0

        for rid in range(num_return_NN):
            if cateTrainTest[query[rid], qid]:
                x += 1
                p += x / (rid * 1.0 + 1.0)

        yescnt_all_phase1[qid_phase1] = x

        if not p:
            apall_phase1[qid_phase1] = 0.0
        else:
            apall_phase1[qid_phase1] = p / (num_return_NN * 1.0)

    # --------------------- phase 2 --------------------#
    apall_phase2 = np.zeros((label_phase_num[1], 1))
    yescnt_all_phase2 = np.zeros((label_phase_num[1], 1))

    for qid_phase2 in range(label_phase_num[1]):
        qid = label_phase[label_phase_num[0] + qid_phase2]
        query = IX[:, qid]
        x, p = 0, 0

        for rid in range(num_return_NN):
            if cateTrainTest[query[rid], qid]:
                x += 1
                p += x / (rid * 1.0 + 1.0)

        yescnt_all_phase2[qid_phase2] = x

        if not p:
            apall_phase2[qid_phase2] = 0.0
        else:
            apall_phase2[qid_phase2] = p / (num_return_NN * 1.0)

    # --------------------- phase 3 --------------------#
    apall_phase3 = np.zeros((label_phase_num[2], 1))
    yescnt_all_phase3 = np.zeros((label_phase_num[2], 1))

    for qid_phase3 in range(label_phase_num[2]):
        qid = label_phase[label_phase_num[0] + label_phase_num[1] + qid_phase3]
        query = IX[:, qid]
        x, p = 0, 0

        for rid in range(num_return_NN):
            if cateTrainTest[query[rid], qid]:
                x += 1
                p += x / (rid * 1.0 + 1.0)

        yescnt_all_phase3[qid_phase3] = x

        if not p:
            apall_phase3[qid_phase3] = 0.0
        else:
            apall_phase3[qid_phase3] = p / (num_return_NN * 1.0)

    # --------------------- phase 4 --------------------#
    apall_phase4 = np.zeros((label_phase_num[3], 1))
    yescnt_all_phase4 = np.zeros((label_phase_num[3], 1))

    for qid_phase4 in range(label_phase_num[3]):
        qid = label_phase[label_phase_num[0] + label_phase_num[1] + label_phase_num[2] + qid_phase4]
        query = IX[:, qid]
        x, p = 0, 0

        for rid in range(num_return_NN):
            if cateTrainTest[query[rid], qid]:
                x += 1
                p += x / (rid * 1.0 + 1.0)

        yescnt_all_phase4[qid_phase4] = x

        if not p:
            apall_phase4[qid_phase4] = 0.0
        else:
            apall_phase4[qid_phase4] = p / (num_return_NN * 1.0)

    # --------------------- phase 5 --------------------#
    apall_phase5 = np.zeros((label_phase_num[4], 1))
    yescnt_all_phase5 = np.zeros((label_phase_num[4], 1))

    for qid_phase5 in range(label_phase_num[4]):
        qid = label_phase[label_phase_num[0] + label_phase_num[1] + label_phase_num[2] + label_phase_num[3] + qid_phase5]
        query = IX[:, qid]
        x, p = 0, 0

        for rid in range(num_return_NN):
            if cateTrainTest[query[rid], qid]:
                x += 1
                p += x / (rid * 1.0 + 1.0)

        yescnt_all_phase5[qid_phase5] = x

        if not p:
            apall_phase5[qid_phase5] = 0.0
        else:
            apall_phase5[qid_phase5] = p / (num_return_NN * 1.0)

    # --------------------- phase 6 --------------------#
    apall_phase6 = np.zeros((label_phase_num[5], 1))
    yescnt_all_phase6 = np.zeros((label_phase_num[5], 1))

    for qid_phase6 in range(label_phase_num[5]):
        qid = label_phase[label_phase_num[0] + label_phase_num[1] + label_phase_num[2] + label_phase_num[3] + label_phase_num[4] + qid_phase6]
        query = IX[:, qid]
        x, p = 0, 0

        for rid in range(num_return_NN):
            if cateTrainTest[query[rid], qid]:
                x += 1
                p += x / (rid * 1.0 + 1.0)

        yescnt_all_phase6[qid_phase6] = x

        if not p:
            apall_phase6[qid_phase6] = 0.0
        else:
            apall_phase6[qid_phase6] = p / (num_return_NN * 1.0)

    # --------------------- phase 7 --------------------#
    apall_phase7 = np.zeros((label_phase_num[6], 1))
    yescnt_all_phase7 = np.zeros((label_phase_num[6], 1))

    for qid_phase7 in range(label_phase_num[6]):
        qid = label_phase[label_phase_num[0] + label_phase_num[1] + label_phase_num[2] + label_phase_num[3] + label_phase_num[4] + label_phase_num[5] + qid_phase7]
        query = IX[:, qid]
        x, p = 0, 0

        for rid in range(num_return_NN):
            if cateTrainTest[query[rid], qid]:
                x += 1
                p += x / (rid * 1.0 + 1.0)

        yescnt_all_phase7[qid_phase7] = x

        if not p:
            apall_phase7[qid_phase7] = 0.0
        else:
            apall_phase7[qid_phase7] = p / (num_return_NN * 1.0)

    # --------------------- all --------------------#

    for qid in range(numTest):
        query = IX[:, qid]
        x, p = 0, 0

        for rid in range(num_return_NN):
            if cateTrainTest[query[rid], qid]:
                x += 1
                p += x/(rid*1.0 + 1.0)
	
        yescnt_all[qid] = x

        if not p: apall[qid] = 0.0
        else: apall[qid] = p/(num_return_NN*1.0)

    return np.mean(apall),apall,yescnt_all,np.mean(apall_phase1),np.mean(apall_phase2),np.mean(apall_phase3),np.mean(apall_phase4),np.mean(apall_phase5),np.mean(apall_phase6),np.mean(apall_phase7)
'''


def topK(cateTrainTest, HammingRank, k=500):
    numTest = cateTrainTest.shape[1]

    precision = np.zeros((numTest, 1))
    recall = np.zeros((numTest, 1))

    topk = HammingRank[:k, :]

    for qid in range(numTest):
        retrieved = topk[:, qid]
        rel = cateTrainTest[retrieved, qid]
        retrieved_relevant_num = np.sum(rel)
        real_relevant_num = np.sum(cateTrainTest[:, qid])

        precision[qid] = retrieved_relevant_num/(k*1.0)
        recall[qid] = retrieved_relevant_num/(real_relevant_num*1.0)

    return precision.mean(), recall.mean()


def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def calc_map(qB, rB, query_L, retrieval_L):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # query_L: {0,1}^{mxl}
    # retrieval_L: {0,1}^{nxl}
    num_query = query_L.shape[0]
    map = 0
    for iter in range(num_query):
        #print('iter', iter)
        gnd = (np.dot(query_L[iter, :], retrieval_L.transpose()) > 0).astype(np.float32)
        #print('gnd', gnd)
        tsum = int(np.sum(gnd))
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        count = np.linspace(1, tsum, tsum)
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        #print('count', count, 'tindex', tindex)
        map = map + np.mean(count / (tindex[1]))
        #print('map', map)
    map = map / num_query
    #print('map', map)
    return map


if __name__ == '__main__':
    hashcode = np.array([[1,0,1,1,0],[0,1,0,1,0],[0,0,1,0,1],[1,0,0,1,0],[1,0,0,0,1],[0,1,0,1,0],[0,0,1,1,0],[0,1,0,1,0]])
    labels = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[1,0,0,0],[0,0,0,1],[0,1,0,0],[0,0,1,0],[0,1,0,0]])
    print('labels:\n',labels)
    hashcode[hashcode==0]=-1
    print(hashcode)

    hammingDist = 0.5*(-np.dot(hashcode,hashcode.transpose())+5)
    print('hammingDist: \n',hammingDist)
    HammingRank = np.argsort(hammingDist, axis=0)
    print('Hamming Rank: \n',HammingRank)

    sim_matrix = np.dot(labels,labels.transpose())
    print('sim_matrix: \n',sim_matrix)
    map = mAP(sim_matrix,HammingRank)
    print(map)


