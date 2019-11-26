import pandas as pd
import numpy as np
from sklearn import svm

cv_mal=pd.read_csv("cv_mal10k.csv")
cv_norm=pd.read_csv("cv_normal10k.csv")
test_mal_data=pd.read_csv("final_test_malware.csv")
test_norm_data=pd.read_csv("final_test_normal.csv")
train_mal=pd.read_csv("train_malware50k.csv")
train_norm=pd.read_csv("train_normal50k.csv")
#del test_mal_data['Unnamed: 0']

def calPer (model, data, flag = 1):
    # Assuming less than 100% accuracy
    testPred = model.predict(data)
    unique, counts = np.unique(testPred, return_counts=True)
    res = np.asarray((unique, counts)).T
    misClassified = res[0][1]
    classified = 0
    if len(res) == 2:
        classified = res[1][1]
    per = (classified * 100) / (classified + misClassified)
    if flag == -1:
        per = (misClassified * 100) / (classified + misClassified)
    print (per)
    return per


def trainTestModel (_nu, _kernel, _gamma, trainData, cvData, testData):
    model = svm.OneClassSVM(nu=_nu, kernel=_kernel, gamma=_gamma)
    model.fit(trainData)

    cvScore = model.score_samples(cvData)
    minVal = np.amin(cvScore)
    maxVal = np.amax(cvScore)
    rangeVal = maxVal - minVal

    testScore = model.score_samples(testData)
    for i in range(len(testScore)):
        if(testScore[i]<minVal):
            testScore[i]=minVal
        elif(testScore[i]>maxVal):
            testScore[i]=maxVal

    classified=[round((1/rangeVal)*(testScore[i]-minVal),2) for i in range(len(testScore))]
    misClassified=[round(1-classified[k],2) for k in range(len(testScore))]

    return classified, misClassified


def tempFunNorm (i, j,k, l):
    malResult = trainTestModel (i, 'rbf', j, train_mal, cv_mal, test_norm_data)
    malClassified = malResult[0]
    malMisClassified = malResult[1]
    
    normResult = trainTestModel (k, 'rbf', l, train_norm, cv_norm, test_norm_data)
    normClassified = normResult[0]
    normMisClassified = normResult[1]

    # Ensemble Result
    malClassResult=[malClassified[j1]*0.5+normMisClassified[j1]*0.5 for j1 in range(len(malClassified))]
    normClassResult=[malMisClassified[j1]*0.5+normClassified[j1]*0.5 for j1 in range(len(malClassified))]

    malCnt=0;normCnt=0
    for k1 in range(len(malClassResult)):
        if(malClassResult[k1]>normClassResult[k1]):
            malCnt=malCnt+1
        else:
            normCnt=normCnt+1

    per = ( normCnt * 100 ) / (malCnt + normCnt)
    return per


def ensemble (testData, fileName, flag = 1):

    file = open(fileName, "w+", 0)
    val = 0.00000001
    mylist = []
    while val < 0.000151811270299:
        val = val * 1.9
        mylist.append(val)
        
    # Train Malware
    i = 2.55476698619e-07
    j = 4.11447777893e-07
    malResult = trainTestModel (i, 'rbf', j, train_mal, cv_mal, testData)
    malClassified = malResult[0]
    malMisClassified = malResult[1]

    # Train Normal
    for k in mylist:
        for l in mylist:
            normResult = trainTestModel (k, 'rbf', l, train_norm, cv_norm, testData)
            normClassified = normResult[0]
            normMisClassified = normResult[1]

            # Ensemble Result
            malClassResult=[malClassified[j1]*0.5+normMisClassified[j1]*0.5 for j1 in range(len(malClassified))]
            normClassResult=[malMisClassified[j1]*0.5+normClassified[j1]*0.5 for j1 in range(len(malClassified))]

            malCnt=0;normCnt=0
            for k1 in range(len(malClassResult)):
                if(malClassResult[k1]>normClassResult[k1]):
                    malCnt=malCnt+1
                else:
                    normCnt=normCnt+1

            malPer = ( malCnt * 100 ) / (malCnt + normCnt)
            normPer = tempFunNorm(i, j, k, l)
            
            print (str(i) + "," + str(j) + "," + str(k) + "," + str(l) + "," + str(malPer) + "," + str(normPer) +  "\n")
            if malPer > 90 and malPer < 100 and normPer > 90 and normPer < 100:
                file.write(str(i) + "," + str(j) + "," + str(k) + "," + str(l) + "," + str(malPer) + "," + str(normPer) +  "\n")
            # print ("mal:", malCnt, "norm:",normCnt)


print ("Mal Results")
ensemble(test_mal_data, "ensRes.csv")
#print ("Norm Results")
#ensemble(test_norm_data, "ensNorm.csv", -1)