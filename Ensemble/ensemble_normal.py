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

    # For Instant Model Verification
    
    per1 = calPer (model, cv_mal, -1)
    per2 = calPer (model, test_mal_data, -1)
    per3 = calPer (model, cv_norm)
    per4 = calPer (model, test_norm_data)

    """ cvScore = model.score_samples(cvData)
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
    misClassified=[round(1-classified[k],2) for k in range(len(testScore))] """
    
    classified = 0
    misClassified = 0
    return classified, misClassified, per1, per2, per3, per4


# Train Malware

file = open("normal.csv", "w+", 0)
val = 0.00000001
mylist = []
while val < 0.000151811270299:
    val = val * 1.5
    mylist.append(val)

mylist.reverse()

for i in mylist:
    for j in mylist:
        print(i, j)
        normResult = trainTestModel (i, 'rbf', j, train_norm, cv_norm, test_norm_data)
        print("----")
        if normResult[2] > 50 and normResult[3] > 50 and normResult[4] > 50 and normResult[5] > 50:
            file.write(str(i) + "," + str(j) + "," + str(normResult[2]) + "," + str(normResult[3]) + "," + str(normResult[4]) + "," + str(normResult[5]) + "\n")
