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
    
    per1 = calPer (model, cv_mal)
    per2 = calPer (model, test_mal_data)
    per3 = calPer (model, cv_norm, -1)
    per4 = calPer (model, test_norm_data, -1)

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

file = open("malware.csv", "w+", 0)
val = 0.00000001
mylist = []
while val < 0.1:
    val = val * 1.9
    mylist.append(val)

for i in mylist:
    for j in mylist:
        print(i, j)
        malResult = trainTestModel (i, 'rbf', j, train_mal, cv_mal, test_mal_data)
        print("----")
        if malResult[2] > 60 and malResult[3] > 60 and malResult[4] > 60 and malResult[5] > 60:
            file.write(str(i) + "," + str(j) + "," + str(malResult[2]) + "," + str(malResult[3]) + "," + str(malResult[4]) + "," + str(malResult[5]) + "\n")


""" malClassified = malResult[0]
malMisClassified = malResult[1]

# Train Normal
normResult = trainTestModel (0.001, 'rbf', 0.00001, train_norm, cv_norm, test_mal_data)
normClassified = normResult[0]
normMisClassified = normResult[1]

# Ensemble Result
malClassResult=[malClassified[j]*0.5+normMisClassified[j]*0.5 for j in range(len(malClassified))]
normClassResult=[malMisClassified[j]*0.5+normClassified[j]*0.5 for j in range(len(malClassified))]

malCnt=0;normCnt=0
for k in range(len(malClassResult)):
    if(malClassResult[k]>normClassResult[k]):
        malCnt=malCnt+1
    else:
        normCnt=normCnt+1

print (malCnt, normCnt) """
