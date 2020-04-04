# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 14:03:27 2019

@author: Rahul
"""
#---------------------------------------------Importing Packages------------------------------------------------
import pandas as pd
from datetime import datetime
import os
from pathlib import Path
#---------------------------------------------------------------------------------------------------------------

#----------------------------------------------Reading CSV File----------------------------------------------------
dataTrain = pd.read_csv('train.csv')
train =pd.DataFrame(dataTrain)
dataTest = pd.read_csv('test.csv')
test =pd.DataFrame(dataTest)
#------------------------------------------------------------------------------------------------------------------

#------------------------------Converting Class Label Value to Positive or Negative------------------------------
train[train.columns[-1]] = train[train.columns[-1]].map({ 0 : 'negative' , 1 : 'positive' })
test[test.columns[-1]] = test[test.columns[-1]].map({ 0 : 'negative' , 1 : 'positive' })
#-----------------------------------------------------------------------------------------------------------------

#----------------------------------------Splitting into Train - Test----------------------------------------------
#division = (int) ( len(df) * 50/100 )
#train = df[0 : division]
#test = df[division : len(df)]
#test =test.reset_index(drop=True)
#-----------------------------------------------------------------------------------------------------------------

#-----------------------------------------------Naive Bayes-------------------------------------------------------

#-----------------------------------------------Training Start---------------------------------------------------

         #---------------Calculating Prior--------------------
no_of_positive = train[train[train.columns[-1]] == 'positive' ].shape[0]
no_of_negative = train[train[train.columns[-1]] == 'negative' ].shape[0]
p_positive = no_of_positive/(no_of_positive+no_of_negative)
p_negative = no_of_negative/(no_of_positive+no_of_negative)
         #----------------------------------------------------
         
         #---------------Calculating Likelihood---------------
attributes = list(train.columns)
attributes = attributes[:-1]

likelihood = {}
for i in range(len(attributes)):
    likelihood[attributes[i]] = {}
    uniqueValuesOfAttribute = train[str(attributes[i])].unique()
    m = len(uniqueValuesOfAttribute)
    p = 1 / ( len(uniqueValuesOfAttribute) )
    
    for j in range(len(uniqueValuesOfAttribute)):
        attributeValue = uniqueValuesOfAttribute[j]
        likelihood[attributes[i]][attributeValue] = {}
        #----------for Positive-----------------------
        a = train.groupby([attributes[i],train.columns[-1]]).size().reset_index()
        b = a[ (a[attributes[i]]== attributeValue) & (a[train.columns[-1]]=='positive') ]
        if(b.empty):
            countOfPositive = 0
        else:
            countOfPositive = b.iat[0,2]
        
        
        
        #----------------------------------------------
        
        #---------for Negative-------------------------
        a = train.groupby([attributes[i],train.columns[-1]]).size().reset_index()
        b = a[ (a[attributes[i]]== attributeValue) & (a[train.columns[-1]]=='negative') ]
        if(b.empty):
            countOfNegative = 0
        else:
            countOfNegative = b.iat[0,2]
        #------------------------m-estimate Probability Calculation------------------------------------
        m_estimate_positive_for_feature = ( (countOfPositive + (m*p)) / ( (no_of_positive)+m ) )
        m_estimate_negative_for_feature = ( (countOfNegative + (m*p)) / ( (no_of_negative)+m ) )
        likelihood[attributes[i]][attributeValue]['positive'] = m_estimate_positive_for_feature
        likelihood[attributes[i]][attributeValue]['negative'] = m_estimate_negative_for_feature
        #----------------------------------------------------------------------
        
        #-----------------------------------------------

         #----------------------------------------------------

#--------------------------------------------Training End------------------------------------------------------

#--------------------------------------------Testing Start------------------------------------------------------
#print(likelihood)
#--------------------------------------------Testing End--------------------------------------------------------
predictedClassLabel = []
for i in range(len(test)):
    testInstance = test.iloc[i]
    testInstance = testInstance[:-1]
    posteriorProbabilityForPositive = 1
    posteriorProbabilityForNegative = 1
    for j in range(len(testInstance)):
        feature_likelihood = likelihood.get(attributes[j])
        feature_value_likelihood = feature_likelihood.get(testInstance[j])
#        if feature_value_likelihood != None:
        feature_value_likelihood_positive = feature_value_likelihood.get('positive')
        feature_value_likelihood_negative = feature_value_likelihood.get('negative')
        posteriorProbabilityForPositive *= feature_value_likelihood_positive
        posteriorProbabilityForNegative *= feature_value_likelihood_negative
    posteriorProbabilityForPositive *= p_positive
    posteriorProbabilityForNegative *= p_negative
    if posteriorProbabilityForPositive >= posteriorProbabilityForNegative:
        predictedClassLabel.append('positive')
    else:
        predictedClassLabel.append('negative')

#-----------------------------------------Adding new column of prediction and exporting to CSV--------------------------------------
test['Predicted Class'] = predictedClassLabel
dirName = 'Naive-Bayes-Predictions' +' '+str(datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
 
try:
    # Create target Directory
    os.mkdir(dirName)
    
except FileExistsError:
    print("Directory " , dirName ,  " already exists")
#    print(df)
p = Path(dirName)
fileName = 'predictions.csv' 
test.to_csv(Path(p, fileName))
#--------------------------------------------------------------------------------------------------------------

#-------------------------------------------Testing End--------------------------------------------------------

#-------------------------------------------Calculating Performance Measures-----------------------------------
confusionMatrix = {}
actual = test[test.columns[-2]]
predicted = test['Predicted Class']
truePositive = 0
falseNegative = 0
falsePositive =0
trueNegative = 0

for i in range(len(test['Predicted Class'])):
        if(actual[i]==predicted[i]):
            if( actual[i] == 'positive' ):
                truePositive += 1
            else:
                trueNegative += 1
        else:
            if( actual[i]== 'positive' and predicted[i]== 'negative' ):
                falseNegative += 1
            else:
                falsePositive += 1

confusionMatrix = {'TP':truePositive,'FN':falseNegative,'FP':falsePositive,'TN':trueNegative}
print(confusionMatrix)
Accuracy = ( confusionMatrix.get('TP') + confusionMatrix.get('TN') )/( confusionMatrix.get('TP') + confusionMatrix.get('TN' ) + confusionMatrix.get('FP') + confusionMatrix.get('FN') )
Sensitivity  = ( confusionMatrix.get('TP') )/( confusionMatrix.get('TP') + confusionMatrix.get('FN') )
Specificity = ( confusionMatrix.get('TN') )/( confusionMatrix.get('TN' ) + confusionMatrix.get('FP') )
print("Accuracy: ",Accuracy)
print()
print("Sensitivity: ",Sensitivity)
print()
print("Specificity: ",Specificity)
#--------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------






        


        
#--------------------------------------------------------------------------------------------------------------