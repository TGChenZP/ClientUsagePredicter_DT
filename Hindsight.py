import pandas as pd
import sys

WEEK_TO_INVESTIGATE = int(sys.argv[1])

# Reading in data from the correct weeks (backtracking). The backtracking process can occur because the choices of score/what week's predictor a particular prediction used is well documented in the multiple csv files
predictions = pd.read_csv(f'./History/Week {WEEK_TO_INVESTIGATE-1}/Predictions.csv')

predictor_used = int(predictions["Predictor's week number"][0])

data = pd.read_csv(f'./History/Week {WEEK_TO_INVESTIGATE+1}/Partial_Output/_3_combined_cleaned_data.csv')

stat = pd.read_csv(f'./History/Week {predictor_used}/Statistics.csv')

choice = stat['Statistics'][9]

coef = pd.read_csv(f'./History/Week {predictor_used}/Partial_Output/_4_UsageScoreCoefficients.csv')
S_Quantiles = pd.read_csv(f'./History/Week {predictor_used}/Partial_Output/_7_S_Quantiles.csv')
P_Quantiles = pd.read_csv(f'./History/Week {predictor_used}/Partial_Output/_6_P_Quantiles.csv')

if choice == 1:
    coef_set = list(coef['R_2-actscore'])
    squant_set = list(S_Quantiles['Score1'])
    pquant_set5 = list(P_Quantiles['Q05-1'])
    pquant_set95 = list(P_Quantiles['Q95-1'])
    
elif choice == 2:
    coef_set = list(coef['NMI_actscore'])
    squant_set = list(S_Quantiles['Score2'])
    pquant_set5 = list(P_Quantiles['Q05-2'])
    pquant_set95 = list(P_Quantiles['Q95-2'])
    
else:
    coef_set = list(coef['NMI_actions'])
    squant_set = list(S_Quantiles['Score3'])
    pquant_set5 = list(P_Quantiles['Q05-3'])
    pquant_set95 = list(P_Quantiles['Q95-3'])
    
# All other code pretty similar to Predict.py - please see commenting for those code    
form_record = data['Form_record']
competency = data['Competency']
user_inductee = data['Users_induction']
domain = data['Domain']

usagescore = list()
percentagechange = list()

for i in range(len(form_record)):
    score = coef_set[0] * form_record[i] + coef_set[1] * competency[i] + coef_set[2] * user_inductee[i]
    usagescore.append(score)

    
def perchange(curr, prev):
    """ Helper function for calculating percentages """
    if prev == 0:
        return 0
    else:
        return (curr/prev)-1

for i in range(len(domain)):
    if (i+1 < len(domain)) and domain[i+1] == domain[i]:
        percentagechange.append(perchange(usagescore[i+1], usagescore[i]))
    else: 
        percentagechange.append(-2)

        
stag = list()
ptag = list()
for i in range(len(percentagechange)):
    if usagescore[i] <= squant_set[0]:
        stag.append(1)
    elif usagescore[i] > squant_set[0] and usagescore[i] <= squant_set[1]:
        stag.append(2)
    else:
        stag.append(3)

asset = data['Assets']
actions = data['Actions']
form_template = data['Form_template']
incident = data['Incident']
users = data['Users']
users_normal = data['Users_norm_emp']

for i in range(len(asset)):
    if asset[i] == 0 and actions[i] == 0 and form_template[i] == 0 and incident[i] == 0 and users[i] == 0 and users_normal[i] == 0 and form_record[i] == 0 and competency[i] == 0 and user_inductee[i] == 0:
        stag[i] = 0
        
for i in range(len(percentagechange)):
    if stag[i] == 1:
        if percentagechange[i] <= pquant_set5[0]:
            ptag.append('Decrease')
        elif percentagechange[i] > pquant_set5[0] and percentagechange[i] <= pquant_set95[0]:
            ptag.append('Normal')
        elif percentagechange[i] > pquant_set95[0]:
            ptag.append('Increase')
        
    elif stag[i] == 2:
        if percentagechange[i] <= pquant_set5[1]:
            ptag.append('Decrease')
        elif percentagechange[i] > pquant_set5[1] and percentagechange[i] <= pquant_set95[1]:
            ptag.append('Normal')
        elif percentagechange[i] > pquant_set95[1]:
            ptag.append('Increase')
            
    elif stag[i] == 3:
        if percentagechange[i] <= pquant_set5[2]:
            ptag.append('Decrease')
        elif percentagechange[i] > pquant_set5[2] and percentagechange[i] <= pquant_set95[2]:
            ptag.append('Normal')
        elif percentagechange[i] > pquant_set95[2]:
            ptag.append('Increase')
    
    else:  
        ptag.append('N/A')
        
        
week = data['Week']

indexforthisweek = list()

for i in range(len(week)):
    if week[i] == WEEK_TO_INVESTIGATE:
        indexforthisweek.append(i)

actualobservations = list()
for i in indexforthisweek:
    actualobservations.append(ptag[i])


predictions.insert(4, 'ActualObs', actualobservations, True)

predictions.to_csv(f'./History/Week {WEEK_TO_INVESTIGATE-1}/Retrospective.csv', index = False)