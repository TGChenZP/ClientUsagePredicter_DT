### CLIENT USAGE PREDICTOR - initial training script
### Code produced by Lang (Ron) Chen August-October 2021 for Lucidity Software
""" Wrangles initial raw data and outputs predictor objects for predicting the client usage trend for the upcoming week """

# Input: 
#    Argument 1: a cut-off date for data; logically should be a Sunday. Must be in form ‘[d]d/[m]m/yyyy’ (If day or month single digit can input just as single digit). 
#         please ensure date is valid and is after the FIRSTDATE (by default July 3rd 2017)

#     Initial raw data files need to be stored in directory ‘./Data’. 
#     -File names must be ‘action.csv’, ‘assets.csv’, ‘attachment.csv’, ‘competency_record.csv’, ‘form_record.csv’, ‘form_template.csv’, ‘incident.csv’, ‘users.csv’, associated with the data of the corresponding filename
#     -Each csv must include columns that include ‘domain’ for company name, and ‘created_at’. ‘users.csv’ must also have column name ‘hr_type’, with data values ‘Casual’, ‘Employee’, ‘Subcontractor’ denoting regular employees and ‘InductionUser’ denoting induction users.
#     -There should be no other tester domains in the data apart from ‘demo’, ‘demo_2’ and ‘cruse’

#     -the dates for 'action.csv', 'competency_record.csv', 'form_record.csv', 'incident.csv', 'users.csv' should be in form of [d]d/[m]m/yyyy
#     -the dates for 'assets.csv', 'form_template.csv' should be in form of yyyy-mm-dd.
#     *if the form of these are different then need to edit PART 2 in the script. 


# Output: several partial outputs - partial outputs of wrangled data and important coefficients, tree objects (.dot), DecisionTreePredictor objects (.pickle), Statistics.csv, exported to various directories including the home directory (relatively '..'), the History directory (into the relevent week) and the current directory 





# PART 0: IMPORTING LIBRARIES

import sys
import os

import pandas as pd
import numpy as np

import math as m

from sklearn import linear_model
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz 
from sklearn.model_selection import KFold

import pickle





# PART 1: CREATE DATE FILE

# Reads in 'cut-off date' for data to be used to train 
currdate = sys.argv[1]

# Assumes that date format in 'dd/mm/yyyy' format
cutoffday = int(currdate.split('/')[0])
cutoffmonth = int(currdate.split('/')[1])
cutoffyear = int(currdate.split('/')[2])

def datejoin(day, month, year):
    """ For joining up three numbers into a date"""
    return (f'{str(day)}/{str(month)}/{str(year)}')


def leapyear(year):
    """ For determining whether a year is a leap year"""
    if year % 4 == 0:
        if year% 100 == 0:
            if year%400 == 0:
                return True
            else:
                return False
        else:
            return True
        
    else:
        return False


# Creates a dictionary matching each day to a week number (counting Week of July 3rd 2021 as Week 1)    
#### FUTURE CHANGE: if wish to include data earlier than Monday July 3rd 2017, change the magic string FIRSTDATE
FIRSTDATE = '03/07/2017'
firstdateday = int(FIRSTDATE.split('/')[0])
firstdatemonth = int(FIRSTDATE.split('/')[1])
firstdateyear = int(FIRSTDATE.split('/')[2])

days = [29, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]    
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
years = range(2017, cutoffyear+1)


datematchweek = dict()
week = 1
count = 0
for year in years:
    for month in months:
        
        if (year == firstdateyear and month < firstdatemonth) or (year == cutoffyear and month > cutoffmonth):
            continue
        
        if month == 2 and leapyear(year):
            indexmonth = 0
            
        else:
            indexmonth = month
        
        for day in range(1, days[indexmonth]+1):
            if (year == firstdateyear and month == firstdatemonth and day < firstdateday) or (year == cutoffyear and month == cutoffmonth and day > cutoffday):
                continue
            
            count += 1
            
            if count == 8:
                count = 1
                week += 1
            
            date = datejoin(day, month, year)
            
            datematchweek[date] = week

dates = list(datematchweek.keys())
weekno = list(datematchweek.values())

# Make the dictionary of dates to week number into a dataframe
DatesToWeek_DF = pd.DataFrame({'dates': dates, 'weekno':weekno})

DatesToWeek_DF.to_csv('../DateMatchWeek.csv')

# Record the week number of the cut-off date
thisweek = max(weekno)





# # PART 2: WRANGLING ORIGINAL DATA

#### FUTURE UPDATE: towranglelist1 stores records which have dates in format [d]d/[m]m/yyyy; towranglelist2 stores records which have dates in format yyyy-mm-dd. Need to update lists accordingly
towranglelist1 = ['action.csv', 'competency_record.csv', 'form_record.csv', 'incident.csv', 'users.csv']
towranglelist2 = ['assets.csv', 'form_template.csv']
towranglelist3 = ['users.csv'] 
towranglelist4 = ['users.csv']
# because users need to be wrangled in terms of employees and inductioneers. 3 is inductionuser 4 is employees

def wrangle(filename, datedata, mode):
    """ cleans the file. 4 modes for four different ways to clean the data - all pretty similar except mode 3 and 4 selects users of particular hr types, and mode 2 deals with dates of a different format """
    data = pd.read_csv(f"./Data/{filename}")
    domain = list(data['domain'])
    
    # First drop: get rid of rows from domains demo and demo_2
    if mode in [1,2]:
        droplist = []
        for i in range(len(domain)):
            if domain[i] in ['demo', 'demo_2', 'cruse']:
                droplist.append(i)
                
    elif mode == 3:
        domain = list(data['domain'])
        hr = list(data['hr_type'])
        
        req = ['InductionUser']
        
        droplist = []
        for i in range(len(domain)):
            if domain[i] in ['demo', 'demo_2', 'cruse'] or hr[i] not in req:
                droplist.append(i)
    
    else:
        domain = list(data['domain'])
        hr = list(data['hr_type'])
        
        req = ['Casual', 'Employee', 'Subcontractor']
        
        droplist = []
        for i in range(len(domain)):
            if domain[i] in ['demo', 'demo_2', 'cruse'] or hr[i] not in req:
                droplist.append(i)

    data = data.drop(droplist)
    
    # re-setup date dictionary from the DataFrame
    dates = list(datedata['dates'])
    weekno = list(datedata['weekno'])
    datematchdict = dict()
    for i in range(len(dates)):
        datematchdict[dates[i]] = weekno[i]
    
    # Second drop: clean out rows whose dates are not within startdate and cutoffdate
    #### FUTURE CHANGE: this step takes quite a lot of time - could be area to improve algorithmically
    data.index = (range(0, len(list(data['created_at'])))) #re-do index after dropping demo and demo_2
    actdate = list(data['created_at'])
    
    # If any data happens to have date in format "dd-mm-yyyy" then need to put file in towranglelist2. Else put in towranglelist1. Note dates should be in format "[d]d/[m]m/yyyy"
    newdroplist = []
    if mode == 2:
        def transform_date(inputdate):
            """ helper function to transform date in format of dd-mm-yyyy into [d]d/[m]m/yyyy which is what datedata produced in PART 1 stores  """
            splitted = inputdate.split('-')
            if int(splitted[1]) < 10:
                month = splitted[1][1]
            else:
                month = splitted[1]

            if int(splitted[2]) < 10:
                day = splitted[2][1]
            else:
                day = splitted[2]

            return f'{day}/{month}/{splitted[0]}'
        
        for i in range(len(actdate)):
            if transform_date(actdate[i].split()[0]) not in dates:
                newdroplist.append(i)
        
    else:
        for i in range(len(actdate)):
            if actdate[i].split()[0] not in dates:
                newdroplist.append(i)
        
    data = data.drop(newdroplist) # drop the rows of data whose dates are not between startdate and cutoffdate
    
    actdate = list(data['created_at']) #reread the date created column now that we've dropped some rows
    
    newdomain = list(data['domain'])
    # get a new list matching each action to the week that they were done in
    actweekno = list()
    
    if mode == 2:
        for i in range(len(actdate)):
            actweekno.append(datematchdict[transform_date(actdate[i].split()[0])])
    else:
        for i in range(len(actdate)):
            actweekno.append(datematchdict[actdate[i].split()[0]]) # use [0] because string also contains hour:minute:second
    
    # At this point, now have two lists newdomain and actweekno: in the former the ith value is the domain of the ith row, and the latter the ith value is the relative week since FIRSTSTARTDATE that the ith row was created in. Now just count them up 
    
    # count up the numbers of actions this week by domain and week
    groupup = dict()
    for i in range(len(actweekno)):
        if f'{newdomain[i]} {actweekno[i]}' in groupup:
            groupup[f'{newdomain[i]} {actweekno[i]}'] += 1
        else:
            groupup[f'{newdomain[i]} {actweekno[i]}'] = 1
            
    groupupkey = list(groupup.keys())
    groupupval = list(groupup.values())

    # create lists that contain just domain name and week number
    out1 = list()
    out2 = list()

    for i in range(len(groupupkey)):
        out1.append(groupupkey[i].split()[0])
        out2.append(groupupkey[i].split()[1])
    
    # export the wrangled file as a csv (each of these files are wrangled version of the raw data files (of each of the client's recorded activity in lucidity) in terms of counts per week per domain)
    out = pd.DataFrame({'Domain': out1, 'Week': out2, 'COUNT': groupupval})
    
    if mode in [1,2]:
        out.to_csv(f'./Partial_Output/_2_{filename.split(".")[0]}_clean.csv', index = False)
        out.to_csv(f'../History/Week {thisweek}/Partial_Output/_2_{filename.split(".")[0]}_clean.csv', index = False)
        
    elif mode == 3:
        out.to_csv('./Partial_Output/_2_Users_Inductee_clean.csv', index = False)
        out.to_csv(f'../History/Week {thisweek}/Partial_Output/_2_{filename.split(".")[0]}_clean.csv', index = False)
        
    else:
        out.to_csv('./Partial_Output/_2_Users_norm_employee_clean.csv', index = False)
        out.to_csv(f'../History/Week {thisweek}/Partial_Output/_2_{filename.split(".")[0]}_clean.csv', index = False)

# OS housekeeping and running each of the files through wrangle()
if not os.path.exists('./Partial_Output'):
    os.mkdir('./Partial_Output')

if not os.path.exists(f'../History/Week {thisweek}/Partial_Output'):
    os.makedirs(f'../History/Week {thisweek}/Partial_Output')
        
for file in towranglelist1:
    wrangle(file, DatesToWeek_DF, 1)
    
for file in towranglelist2:
    wrangle(file, DatesToWeek_DF, 2)
    
for file in towranglelist3:
    wrangle(file, DatesToWeek_DF, 3)
    
for file in towranglelist4:
    wrangle(file, DatesToWeek_DF, 4)

    

    

# PART 3: COMBINE PREVIOUSLY WRANGLED DATAFRAMES INTO ONE (FILLING IN WEEKS WITH NO ACTIVITY)

# import all cleaned data
asset = pd.read_csv('./Partial_Output/_2_assets_clean.csv')
actions = pd.read_csv('./Partial_Output/_2_action_clean.csv')
competency = pd.read_csv('./Partial_Output/_2_competency_record_clean.csv')
form_record = pd.read_csv('./Partial_Output/_2_form_record_clean.csv')
form_templates = pd.read_csv('./Partial_Output/_2_form_template_clean.csv')
incidents = pd.read_csv('./Partial_Output/_2_incident_clean.csv')
users = pd.read_csv('./Partial_Output/_2_users_clean.csv')
users_induct = pd.read_csv('./Partial_Output/_2_users_Inductee_clean.csv')
users_norm_emp = pd.read_csv('./Partial_Output/_2_users_norm_employee_clean.csv')

# Find a set of the domain names - for finding the "earliest recorded date" of activity/usage for each
set1 = set(asset['Domain'])
set2 = set(actions['Domain'])
set3 = set(competency['Domain'])
set4 = set(form_record['Domain'])
set5 = set(form_templates['Domain'])
set6 = set(incidents['Domain'])
set7 = set(users['Domain'])

fullset = set1.union(set2).union(set3).union(set4).union(set5).union(set6).union(set7)
fullsetlist = list(fullset) # Now have a full set of the domains
fullsetlist.sort()

iteration = [asset, actions, competency, form_record, form_templates, incidents, users]
newiteration = [asset, actions, competency, form_record, form_templates, incidents, users, users_induct, users_norm_emp]

# Find first week recorded and put them in a dictionary of (key: value) = (domain name: first week of activity)
startweek = dict()

for data in newiteration:
    dom = list(data['Domain'])
    week = list(data['Week'])
    count = list(data['COUNT'])
    
    for i in range(len(dom)):
        if dom[i] in startweek:
            if week[i] < startweek[dom[i]]:
                startweek[dom[i]] = week[i]
        else:
            startweek[dom[i]] = week[i]
            
startweeklist = list(startweek.items())
startweeklist.sort()

# Create a template for recording the data (now fill out gaps between start week and week 216 where there is 0 data)
combineddatatemplate = dict()

# first initiate a blank dictionary with all weeks from first week of activity to cutoffdate's week
for i in range(len(startweeklist)):
    for j in range(startweeklist[i][1], thisweek+1):
        combineddatatemplate[f'{startweeklist[i][0]} {j}'] = 0

# create blank copies of this initialised template dictionary, and fill them in based on counts from the output of PART 2
assetcomb = combineddatatemplate.copy()
actionscomb = combineddatatemplate.copy()
competencycomb = combineddatatemplate.copy()
form_recordcomb = combineddatatemplate.copy()
form_templatescomb = combineddatatemplate.copy()
incidentscomb = combineddatatemplate.copy()
userscomb = combineddatatemplate.copy()
users_inductcomb = combineddatatemplate.copy()
users_norm_empcomb = combineddatatemplate.copy()

dictlist = [assetcomb, actionscomb, competencycomb, form_recordcomb, form_templatescomb, incidentscomb, userscomb, users_inductcomb, users_norm_empcomb]

# Now fill in the details where there are records (because all dictionary slots initialised, only need to repalce data for weeks where there was a count recorded, and all other are fine to be left untouched - just ends up being 0)
for k in range(len(dictlist)):
    dom = list(newiteration[k]['Domain'])
    week = list(newiteration[k]['Week'])
    count = list(newiteration[k]['COUNT'])
    
    for i in range(len(dom)):
        dictlist[k][f'{dom[i]} {week[i]}'] = count[i]

uniqueid = list(assetcomb.keys())
assetcount = list(assetcomb.values())
actioncount = list(actionscomb.values())
competencycount = list(competencycomb.values())
form_recordcount = list(form_recordcomb.values())
form_templatescount = list(form_templatescomb.values())
incidentscount = list(incidentscomb.values())
userscount = list(userscomb.values())
users_inductcount = list(users_inductcomb.values())
users_norm_empcount = list(users_norm_empcomb.values())

# create two more lists that contain just domain and just week - maximises chances of making future wrangling easier
doms = []
weekss = []

for i in range(len(uniqueid)):
    doms.append(uniqueid[i].split()[0])
    weekss.append(uniqueid[i].split()[1])
    
# Create a new column for counting the number of weeks since particular company started at Lucidity
selfweeks = []
count = 0

# (logic of loop fairly simple - if domain column runs into new company then reset the count)
prev = doms[0]
for i in range(len(doms)):
    if doms[i] == prev:
        count += 1
        selfweeks.append(count)
    else:
        count = 1
        selfweeks.append(count)
    prev = doms[i]

# Create a prelim score of sum of the three attributes to be used in determining a function
baseactscore1 = list()
for i in range(len(uniqueid)):
    baseactscore1.append(form_recordcount[i]+competencycount[i]+users_inductcount[i])

# turn it into one dataframe and output
out = pd.DataFrame({'ID': uniqueid, 'Domain': doms, 'Week': weekss, 'Selfweeks': selfweeks,
                    'Assets': assetcount, 'Actions': actioncount, 'Competency': competencycount, 
                    'Form_record': form_recordcount, 'Form_template': form_templatescount,
                   'Incident': incidentscount, 'Users': userscount, 
                    'Users_induction': users_inductcount, 'Users_norm_emp': users_norm_empcount, 
                    "Prelim_action_score": baseactscore1})

# This file now has counts of all activities grouped by week by client/domain, sorted by domian and clients.  
out.to_csv("./Partial_Output/_3_combined_cleaned_data.csv", index = False)
out.to_csv(f"../History/Week {thisweek}/Partial_Output/_3_combined_cleaned_data.csv", index = False)





# PART 4: FIND COEFFICIENTS OF USAGE SCORE FUNCTION

# reread data
data = pd.read_csv('./Partial_Output/_3_combined_cleaned_data.csv')

form_rec = list(data['Form_record'])
compet = list(data['Competency'])
uinduct = list(data['Users_induction'])
prelim_actionscore = list(data['Prelim_action_score'])
actions = list(data['Actions'])

# prepare data for working with the prelim_actionscore
nform_rec = list()
ncompet = list()
nuinduct = list()
nprelim_actionscore = list()

# ignore any row where all three attributes are 0 (so not to distort the data)
# (while this seems complicated - it is useful for the integrity of the prediction later on that we filled in any gap weeks where there were no activity at all. but for this stage we must not include these rows)
for i in range(len(form_rec)):
    if prelim_actionscore[i] != 0:
        nform_rec.append(form_rec[i])
        ncompet.append(compet[i])
        nuinduct.append(uinduct[i])
        nprelim_actionscore.append(prelim_actionscore[i])
    
# prepare data for working with actions
mform_rec = list()
mcompet = list()
muinduct = list()
mactions = list()

for i in range(len(form_rec)):
    if form_rec[i] != 0 or compet[i] != 0 or uinduct[i] != 0 or actions[i] != 0:
        mform_rec.append(form_rec[i])
        mcompet.append(compet[i])
        muinduct.append(uinduct[i])
        mactions.append(actions[i])
        
# Finding Coefficients By Regression
# Regression - Prelim Score
nform_reca = np.array(nform_rec)
nform_reca = nform_reca.reshape(-1,1)

ncompeta = np.array(ncompet)
ncompeta = ncompeta.reshape(-1,1)

nuinducta = np.array(nuinduct)
nuinducta = nuinducta.reshape(-1,1)

nprelim_actionscorea = np.array(nprelim_actionscore)
nprelim_actionscorea = nprelim_actionscorea.reshape(-1,1)

lm1 = linear_model.LinearRegression()
model = lm1.fit(nform_reca, nprelim_actionscorea)
r2_test_lm1 = lm1.score(nform_reca, nprelim_actionscorea)

lm2 = linear_model.LinearRegression()
model = lm2.fit(ncompeta, nprelim_actionscorea)
r2_test_lm2 = lm2.score(ncompeta, nprelim_actionscorea)

lm3 = linear_model.LinearRegression()
model = lm3.fit(nuinducta, nprelim_actionscorea)
r2_test_lm3 = lm3.score(ncompeta, nprelim_actionscorea)

# Regression - Actions
mform_reca = np.array(mform_rec)
mform_reca = mform_reca.reshape(-1,1)

mcompeta = np.array(mcompet)
mcompeta = mcompeta.reshape(-1,1)

muinducta = np.array(muinduct)
muinducta = muinducta.reshape(-1,1)

mactionsa = np.array(mactions)
mactionsa = mactionsa.reshape(-1,1)

lm4 = linear_model.LinearRegression()
model = lm4.fit(mform_reca, mactionsa)
r2_test_lm4 = lm4.score(mform_reca, mactionsa)

lm5 = linear_model.LinearRegression()
model = lm5.fit(mcompeta, mactionsa)
r2_test_lm5 = lm5.score(mcompeta, mactionsa)

lm6 = linear_model.LinearRegression()
model = lm6.fit(muinducta, mactionsa)
r2_test_lm6 = lm6.score(mcompeta, mactionsa)

# Finding Coefficients By NMI
# NMI - Prelim score
nmi1 = normalized_mutual_info_score(nform_rec, nprelim_actionscore)

nmi2 = normalized_mutual_info_score(ncompet, nprelim_actionscore)

nmi3 = normalized_mutual_info_score(nuinduct, nprelim_actionscore)

# NMI
nmi4 = normalized_mutual_info_score(mform_rec, mactions)

nmi5 = normalized_mutual_info_score(mcompet, mactions)

nmi6 = normalized_mutual_info_score(muinduct, mactions)

# Output into a file
list1 = [r2_test_lm1, r2_test_lm2, r2_test_lm3]
list2 = [r2_test_lm4, r2_test_lm5, r2_test_lm6]
list3 = [nmi1, nmi2, nmi3]
list4 = [nmi4, nmi5, nmi6]
index = ['Form_record', 'Competency', "User_inductee"]

#### FUTURE CHANGE: leave out list 2 (regression actions) because it has a negative - cannot see a change back to positive coefficient in the near future so left out. 
out = pd.DataFrame({"R_2-actscore":list1, "NMI_actscore":list3, "NMI_actions":list4}, index = index)

# This file stores the coefficients for the usage scores (all three methods)
out.to_csv("./Partial_Output/_4_UsageScoreCoefficients.csv")
out.to_csv(f"../History/Week {thisweek}/Partial_Output/_4_UsageScoreCoefficients.csv")





# PART 5: CREATE USAGE SCORES

# reread data
data = pd.read_csv('./Partial_Output/_3_combined_cleaned_data.csv')
coef = pd.read_csv('./Partial_Output/_4_UsageScoreCoefficients.csv')

r2_own = list()
nmi_own = list()
nmi_actions = list()

r2_owncoef = list(coef['R_2-actscore'])
nmi_owncoef = list(coef['NMI_actscore'])
nmi_actionscoef = list(coef['NMI_actions'])

form_rec = list(data['Form_record'])
compet = list(data['Competency'])
uinduct = list(data['Users_induction'])

# for each row calculate the three different scores using the three different sets of coefficients and store them in three different lists (later columns)
for i in range(len(form_rec)):
    x = form_rec[i]
    y = compet[i]
    z = uinduct[i]
    
    score1 = r2_owncoef[0]*x + r2_owncoef[1]*y + r2_owncoef[2]*z
    score2 = nmi_owncoef[0]*x + nmi_owncoef[1]*y + nmi_owncoef[2]*z
    score3 = nmi_actionscoef[0]*x + nmi_actionscoef[1]*y + nmi_actionscoef[2]*z
    
    r2_own.append(score1)
    nmi_own.append(score2)
    nmi_actions.append(score3)

# Add the columns into the data frame
data.insert(14, "Score1", r2_own, True)
data.insert(15, "Score2", nmi_own, True)
data.insert(16, "Score3", nmi_actions, True)

# Now also work on returning adding another column for each score: the % change in the next week - this is an important variable for our predicting work (it is the preliminary label for each column)
percentchange1 = list()
percentchange2 = list()
percentchange3 = list()
domain = data['Domain']

def perchange(curr, prev):
    """ Helper function for calculating percentage change """
    if prev == 0:
        return 0
    else:
        return (curr/prev)-1

# ** Note used logic of appending the percentage increase of next week into current week as the preliminary 'y-value'/'label'/'classification' of the rows - otherwise we won't be predicting - rather recounting the activity of the week gone by
for i in range(len(domain)):
    if (i+1 < len(domain)) and domain[i+1] == domain[i]:
        percentchange1.append(perchange(r2_own[i+1], r2_own[i]))
        percentchange2.append(perchange(nmi_own[i+1], nmi_own[i]))
        percentchange3.append(perchange(nmi_actions[i+1], nmi_actions[i]))
    else: 
        percentchange1.append(-2)
        percentchange2.append(-2)
        percentchange3.append(-2)
        # here used -2 becuase its an impossible score - this will serve as a 'tag' to remove rows we don't want to use for test-train out later

# Add the columns into the data frame
data.insert(15, "%Change Score 1", percentchange1, True)
data.insert(17, "%Change Score 2", percentchange2, True)
data.insert(19, "%Change Score 3", percentchange3, True)





# PART 6: DROP OFF TAIL ZERO

# Remake each column of data DataFrame into a list
d0 = list(data['ID'])
d1 = list(data['Domain'])
d2 = list(data['Week'])
d3 = list(data['Selfweeks'])
a1 = list(data['Assets'])
a2 = list(data['Actions'])
a3 = list(data['Competency'])
a4 = list(data['Form_record'])
a5 = list(data['Form_template'])
a6 = list(data['Incident'])
a7 = list(data['Users']) 
a8 = list(data['Users_induction'])
a9 = list(data['Users_norm_emp'])
s1 = list(data['Score1'])
s2 = list(data['Score2'])
s3 = list(data['Score3'])
p1 = list(data['%Change Score 1'])
p2 = list(data['%Change Score 2'])
p3 = list(data['%Change Score 3'])

# Initialise a list for marking rows which are at the back end of a company's recorded weeks but all of the last weeks are 0 (from now on referred tail zero weeks)
tailzero = list()
for i in range(len(d0)):
    tailzero.append(0)
    
# Create this list (**Note this is the same for all three scores as it only depends on the original attributes, not the scores. or in other words if one row has no activity then all 3 scores would be zero so no need for three columns to mark tailzeros)
prev = None
switch = True # means is not currently in a streak of zeros
for i in range(len(d0)):
    if d1[i] != prev:
        if switch == False:
            for j in range(currindex, i):
                tailzero[j] = 1
            
        switch = True
        
    if a1[i] == 0 and a2[i] == 0 and a3[i] == 0 and a4[i] == 0 and a5[i] == 0 and a6[i] == 0 and a7[i] == 0:
        if switch == True:
            currindex = i # represents the first week of the tail-zeros
            switch = False # indicates it is in a streak of tail-zero
    
    else:
        switch = True
    
    prev = d1[i]    

# Need to do one more run for the last domain.
if switch == False:
    for j in range(currindex, i):
        tailzero[j] = 1

# Add the columns into the data frame
data.insert(20, "Tailzero", tailzero, True)





# PART 7: WIPES OFF ANY WEEKS IN THE FIRST 26 WEEKS OF A COMPANY'S USAGE THAT ARE CONSIDERED EARLY FLUCTUATIONS

# Method used: if a company's usage score in a particular week is lower than its own overall 25 percentile but its associated percentage change (same row) is higher than the all-company 75 percentile and this week is in the first 26 weeks, or if this company hasn't been with Lucidity for 26 weeks yet, then it and all previous weeks are wiped off (not actually deleted but marked by a new column) 

# Remake newest column from Data DataFrame as a list
t = list(data['Tailzero'])

def excl_start(p, s, t, d1):
    """ Helper function for creating the columns that mark dropping the start """
    # first get relevent columns (don't intake any rows that are marked to be 'wiped off' - in this case the tail and any with percentage change -2)
    p_no_tail = list()
    for i in range(len(d1)):
        if t[i] == 0 and p[i] != -2:
            p_no_tail.append(p[i])

    # work out the overall q3 for percentage rise
    p_no_tail_array = np.array(p_no_tail)
    p_q3 = np.quantile(p_no_tail_array, .75)

    # tag any row where %change is over the 75% for overall, and its score is less than 25% for its own usage score (or hasn't been with company for 26 weeks)
    droplist = list()
    for i in range(len(d1)):
        droplist.append(0)

    tmp = list()
    prev = d1[0]
    startindex = 0
    for i in range(len(d1)):
        if d1[i] != prev:
            # if company hasn't been with Lucidity for 26 weeks then all data should be excluded
            if i - startindex + 1 > 26:
                count = startindex-1
                tmp_array = np.array(tmp)
                owns_q1 = np.quantile(tmp_array, .25)

                for j in range(startindex, startindex+26):
                    if p[j] > p_q3 and s[j] < owns_q1:
                        count = j
            
            else:     
                count = i
            
            for k in range(startindex, count+1):
                droplist[k] = 1

            tmp = list()
            startindex = i

        else:
            if t[i] == 0:
                tmp.append(s[i])


        prev = d1[i]
    
    if i - startindex + 1 > 26:
        count = startindex-1
        tmp_array = np.array(tmp)
        owns_q1 = np.quantile(tmp_array, .25)

        for j in range(startindex, startindex+26):
            if p[j] > p_q3 and s[j] < owns_q1:
                count = j
    else:     
        count = i
        
    for k in range(startindex, count+1):
        droplist[k] = 1

    # theory of the algorithm is: first put each company's score into a list, then use it to caluclate the 25 percentile, then run through the company's first 26 weeks' data, and if any weeks match the criterion for wipeoff then all previous weeks needs to be wiped off too
    # because of structure of code need to do one more loop at the end
    
    return droplist


# Add the columns into the data frame
data.insert(21, "Drop_start1", excl_start(p1, s1, t, d1), True)

data.insert(22, "Drop_start2", excl_start(p2, s2, t, d1), True)

data.insert(23, "Drop_start3", excl_start(p3, s3, t, d1), True)





# PART 8: CREATE SECONDARY ATTRIBUTES (HORIZONTAL QUANTILES, VERTICAL QUANTILES AND TRENDS) FOR SCORES

# Remake newest columns from Data DataFrame as a list
drop1 = list(data['Drop_start1'])
drop2 = list(data['Drop_start2'])
drop3 = list(data['Drop_start3'])


def hor(p, drop, t, d1):
    """ Helper function for calculating horizontal comparison of quantile: where does a company rank in terms of activity scores in all of its history? (only up to 'that particular week', not after. e.g. have 30 weeks of data but horizontal quantile at week 15 only considers its quantile out of the data in the first 15 weeks, assuming none of the first 15 weeks are wiped off) """
    
    horizontal = list()
    for i in range(len(d1)):
        horizontal.append(2)
        # append an impossible value for quantile - easier to wipe off later on (in fact not used - but still good to have it to be attached to rows that should be wiped off)

    prev = d1[0]
    tmp = list()
    for i in range(len(d1)):
        if d1[i] != prev:
            tmp = list()


        if t[i] == 0 and drop[i] == 0:
            tmp.append(p[i])

            tmp.sort()

            # for labelling data with average in case of tie

            tmp2 = list()
            for j in range(len(tmp)):
                if tmp[j] == p[i]:
                    tmp2.append(j)
            mean = sum(tmp2)/len(tmp2)
            horizontal[i] = mean/len(tmp)

        prev = d1[i]
    
    return horizontal
    

def vert(p, drop, t, d0, d1, d3):
    """ Helper function for calculating vertical comparison of quantile: where does a company rank in terms of activity scores in all companies in this week? (doesn't include companies whose data is 'wiped off' that week) """
    
    # first create a list for each of the weeks
    listoflist = list()
    for i in range(max(d3)):
        tmp = list()
        listoflist.append(tmp)


    # add in data of relevent weeks
    for i in range(len(d0)):
        if t[i] == 0 and drop[i] == 0:
            listoflist[d3[i]-1].append(p[i])

    # sort each of the lists of data now grouped in weeks
    for i in range(len(listoflist)):
        tmp = listoflist[i]
        tmp.sort()
        listoflist[i] = tmp

    vertical = list()
    for i in range(len(d0)):
        vertical.append(2)
        # append an impossible value for quantile - easier to wipe off later on

    for i in range(len(d1)):
         if t[i] == 0 and drop[i] == 0:

            tmp2 = list()
            for j in range(len(listoflist[d3[i]-1])):
                if listoflist[d3[i]-1][j] == p[i]:
                    tmp2.append(j)
            mean = sum(tmp2)/len(tmp2)
            vertical[i] = mean/len(listoflist[d3[i]-1])
    
    return vertical


def trendd(p, drop, t, d0, d1):
    """ Helper function for calculating trend the company is undergoing: how many weeks in a row has a company been increasing/decreasing? Positive numbers denote number of weeks in a row of increase, Negative numbers denote number of weeks in a row of decrease. If consistently 0% change then recorded as 0, if change of trend from increase to decrease then that week is marked as 0 """
    
    #### FUTURE CHANGE: currently the first week after a change from inc to dec or dec to inc is marked as 0. While this is intuitive, when the streaks are up and running the data actually then represent value+1 weeks in a row of increase. e.g. two weeks in a row of increase but second week's trend value only has value '1'. This is unlikely to make substantial difference but still worth a shot
    #### FUTURE CHANGE: upon review, it was realised that the percentage associated with each row is actually the % change of next week. It perhaps may make sense to calculate the scores based on p[i-1] instead. May be worth an attempt but regardless it is really just a secondary attribute so how much influence it has is doubtful. It may also be worth it to study the graphviz output and see how high near the root this attribute sits to determine whether it is worthwhile to attempt a change
    
    trend = list()
    for i in range(len(d0)):
        trend.append(0)
    
    prev = d1[i]
    
    for i in range(len(d1)):
        if d1[i] != prev:
            continue
        elif t[i] == 0 and drop[i] == 0:
            if trend[i-1] == 0:
                if p[i] > 0:
                    trend[i] = trend[i-1] + 1
                elif p[i] < 0:
                    trend[i] = trend[i-1] - 1
                else:
                    trend[i] = 0

            if trend[i-1] > 0:
                if p[i] > 0:
                    trend[i] = trend[i-1] + 1
                else:
                    trend[i] = 0

            if trend[i-1] < 0:
                if p[i] < 0:
                    trend[i] = trend[i-1] - 1
                else:
                    trend[i] = 0

        prev = d1[i]
        
    return trend


# Add the columns into the data frame
data.insert(14, "Horizontal quantile 1", hor(s1, drop1, t, d1), True)
data.insert(15, "Vertical quantile 1", vert(s1, drop1, t, d0, d1, d2), True)
data.insert(16, "Trend 1", trendd(p1, drop1, t, d0, d1), True)

data.insert(17, "Horizontal quantile 2", hor(s2, drop2, t, d1), True)
data.insert(18, "Vertical quantile 2", vert(s2, drop2, t, d0, d1, d2), True)
data.insert(19, "Trend 2", trendd(p2, drop2, t, d0, d1), True)

data.insert(20, "Horizontal quantile 3", hor(s3, drop3, t, d1), True)
data.insert(21, "Vertical quantile 3", vert(s3, drop3, t, d0, d1, d2), True)
data.insert(22, "Trend 3", trendd(p3, drop3, t, d0, d1), True)





# PART 9: FIND DISTRIBUTION AND TAG 

# an extra list here to exclude any rows where there were no activity for a particular company in a particular week. #### Future Update: this was a last minute addition so this column was not inserted into the data DF. When adding it to the DF later, beware of the indexes for pd.Dataframe.insert() later on in the code
allzero = list()

for i in range(len(d0)):
    if a1[i] == 0 and a2[i] == 0 and a3[i] == 0 and a4[i] == 0 and a5[i] == 0 and a6[i] == 0 and a7[i] == 0:
        allzero.append(1)
        if d1[i-1] == d1[i]:
            allzero[i-1] = 1
    else:
        allzero.append(0)

        
def distb(p, s, drop, t, d0, d1, allzero):
    """ Helper function to determine the 5% and 95% quantile for the bottom 25%, middle 50% and upper 25% of the three scores - this is done as later on there will be three predictors made for each Usage Score calculation method - one for the bottom 25%, one for the upper 25% and one for the middle 50% in terms of scores. This is done as likely the percentage change for lower scores would have much more varience than the upper 25%. So different predictors need to be fitted to account for different levels of sensitivity. the 5% and 95% quantile is done so that 'actual labels' can be applied to the data - calling them 'decrease', 'normal' and 'increase'. the 5% and 95% are chosen so to approximately get 10% of companies flagged each week """
    
    # first split the data into three groups in terms of their scores. end up with bottom 25%, middle 50% and top 25%
    news = list()
    newp = list()
    for i in range(len(d1)):
        if t[i] == 0 and drop[i] == 0 and p[i] != -2 and allzero[i] == 0:
            news.append(s[i])
            newp.append(p[i])

    newsarray = np.array(news)

    s_q1 = np.quantile(newsarray, .25)
    s_q3 = np.quantile(newsarray, .75)

    tag = list()
    for i in range(len(d0)):
        tag.append(0)
        # here 0 is the impossible value 
    
    # label the data top 25% score (3), bot 25% score (1) or middle 50% score (2) by adding an extra column to the data
    for i in range(len(d0)):
        if t[i] == 0 and drop[1] == 0 and p[i] != -2 and allzero[i] == 0:
            if s[i] <= s_q1:
                tag[i] = 1
            elif s_q1 < s[i] and s[i] <= s_q3:
                tag[i] = 2
            else:
                tag[i] = 3
    
    # now find the 5 percentile and 95 percentile of percentage change as the cutoff values for labelling the data as increase, decrease or normal
    bots = list()
    botp = list()
    mids = list()
    midp = list()
    tops = list()
    topp = list()

    for i in range(len(news)):
        if news[i] <= s_q1:
            bots.append(news[i])
            botp.append(newp[i])
        elif s_q1 < news[i] and news[i] <= s_q3:
            mids.append(news[i])
            midp.append(newp[i])
        else:
            tops.append(news[i])
            topp.append(newp[i])

    botp_array = np.array(botp)
    midp_array = np.array(midp)
    topp_array = np.array(topp)

    botp_q1 = np.quantile(botp_array, .05)
    botp_q3 = np.quantile(botp_array, .95)
    midp_q1 = np.quantile(midp_array, .05)
    midp_q3 = np.quantile(midp_array, .95)
    topp_q1 = np.quantile(topp_array, .05)
    topp_q3 = np.quantile(topp_array, .95)
    
    q1s = [botp_q1, midp_q1, topp_q1]
    q3s = [botp_q3, midp_q3, topp_q3]
    
    return tag, q1s, q3s, s_q1, s_q3

## FUTURE CHANGE: If one day the files are so large that the time required for this function is large, then consider splitting the distb functino into 2 functions (but then need to re-input the variables)

# Add the columns into the data frame
data.insert(33, "Tag1", distb(p1, s1, drop1, t, d0, d1, allzero)[0], True)
data.insert(34, "Tag2", distb(p2, s2, drop2, t, d0, d1, allzero)[0], True)
data.insert(35, "Tag3", distb(p3, s3, drop3, t, d0, d1, allzero)[0], True)
data.insert(36, "allzero", allzero, True)

data.to_csv('./Partial_Output/_5_ManipulatedData.csv')
data.to_csv(f'../History/Week {thisweek}/Partial_Output/_5_ManipulatedData.csv')

q1s_1 = distb(p1, s1, drop1, t, d0, d1, allzero)[1]
q3s_1 = distb(p1, s1, drop1, t, d0, d1, allzero)[2]
s_q_1 = distb(p1, s1, drop1, t, d0, d1, allzero)[3:5]
index = ["Bottom 25% S", "Middle 50% S", "Top 25% S"]

P_Quantile_DF = pd.DataFrame({"Q05-1": q1s_1, "Q95-1": q3s_1}, index = index)

q1s_2 = distb(p2, s2, drop2, t, d0, d1, allzero)[1]
q3s_2 = distb(p2, s2, drop2, t, d0, d1, allzero)[2]
P_Quantile_DF.insert(2, "Q05-2", q1s_2)
P_Quantile_DF.insert(3, "Q95-2", q3s_2)
s_q_2 = distb(p2, s2, drop2, t, d0, d1, allzero)[3:5]

q1s_3 = distb(p3, s3, drop3, t, d0, d1, allzero)[1]
q3s_3 = distb(p3, s3, drop3, t, d0, d1,allzero)[2]
P_Quantile_DF.insert(4, "Q05-3", q1s_3)
P_Quantile_DF.insert(5, "Q95-3", q3s_3)
s_q_3 = distb(p3, s3, drop3, t, d0, d1, allzero)[3:5]

P_Quantile_DF.to_csv("./Partial_Output/_6_P_Quantiles.csv")
P_Quantile_DF.to_csv(f"../History/Week {thisweek}/Partial_Output/_6_P_Quantiles.csv")

sIndex = ['S-Q1', 'S-Q3']
S_Quantile_DF = pd.DataFrame({"Score1": s_q_1, "Score2": s_q_2, "Score3": s_q_3}, index = sIndex)
S_Quantile_DF.to_csv("./Partial_Output/_7_S_Quantiles.csv")
S_Quantile_DF.to_csv(f"../History/Week {thisweek}/Partial_Output/_7_S_Quantiles.csv")





# PART 10: SPLITTING DATA AND FITTING MODEL

# Remake newest columns from Data DataFrame as a list
q1_1 = list(P_Quantile_DF['Q05-1'])
q3_1 = list(P_Quantile_DF['Q95-1'])
q1_2 = list(P_Quantile_DF['Q05-2'])
q3_2 = list(P_Quantile_DF['Q95-2'])
q1_3 = list(P_Quantile_DF['Q05-3'])
q3_3 = list(P_Quantile_DF['Q95-3'])

tag1 = list(data['Tag1'])
tag2 = list(data['Tag2'])
tag3 = list(data['Tag3'])



# write function to help with returning accuracy score
def as_pro(y_test, y_pred, name):
    """ Accuracy Score Pro - returns just the true positive rate as this is what is most sought after by the company """
    y_test = list(y_test[name])
    y_pred = list(y_pred)
    
    length = len(y_test)
    
    lengthtrue = 0
    lengthfalse = 0
    
    for i in range(length):
        if y_test[i] == 'Normal':
            lengthfalse += 1
        else:
            lengthtrue += 1
    
    counttp = 0
    countfp = 0
    countfn = 0
    counttn = 0
    
    for i in range(length):
        if y_test[i] == 'Normal':
            if y_pred[i] == 'Normal':
                counttn += 1
            
            else:
                countfp += 1
        else:
            if y_pred[i] != 'Normal':
                counttp += 1
            
            else:
                countfn += 1
    
    return [counttp/lengthtrue, countfp/lengthfalse, countfn/lengthtrue, counttn/lengthfalse]


def find_score(p, q1, q3, tag, data):
    """ create the tags for training based on tags and quantiles - increase, normal or decrease """
    
    # create the y values for training the data
    bottag = list()
    midtag = list()
    toptag = list()
    
    for i in range(len(tag)):
        if tag[i] == 1:

            if p[i] <= q1[0]:
                
                bottag.append("Decrease")
                
            elif q1[0] < p[i] and p[i] <= q3[0]:
                
                bottag.append("Normal")
                
            else:
                
                bottag.append("Increase")

        elif tag[i] == 2:

            if p[i] <= q1[1]:
                
                midtag.append("Decrease")
                
            elif q1[1] < p[i] and p[i] <= q3[1]:
                
                midtag.append("Normal")
                
            else:
                
                midtag.append("Increase")

        elif tag[i] == 3:

            if p[i] <= q1[2]:
                
                toptag.append("Decrease")
                
            elif q1[2] < p[i] and p[i] <= q3[2]:
                
                toptag.append("Normal")
                
            else:
                
                toptag.append("Increase")

    # make them into a dataframe
    bottag = pd.DataFrame({"Bottom Labels": bottag})
    midtag = pd.DataFrame({"Middle Labels": midtag})
    toptag = pd.DataFrame({"Top Labels": toptag})

    # create some lists to drop rows and drop them - the remaining dataset is what we will test
    droplist1 = list()
    droplist2 = list()
    droplist3 = list()

    for i in range(len(tag)):
        if tag[i] != 1:
            droplist1.append(i)
        if tag[i] != 2:
            droplist2.append(i)
        if tag[i] != 3:
            droplist3.append(i)

    pretestbot = data.drop(droplist1)
    pretestmid = data.drop(droplist2)
    pretesttop = data.drop(droplist3)

    # project only columns that have the training attributes - this is now the x values for training the data - they will match the y values because the order of the dataframe and the list bottag midtag toptag etc are held constant by python.
    pretestbot = pretestbot[['Assets', 'Actions', 'Competency', 'Form_record', 'Form_template', 'Incident', 'Users', 'Users_induction', 'Users_norm_emp', 'Horizontal quantile 1', 'Vertical quantile 1', 'Trend 1']]
    pretestmid = pretestmid[['Assets', 'Actions', 'Competency', 'Form_record', 'Form_template', 'Incident', 'Users', 'Users_induction', 'Users_norm_emp', 'Horizontal quantile 1', 'Vertical quantile 1', 'Trend 1']]
    pretesttop = pretesttop[['Assets', 'Actions', 'Competency', 'Form_record', 'Form_template', 'Incident', 'Users', 'Users_induction', 'Users_norm_emp', 'Horizontal quantile 1', 'Vertical quantile 1', 'Trend 1']]
    
    # do k=5 fold testing and get the average score for each of the three testers
    k = 5

    kf = KFold(n_splits = k, shuffle = True, random_state = 42)
    acc_score1 = []
    for train_index, test_index in kf.split(pretestbot):
        x_train, x_test = pretestbot.iloc[train_index, :], pretestbot.iloc[test_index,:]
        y_train, y_test = bottag.iloc[train_index, :], bottag.iloc[test_index,:]

        dtbot = DecisionTreeClassifier(criterion = 'entropy', random_state = 42, max_depth = 12)
        dtbot.fit(x_train, y_train)

        y_pred = dtbot.predict(x_test)
        acc_score1.append(as_pro(y_test, y_pred, 'Bottom Labels')[0])

    avgbot = sum(acc_score1)/k


    kf = KFold(n_splits = k, shuffle = True, random_state = 42)
    acc_score2 = []
    for train_index, test_index in kf.split(pretestmid):
        x_train, x_test = pretestmid.iloc[train_index, :], pretestmid.iloc[test_index,:]
        y_train, y_test = midtag.iloc[train_index, :], midtag.iloc[test_index,:]

        dtmid = DecisionTreeClassifier(criterion = 'entropy', random_state = 42, max_depth = 12)
        dtmid.fit(x_train, y_train)

        y_pred = dtmid.predict(x_test)
        acc_score2.append(as_pro(y_test, y_pred, 'Middle Labels')[0])

    avgmid = sum(acc_score2)/k


    kf = KFold(n_splits = k, shuffle = True, random_state = 42)
    acc_score3 = []
    for train_index, test_index in kf.split(pretesttop):
        x_train, x_test = pretesttop.iloc[train_index, :], pretesttop.iloc[test_index,:]
        y_train, y_test = toptag.iloc[train_index, :], toptag.iloc[test_index,:]

        dttop = DecisionTreeClassifier(criterion = 'entropy', random_state = 42, max_depth = 12)
        dttop.fit(x_train, y_train)

        y_pred = dttop.predict(x_test)
        acc_score3.append(as_pro(y_test, y_pred, 'Top Labels')[0])

    avgtop = sum(acc_score3)/k

    # calculate a weighted score for the k-fold tested predictors of this Usage Score function
    totallen = len(bottag) + len(midtag) + len(toptag)
    weighted_score = len(bottag)/totallen * avgbot + len(midtag)/totallen * avgmid + len(toptag)/totallen * avgtop
    
    return weighted_score

scores = [find_score(p1, q1_1, q3_1, tag1, data), find_score(p2, q1_2, q3_2, tag2, data), find_score(p3, q1_3, q3_3, tag3, data)]

# find the Usage Score function that returns the highest Weighted Score. This is what we will select to be our final predictor. 
curr = 0
currmax = scores[0]
for i in range(len(scores)):
    if scores[i] > currmax:
        curr = i

selection = curr + 1 # To change from indicies into actual Usage Score coefficient set number.  

#### FUTURE CHANGE: the selection process for the best Usage Score to use for prediction purposes is something that can be reconsidered in future.





# PART 11: FIND BEST FIT ON SELECTED USAGE SCORE FUNCTION'S PREDICTOR (ADJUSTING DEPTH) AND OUTPUT RESULT

# write function to help with returning detailed accuracy score
def as_pro2(y_test, y_pred, name):
    """ Accuracy Score Pro - version 2 - more detailed than as_pro but purely used to output statistics, designed specifically for our purposes (because of two types of Positive) """
    y_test = list(y_test[name])
    y_pred = list(y_pred)
    
    length = len(y_test)
    
    # Get number of actual "increase", number of actual "decrease" and number of actual "normal"
    lengthtrue1 = 0
    lengthtrue2 = 0
    lengthfalse = 0
    
    for i in range(length):
        if y_test[i] == 'Normal':
            lengthfalse += 1
        elif y_test[i] == 'Increase':
            lengthtrue1 += 1
        else:
            lengthtrue2 += 1
            
    
    counttp1 = 0     # should be inc, predicted inc
    counttp2 = 0     # should be dec, predicted dec
    counttn = 0      # should be normal, predicted normal
    
    countfp1 = 0     # should be normal, predicted inc
    countfp2 = 0     # should be normal, predicted dec
    
    countfp1bad = 0  # should be inc, predicted dec
    countfp2bad = 0  # should be dec, predicted inc
    
    countfn1 = 0     # should be inc, predicted normal 
    countfn2 = 0     # should be dec, predicted normal
    
    
    for i in range(length):
        if y_test[i] == 'Increase':
            if y_pred[i] == 'Increase':
                counttp1 += 1
            
            elif y_pred[i] == 'Decrease':
                countfp1bad += 1
            
            else:
                countfn1 += 1
        
        elif y_test[i] == 'Decrease':
            if y_pred[i] == 'Decrease':
                counttp2 += 1
            
            elif y_pred[i] == 'Increase':
                countfp2bad += 1
            
            else:
                countfn2 += 1
        
        else:
            if y_pred[i] == 'Normal':
                counttn += 1
            
            elif y_pred[i] == 'Increase':
                countfp1 += 1
            
            else:
                countfp2 += 1
    
    return [counttp1/lengthtrue1, counttp2/lengthtrue2, countfp1bad/lengthtrue1, countfp2bad/lengthtrue2, countfp1/lengthfalse, countfp2/lengthfalse, countfn1/lengthtrue1, countfn2/lengthtrue2, counttn/lengthfalse]


def train(p, q1, q3, tag, data, selection, thisdate):
    """ Helper function for finding which maxdepth gives best performance in the Usage Score function that have been chosen in PART 10 """
    # repeats process for final preprocessing of data into format that sklearn.tree likes it. Comments same as code in function of PART 10; please see above.
    bottag = list()
    midtag = list()
    toptag = list()
    
    for i in range(len(tag)):
        if tag[i] == 1:

            if p[i] <= q1[0]:
                bottag.append("Decrease")
            elif q1[0] < p[i] and p[i] <= q3[0]:
                bottag.append("Normal")
            else:
                bottag.append("Increase")

        elif tag[i] == 2:

            if p[i] <= q1[1]:
                midtag.append("Decrease")
            elif q1[1] < p[i] and p[i] <= q3[1]:
                midtag.append("Normal")
            else:
                midtag.append("Increase")

        elif tag[i] == 3:

            if p[i] <= q1[2]:
                toptag.append("Decrease")
            elif q1[2] < p[i] and p[i] <= q3[2]:
                toptag.append("Normal")
            else:
                toptag.append("Increase")

    # make them into a dataframe
    bottag = pd.DataFrame({"Bottom Labels": bottag})
    midtag = pd.DataFrame({"Middle Labels": midtag})
    toptag = pd.DataFrame({"Top Labels": toptag})

    # create some lists to drop rows and drop them - the remaining dataset is what we will test
    droplist1 = list()
    droplist2 = list()
    droplist3 = list()

    for i in range(len(tag)):
        if tag[i] != 1:
            droplist1.append(i)
        if tag[i] != 2:
            droplist2.append(i)
        if tag[i] != 3:
            droplist3.append(i)

    pretestbot = data.drop(droplist1)
    pretestmid = data.drop(droplist2)
    pretesttop = data.drop(droplist3)

    # project only columns that have the training attributes
    pretestbot = pretestbot[['Assets', 'Actions', 'Competency', 'Form_record', 'Form_template', 'Incident', 'Users', 'Users_induction', 'Users_norm_emp', 'Horizontal quantile 1', 'Vertical quantile 1', 'Trend 1']]
    pretestmid = pretestmid[['Assets', 'Actions', 'Competency', 'Form_record', 'Form_template', 'Incident', 'Users', 'Users_induction', 'Users_norm_emp', 'Horizontal quantile 1', 'Vertical quantile 1', 'Trend 1']]
    pretesttop = pretesttop[['Assets', 'Actions', 'Competency', 'Form_record', 'Form_template', 'Incident', 'Users', 'Users_induction', 'Users_norm_emp', 'Horizontal quantile 1', 'Vertical quantile 1', 'Trend 1']]

    # Utilises train_test_split to split data into testing and training
    x_trainbot, x_testbot, y_trainbot, y_testbot = train_test_split(pretestbot, bottag, train_size = 0.8, test_size = 0.2, random_state = 42)
    x_trainmid, x_testmid, y_trainmid, y_testmid = train_test_split(pretestmid, midtag, train_size = 0.8, test_size = 0.2, random_state = 42)
    x_traintop, x_testtop, y_traintop, y_testtop = train_test_split(pretesttop, toptag, train_size = 0.8, test_size = 0.2, random_state = 42)

    # run data with maxdepth 1 to 100 to find which has the best 
    ws = list()
    rt = list()

    for i in range(1, 101):
    
    #### FUTURE CHANGE: for each of the depth do k-fold test train k=5
    
        dtbot = DecisionTreeClassifier(criterion = 'entropy', random_state = 1, max_depth = i)
        dtbot.fit(x_trainbot, y_trainbot)
        dtmid = DecisionTreeClassifier(criterion = 'entropy', random_state = 1, max_depth = i)
        dtmid.fit(x_trainmid, y_trainmid)
        dttop = DecisionTreeClassifier(criterion = 'entropy', random_state = 1, max_depth = i)
        dttop.fit(x_traintop, y_traintop)

        y_predbot = dtbot.predict(x_testbot)
        bot_acc_score = as_pro(y_testbot, y_predbot, 'Bottom Labels')[0]

        y_predmid = dtmid.predict(x_testmid)
        mid_acc_score = as_pro(y_testmid, y_predmid, 'Middle Labels')[0]

        y_predtop = dttop.predict(x_testtop)
        top_acc_score = as_pro(y_testtop, y_predtop, 'Top Labels')[0]


        totallen = len(bottag) + len(midtag) + len(toptag)
        weighted_score = len(bottag)/totallen * bot_acc_score + len(midtag)/totallen * mid_acc_score + len(toptag)/totallen * top_acc_score
        ws.append(weighted_score)
    
    
    # pick out the maxdepth that gives highest weighted score
    maxws = 0
    current = 1
    for i in range(len(ws)):
        if ws[i] > maxws:
            current = i+1
            maxws = ws[i]
            
    # train tree for that depth
    dtbot = DecisionTreeClassifier(criterion = 'entropy', random_state = 1, max_depth = current)
    dtbot.fit(x_trainbot, y_trainbot)
    dtmid = DecisionTreeClassifier(criterion = 'entropy', random_state = 1, max_depth = current)
    dtmid.fit(x_trainmid, y_trainmid)
    dttop = DecisionTreeClassifier(criterion = 'entropy', random_state = 1, max_depth = current)
    dttop.fit(x_traintop, y_traintop)
    
    
    # Produce statistics for this predictor and output into relevent directories
    y_predbot = dtbot.predict(x_testbot)
    y_predmid = dtmid.predict(x_testmid)
    y_predtop = dttop.predict(x_testtop)
    
    statbot = as_pro2(y_testbot, y_predbot, 'Bottom Labels')
    statmid = as_pro2(y_testmid, y_predmid, 'Middle Labels')
    stattop = as_pro2(y_testtop, y_predtop, 'Top Labels')
    
    stat = list()
    for i in range(9):
        stat.append(len(bottag)/totallen * statbot[i] + len(midtag)/totallen * statmid[i] + len(toptag)/totallen * stattop[i])
    
    stat.append(selection)
    stat.append(thisweek)
    
    Index = ['True Positive 1', 'True Positive 2', 'Bad False Positive 1', 'Bad False Positive 2', 'False Positive 1', 'False Positive 2', 'False Negative 1', 'False Negative 2', 'True Negative', 'Selection', 'Thisweek']
    stat_out = pd.DataFrame({"Statistics": stat}, index = Index)
    
    # This file contains the statistics (outputs of accuracy score pro 2) for the predictor, as well as the usage score coefficients (r^2, NMI activity_score or NMI actions) that was selected, and also the week up to which the data was used to produce the predictor (useful for checking in what week the predictor came from in the home directory)
    stat_out.to_csv("Statistics_Init.csv")
    
    stat_out.to_csv("../Statistics.csv")
    
    if not os.path.exists(f'../History/Week {thisweek}'):
        os.mkdir(f'../History/Week {thisweek}')
    
    stat_out.to_csv(f'../History/Week {thisweek}/Statistics.csv')
    
    
    # exports tree objects for visualisation later
    # Please open ../Visualise_tree.ipynb on Jupyterlab to visualise the trees - but note if ran ContinuousTrain.py could be visualising a different tree rather than the initial one.
    
    if not os.path.exists('./Tree_visual'):
        os.mkdir('./Tree_visual')

    export_graphviz(dtbot, out_file = f"./Tree_visual/dtbot.dot", feature_names = pretestbot.columns, filled = True, rounded = True)
    export_graphviz(dtmid, out_file = f"./Tree_visual/dtmid.dot", feature_names = pretestbot.columns, filled = True, rounded = True)
    export_graphviz(dttop, out_file = f"./Tree_visual/dttop.dot", feature_names = pretestbot.columns, filled = True, rounded = True)

    if not os.path.exists(f'../History/Week {thisweek}/Tree_visual'):
        os.makedirs(f'../History/Week {thisweek}/Tree_visual')

    export_graphviz(dtbot, out_file = f"../History/Week {thisweek}/Tree_visual/dtbot.dot", feature_names = pretestbot.columns, filled = True, rounded = True)
    export_graphviz(dtmid, out_file = f"../History/Week {thisweek}/Tree_visual/dtmid.dot", feature_names = pretestbot.columns, filled = True, rounded = True)
    export_graphviz(dttop, out_file = f"../History/Week {thisweek}/Tree_visual/dttop.dot", feature_names = pretestbot.columns, filled = True, rounded = True)

    if not os.path.exists('../Tree_visual'):
        os.mkdir('../Tree_visual')

    export_graphviz(dtbot, out_file = f"../Tree_visual/dtbot.dot", feature_names = pretestbot.columns, filled = True, rounded = True)
    export_graphviz(dtmid, out_file = f"../Tree_visual/dtmid.dot", feature_names = pretestbot.columns, filled = True, rounded = True)
    export_graphviz(dttop, out_file = f"../Tree_visual/dttop.dot", feature_names = pretestbot.columns, filled = True, rounded = True)
    
    return dtbot, dtmid, dttop

# make the final predictor
if selection == 1:
    dtbot, dtmid, dttop = train(p1, q1_1, q3_1, tag1, data, selection, thisweek)
elif selection == 2:
    dtbot, dtmid, dttop = train(p2, q1_2, q3_2, tag2, data, selection, thisweek)
else:
    dtbot, dtmid, dttop = train(p3, q1_3, q3_3, tag3, data, selection, thisweek)

# save the relevent predictors as an OS file in relevent directories
if not os.path.exists('./Prediction_Objects'):
    os.mkdir('./Prediction_Objects')
    
pick_out1 = open('./Prediction_Objects/dtbot.pickle','wb')
pickle.dump(dtbot, pick_out1)
pick_out2 = open('./Prediction_Objects/dtmid.pickle','wb')
pickle.dump(dtmid, pick_out2)
pick_out3 = open('./Prediction_Objects/dttop.pickle','wb')
pickle.dump(dttop, pick_out3)

if not os.path.exists('../Prediction_Objects'):
    os.mkdir('../Prediction_Objects')

pick_out1 = open('../Prediction_Objects/dtbot.pickle','wb')
pickle.dump(dtbot, pick_out1)
pick_out2 = open('../Prediction_Objects/dtmid.pickle','wb')
pickle.dump(dtmid, pick_out2)
pick_out3 = open('../Prediction_Objects/dttop.pickle','wb')
pickle.dump(dttop, pick_out3)

if not os.path.exists(f'../History/Week {thisweek}/Prediction_Objects'):
    os.makedirs(f'../History/Week {thisweek}/Prediction_Objects')

pick_out1 = open(f'../History/Week {thisweek}/Prediction_Objects/dtbot.pickle','wb')
pickle.dump(dtbot, pick_out1)
pick_out2 = open(f'../History/Week {thisweek}/Prediction_Objects/dtmid.pickle','wb')
pickle.dump(dtmid, pick_out2)
pick_out3 = open(f'../History/Week {thisweek}/Prediction_Objects/dttop.pickle','wb')
pickle.dump(dttop, pick_out3)
