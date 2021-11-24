### CLIENT USAGE PREDICTOR - prediction script
### Code produced by Lang (Ron) Chen August-October 2021 for Lucidity Software
""" Wrangles raw data of recent weeks, appends it to the previous wrangled data and outputs predictions for upcoming week """

# Input: 
#    Argument 1: the start date for the new data (only include data from after this date); logically should be a Monday. If this script is run as preferably (weekly), then this argument should be the Monday of the previous week. Must be in form ‘[d]d/[m]m/yyyy’ (If day or month single digit can input just as single digit). 
#         please ensure date is valid and is after the FIRSTDATE (by default July 3rd 2017)
#    Argument 2: a cut-off date for data; logically should be a Sunday Must be in form ‘[d]d/[m]m/yyyy’ (If day or month single digit can input just as single digit). 
#         please ensure date is valid and is after the FIRSTDATE (by default July 3rd 2017) and after the date of Argument 1

#     Initial raw data files need to be stored in directory './History/Week {this week week number}/Data’. 
#     -File names must be ‘action.csv’, ‘assets.csv’, ‘attachment.csv’, ‘competency_record.csv’, ‘form_record.csv’, ‘form_template.csv’, ‘incident.csv’, ‘users.csv’, associated with the data of the corresponding filename
#     -Each csv must include columns that include ‘domain’ for company name, and ‘created_at’. ‘users.csv’ must also have column name ‘hr_type’, with data values ‘Casual’, ‘Employee’, ‘Subcontractor’ denoting regular employees and ‘InductionUser’ denoting induction users.
#     -There should be no other tester domains in the data apart from ‘demo’, ‘demo_2’ and ‘cruse’

#     -the dates for all files should be in form of yyyy-mm-dd.
#     *if the form of these are different then need to edit PART 2 in the script. 


# Output: several partial outputs - partial outputs of wrangled data and prediction output exported to various directories including the home directory (relatively '.') and the History directory (into the relevent week) 






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
# Slightly different from PART 1 of InitialTraining.py's part 1 because it also needs to account for a subset of dates (from just the date of argument 1 to the date of argument 2)

# Reads in 'cut-off date' for data to be used to train 
datastartdate = sys.argv[1]
currdate = sys.argv[2]

# Assumes that date format in 'dd/mm/yyyy' format
cutoffday = int(currdate.split('/')[0])
cutoffmonth = int(currdate.split('/')[1])
cutoffyear = int(currdate.split('/')[2])

def datejoin(day, month, year):
    """For joining up three numbers into a date"""
    return (f'{str(day)}/{str(month)}/{str(year)}')


def leapyear(year):
    """For determining whether a year is a leap year"""
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

datastartdateday = int(datastartdate.split('/')[0])
datastartdatemonth = int(datastartdate.split('/')[1])
datastartdateyear = int(datastartdate.split('/')[2])

days = [29, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]    
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
years = range(2017, cutoffyear+1)


datematchweek = dict()
minidatematchweek = dict()
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
            
            if year > datastartdateyear:
                
                minidatematchweek[date] = week
                
            elif year == datastartdateyear:
                
                if month > datastartdatemonth:
                    
                    minidatematchweek[date] = week
                    
                elif month == datastartdatemonth and day >= datastartdateday:  
                            
                    minidatematchweek[date] = week

dates = list(datematchweek.keys())
weekno = list(datematchweek.values())

# Make the dictionary of dates to week number into a dataframe
DatesToWeek_DF = pd.DataFrame({'dates': dates, 'weekno':weekno})

DatesToWeek_DF.to_csv('DateMatchWeek.csv')

# Record the week number of the cut-off date
thisweek = max(weekno)

# create a subset just for wrangling this week's data 
minidates = list(minidatematchweek.keys())
miniweekno = list(minidatematchweek.values())

MiniDatesToWeek_DF = pd.DataFrame({'dates': minidates, 'weekno':miniweekno})

datastartweek = min(miniweekno)



# If has already run ContinuousTrain.py with same arguments then this PART 2 and PART 3 do not need to be rerun
if not os.path.isfile(f'./History/Week {thisweek}/_3_combined_cleaned_data.csv'):
    # PART 2: WRANGLING ORIGINAL DATA

    #### FUTURE UPDATE: towranglelist1 stores records which have dates in format [d]d/[m]m/yyyy; towranglelist2 stores records which have dates in format yyyy-mm-dd. Need to update lists accordingly
    towranglelist1 = [] #### FUTURE UPDATE: because all data at the time of writing came from Quicksight, all their formats were yyyy-mm-dd so all were put into towranglelist2
    towranglelist2 = ['action.csv', 'competency_record.csv', 'form_record.csv', 'incident.csv', 'users.csv', 'assets.csv', 'form_template.csv']
    towranglelist3 = ['users.csv'] 
    towranglelist4 = ['users.csv']
    # because users need to be wrangled in terms of employees and inductioneers. 3 is inductionuser 4 is employees

    def wrangle(filename, datedata, mode):
        """ cleans the file. 4 modes for four different ways to clean the data - all pretty similar except mode 3 and 4 selects users of particular hr types, and mode 2 deals with dates of a different format """
        data = pd.read_csv(f"./History/Week {thisweek}/Data/{filename}")
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
        if mode in [2,3,4]:
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

        if mode in [2, 3, 4]:
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
            out.to_csv(f'./History/Week {thisweek}/Partial_Output/_2_{filename.split(".")[0]}_clean.csv', index = False)

        elif mode == 3:
            out.to_csv(f'./History/Week {thisweek}/Partial_Output/_2_Users_Inductee_clean.csv', index = False)

        else:
            out.to_csv(f'./History/Week {thisweek}/Partial_Output/_2_Users_norm_employee_clean.csv', index = False)

    # OS housekeeping and running each of the files through wrangle()
    if not os.path.exists(f'./History/Week {thisweek}/Partial_Output'):
        os.mkdir(f'./History/Week {thisweek}/Partial_Output')

    for file in towranglelist1:
        wrangle(file, MiniDatesToWeek_DF, 1)

    for file in towranglelist2:
        wrangle(file, MiniDatesToWeek_DF, 2)

    for file in towranglelist3:
        wrangle(file, MiniDatesToWeek_DF, 3)

    for file in towranglelist4:
        wrangle(file, MiniDatesToWeek_DF, 4)





    # PART 3: COMBINE PREVIOUSLY WRANGLED DATAFRAMES INTO ONE (FILLING IN WEEKS WITH NO ACTIVITY)

    # import all cleaned data
    asset = pd.read_csv(f'./History/Week {thisweek}/Partial_Output/_2_assets_clean.csv')
    actions = pd.read_csv(f'./History/Week {thisweek}/Partial_Output/_2_action_clean.csv')
    competency = pd.read_csv(f'./History/Week {thisweek}/Partial_Output/_2_competency_record_clean.csv')
    form_record = pd.read_csv(f'./History/Week {thisweek}/Partial_Output/_2_form_record_clean.csv')
    form_templates = pd.read_csv(f'./History/Week {thisweek}/Partial_Output/_2_form_template_clean.csv')
    incidents = pd.read_csv(f'./History/Week {thisweek}/Partial_Output/_2_incident_clean.csv')
    users = pd.read_csv(f'./History/Week {thisweek}/Partial_Output/_2_users_clean.csv')
    users_induct = pd.read_csv(f'./History/Week {thisweek}/Partial_Output/_2_users_Inductee_clean.csv')
    users_norm_emp = pd.read_csv(f'./History/Week {thisweek}/Partial_Output/_2_users_norm_employee_clean.csv')

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

    # read in data file _3_ from last week (for concatenating this week's data onto)
    olddata = pd.read_csv(f'./History/Week {datastartweek-1}/Partial_Output/_3_combined_cleaned_data.csv')
    olddoms = list(set(olddata['Domain']))

    # Create a template for recording the data (now fill out gaps between start week and week 216 where there is 0 data)
    combineddatatemplate = dict()

    # first initiate a blank dictionary with all weeks from the week of argument 1 of this script (the date from which we should start counting in the data)
    # this is slightly different from InitialTraining because we are adding new weeks on top of last time's data, so even if no activity at all these weeks we still need to include it.
    for dom in olddoms:
        for j in range(datastartweek, thisweek+1):
            combineddatatemplate[f'{dom} {j}'] = 0

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

    # make all the selfweek number for this new data an impossible value of -1
    prev = doms[0]
    for i in range(len(doms)):
        selfweeks.append(-1)


    # Create a prelim score of sum of the three attributes to be used in determining a function
    baseactscore1 = list()
    for i in range(len(uniqueid)):
        baseactscore1.append(form_recordcount[i]+competencycount[i]+users_inductcount[i])

    # turn it into one dataframe and concatanate onto old data
    out = pd.DataFrame({'ID': uniqueid, 'Domain': doms, 'Week': weekss, 'Selfweeks': selfweeks,
                        'Assets': assetcount, 'Actions': actioncount, 'Competency': competencycount, 
                        'Form_record': form_recordcount, 'Form_template': form_templatescount,
                       'Incident': incidentscount, 'Users': userscount, 
                        'Users_induction': users_inductcount, 'Users_norm_emp': users_norm_empcount, 
                        "Prelim_action_score": baseactscore1})

    combineddata = pd.concat([olddata, out])
    combineddata = combineddata.sort_values(['Domain', 'Week'], axis=0) # Sorting the dataframe by domain and weeks

    selfweeks = list(combineddata['Selfweeks'])
    domain = list(combineddata['Domain'])

    # for any self weeks that are -1 (newly added on), just add one onto it from previous week. (it works since the data is sorted)
    prev = domain[0]
    for i in range(len(selfweeks)):

        if domain[i] == prev and selfweeks[i] == -1:
            selfweeks[i] = selfweeks[i-1] + 1

        elif domain[i] != prev and selfweeks[i] == -1:
            selfweeks[i] = 1

        prev = domain[i]

    # replacing the column 'Selfweeks' with correct data        
    combineddata['Selfweeks'] = selfweeks

    # This file now has counts of all activities grouped by week by client/domain, sorted by domian and clients. 
    combineddata.to_csv(f'./History/Week {thisweek}/Partial_Output/_3_combined_cleaned_data.csv', index = False)





# PART 4 EXCLUDED BECAUSE WE ARE PREDICTING RATHER THAN CREATING A PREDICTOR, SO WE USE THE USAGE SCORES CREATED LAST WEEK IN PART 5





# PART 5: CREATE USAGE SCORES

# reread data
Stat = pd.read_csv(f'./Statistics.csv') # find out which week our predictor was created in and get the coefficients from relevent week)
weekofpredictor = int(Stat['Statistics'][10])

data = pd.read_csv(f'./History/Week {thisweek}/Partial_Output/_3_combined_cleaned_data.csv')
coef = pd.read_csv(f'./History/Week {weekofpredictor}/Partial_Output/_4_UsageScoreCoefficients.csv')

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
    p_q3


    # tag any row where %change is over the 75% for overall, and its score is less than 25% for its own usage score
    droplist = list()
    for i in range(len(d1)):
        droplist.append(0)

    tmp = list()
    prev = d1[0]
    startindex = 0
    for i in range(len(d1)):
        if d1[i] != prev:
            count = startindex-1
            tmp_array = np.array(tmp)
            owns_q1 = np.quantile(tmp_array, .25)

            for j in range(startindex, startindex+26):
                if p[j] > p_q3 and s[j] < owns_q1:
                    count = j

            for k in range(startindex, count+1):
                droplist[k] = 1

            tmp = list()
            startindex = i

        else:
            if t[i] == 0 and p[i] != -2:
                tmp.append(s[i])


        prev = d1[i]

    count = startindex-1
    tmp_array = np.array(tmp)
    owns_q1 = np.quantile(tmp_array, .25)

    for j in range(startindex, startindex+26):
        if p[j] > p_q3 and s[j] < owns_q1:
            count = j

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
        # append an impossible number for quantile - easier to wipe off later on (in fact not used - but still good to have it to be attached to rows that should be wiped off)

    prev = d1[0]
    tmp = list()
    for i in range(len(d1)):
        if d1[i] != prev:
            tmp=list()


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
        # append an impossible number for quantile - easier to wipe off later on

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
data.insert(14, "Horizontal quantile 1", hor(p1, drop1, t, d1), True)
data.insert(15, "Vertical quantile 1", vert(p1, drop1, t, d0, d1, d3), True)
data.insert(16, "Trend 1", trendd(p1, drop1, t, d0, d1), True)

data.insert(17, "Horizontal quantile 2", hor(p2, drop2, t, d1), True)
data.insert(18, "Vertical quantile 2", vert(p2, drop2, t, d0, d1, d3), True)
data.insert(19, "Trend 2", trendd(p2, drop2, t, d0, d1), True)

data.insert(20, "Horizontal quantile 3", hor(p3, drop3, t, d1), True)
data.insert(21, "Vertical quantile 3", vert(p3, drop3, t, d0, d1, d3), True)
data.insert(22, "Trend 3", trendd(p3, drop3, t, d0, d1), True)

data.to_csv(f"./History/Week {thisweek}/Partial_Output/_5_1_Partial_ManipulatedData.csv", index = False)





# PART 9: CREATE TAGS

# read in Quantiles for scores data from last week to create tags
Stat = pd.read_csv(f'./Statistics.csv') # find out which week our predictor was created in and get the coefficients from relevent week)
weekofpredictor = int(Stat['Statistics'][10])

S_Quantile_DF = pd.read_csv(f'./History/Week {weekofpredictor}/Partial_Output/_7_S_Quantiles.csv')

droplist = []
weeksss = data['Week']
for i in range(len(weeksss)):
    if weeksss[i] != thisweek:
        droplist.append(i)

thisweekdata = data.drop(droplist)

d00 = list(thisweekdata['ID'])
d10 = list(thisweekdata['Domain'])
d30 = list(thisweekdata['Selfweeks'])
p10 = list(thisweekdata['%Change Score 1'])
p20 = list(thisweekdata['%Change Score 2'])
p30 = list(thisweekdata['%Change Score 3'])
s10 = list(thisweekdata['Score1'])
s20 = list(thisweekdata['Score2'])
s30 = list(thisweekdata['Score3'])
drop10 = list(thisweekdata['Drop_start1'])
drop20 = list(thisweekdata['Drop_start2'])
drop30 = list(thisweekdata['Drop_start3'])
t0 = list(thisweekdata['Tailzero'])

a10 = list(thisweekdata['Assets'])
a20 = list(thisweekdata['Actions'])
a30 = list(thisweekdata['Competency'])
a40 = list(thisweekdata['Form_record'])
a50 = list(thisweekdata['Form_template'])
a60 = list(thisweekdata['Incident'])
a70 = list(thisweekdata['Users']) 

# an extra list here to exclude any rows where there were no activity for a particular company in a particular week. #### Future Update: this was a last minute addition so this column was not inserted into the data DF. When adding it to the DF later, beware of the indexes for pd.Dataframe.insert() later on in the code
allzero = list()

for i in range(len(d00)):
    if a10[i] == 0 and a20[i] == 0 and a30[i] == 0 and a40[i] == 0 and a50[i] == 0 and a60[i] == 0 and a70[i] == 0:
        allzero.append(1)
    else:
        allzero.append(0)

        
def distb(p, s, drop, t, d0, d1, squant, d3, allzero):
    """ Helper function for tagging each row as "bottom 25% usage score", "middle 50% usage score" and "top 25% usage score" """

    tag = list()
    for i in range(len(d0)):
        tag.append(0)
        # here 0 is the impossible value 
    
    # label the data top 25% score (3), bot 25% score (1) or middle 50% score (2) by adding an extra column to the data
    for i in range(len(d0)):
        if t[i] == 0 and drop[i] == 0 and d3[i] > 26 and allzero[i] == 0:
            if s[i] <= squant[0]:
                tag[i] = 1
            elif squant[0] < s[i] and s[i] <= squant[1]:
                tag[i] = 2
            else:
                tag[i] = 3
    
    return tag

# Add the columns into the data frame
thisweekdata.insert(33, "Tag1", distb(p10, s10, drop10, t0, d00, d10, list(S_Quantile_DF['Score1']), d30, allzero), True)
thisweekdata.insert(34, "Tag2", distb(p20, s20, drop20, t0, d00, d10, list(S_Quantile_DF['Score2']), d30, allzero), True)
thisweekdata.insert(35, "Tag3", distb(p30, s30, drop30, t0, d00, d10, list(S_Quantile_DF['Score3']), d30, allzero), True)

thisweekdata.to_csv(f"./History/Week {thisweek}/Partial_Output/_5_2_Partial_ManipulatedData_Thisweek.csv", index = False)

newindex = list()
for i in range(len(thisweekdata['ID'])):
    newindex.append(i)
    
thisweekdata.index = newindex





# PART 10: SPLITTING DATA AND FITTING MODEL

# Remake newest columns from Data DataFrame as a list
choice = Stat['Statistics'][9]

tag10 = list(thisweekdata['Tag1'])
tag20 = list(thisweekdata['Tag2'])
tag30 = list(thisweekdata['Tag3'])


with open('./Prediction_Objects/dtbot.pickle', 'rb') as f:
    dtbot = pickle.load(f)
with open('./Prediction_Objects/dtmid.pickle', 'rb') as g:
    dtmid = pickle.load(g)
with open('./Prediction_Objects/dttop.pickle', 'rb') as h:
    dttop = pickle.load(h)

def predict(p, tag, data, dtbot, dtmid, dttop):
    """ create the tags for training based on tags and quantiles - increase, normal or decrease, and conducting predicting for the data """
    
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
    botdata = pretestbot[['Assets', 'Actions', 'Competency', 'Form_record', 'Form_template', 'Incident', 'Users', 'Users_induction', 'Users_norm_emp', 'Horizontal quantile 1', 'Vertical quantile 1', 'Trend 1']]
    middata = pretestmid[['Assets', 'Actions', 'Competency', 'Form_record', 'Form_template', 'Incident', 'Users', 'Users_induction', 'Users_norm_emp', 'Horizontal quantile 1', 'Vertical quantile 1', 'Trend 1']]
    topdata = pretesttop[['Assets', 'Actions', 'Competency', 'Form_record', 'Form_template', 'Incident', 'Users', 'Users_induction', 'Users_norm_emp', 'Horizontal quantile 1', 'Vertical quantile 1', 'Trend 1']]
    
    overalldom = list()
    overallpred = list()
    score_range = list()

    if len(botdata) != 0:
        botpred = dtbot.predict(botdata)
        botdom = list(pretestbot['Domain'])
        overalldom = overalldom + botdom
        overallpred = overallpred + list(botpred)
        for i in range(len(botdata)):
            score_range.append('Bottom 25%')
            
    if len(middata) != 0:
        midpred = dtmid.predict(middata)
        middom = list(pretestmid['Domain'])
        overalldom = overalldom + middom
        overallpred = overallpred + list(midpred)
        for i in range(len(middata)):
            score_range.append('Middle 50%')
            
    if len(topdata) != 0:
        toppred = dttop.predict(topdata)
        topdom = list(pretesttop['Domain'])
        overalldom = overalldom + topdom
        overallpred = overallpred + list(toppred)
        for i in range(len(topdata)):
            score_range.append('Top 25%')
    
    alldom = list(thisweekdata['Domain'])
    for i in range(len(alldom)):
        if alldom[i] not in overalldom:
            overalldom.append(alldom[i])
            overallpred.append('N/A')
            score_range.append('N/A')
    

    output = pd.DataFrame({'Domain': overalldom, 'Prediction': overallpred, 'Usage Score': score_range})
    
    return output

if choice == 1:               
    predictions = predict(p10, tag10, thisweekdata, dtbot, dtmid, dttop)
elif choice == 2:
    predictions = predict(p20, tag20, thisweekdata, dtbot, dtmid, dttop)
else:
    predictions = predict(p30, tag30, thisweekdata, dtbot, dtmid, dttop)

predictions = predictions.sort_values(['Domain'], axis=0)

# adding on the Client and Client code for easier mapping later on
match = pd.read_excel('Mapping.xlsx')

finaldom = list(predictions['Domain'])

dommatchdict1 = dict() # dictionary for storing client name
dommatchdict2 = dict() # dictionary for storing client id

for i in range(len(finaldom)):
    dommatchdict1[finaldom[i]] = ''
    dommatchdict2[finaldom[i]] = ''

domain1 = list(match['Domain1'])
clientname = list(match['Client Code'])
clientid = list(match['Client'])
for i in range(len(domain1)):
    if domain1[i] in dommatchdict1:
        dommatchdict1[domain1[i]] = clientname[i]
        dommatchdict2[domain1[i]] = clientid[i]

dommatchlist1 = list(dommatchdict1.values())
dommatchlist2 = list(dommatchdict2.values())

predictions.insert(1, "Client name", dommatchlist1, True)
predictions.insert(2, "Client id", dommatchlist2, True)

# adding one last column for the predictor's own week number
predictorweeknumber = list()
predictorweeknumber.append(Stat['Statistics'][10])

for i in range(len(dommatchlist1)-1):
    predictorweeknumber.append(m.nan)

predictions.insert(5, 'Predictor\'s week number', predictorweeknumber, True)

predictions.to_csv('Predictions.csv', index = False)
predictions.to_csv(f'./History/Week {thisweek}/Predictions.csv', index = False)