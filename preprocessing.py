import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as sts
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.model_selection import StratifiedKFold, train_test_split
import warnings
from sklearn.metrics import mutual_info_score #brauchen wir nicht mehr
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler 
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
#ihr müsst evt. noch imblearn runterladen (pip install imbalanced-learn)


warnings.filterwarnings("ignore")

#teilweise hat nans die bedeuten die leute nehmen keine drogen... können wir also doch nicht löschen zB SRCPNRNM2 gibt noch viele dort ZB PNRMAINRSN
#beispiel statt replace df_cleaned.loc[df_cleaned['ALBSTWAY'] == 11, 'ALBSTWAY'] = 1
def skip99(data):
    data=data.replace(999,99)
    data=data.replace(9999,99)
    data=data.replace(89,99)
    data=data.replace(989,99)
    data=data.replace(9989,99)

    return data

def blank(data):
    df_cleaned=data.replace(998,98)
    df_cleaned=data.replace(9998,98)

    num_rows_with_98 = (df_cleaned == 98).any(axis=1).sum()

    for col in df_cleaned.columns:
        if (df_cleaned[col] == 98).sum() > 22000:  #eigentlich könnte man hier die nans dazunehmen
            del df_cleaned[col]
    
    return df_cleaned

def clean_data(data):
    
    #Alle Fragen zu Drogenkonsum
    df_new = data.loc[:,'CIGEVER':'MMFGIVE']
    df_new.drop(df_new.loc[:,'PREGNANT':'YMDEIMUDPY'].columns, axis='columns', inplace=True)
    print("CIGO",df_new["CIGOFRSM"].unique())
    #df_work=data.loc[:,('QUESTID2','CADRLAST':'MMFGIVE')]

    #df_new=pd.merge(df_new, data.loc[:,'CADRLAST':'MMFGIVE'],how='left', on='QUESTID2') Fehler weil Questid2 nicht im Dataframe vohanden ist

    #Alles inklusive Mental Health Einteilung
    df_new['Mental_health_status'] = data['MI_CAT_U'] 

    #Übersicht neues Dataset
    print('Shape of dataset: ',df_new.shape)
    print('++++Columns Index++++')
    print(df_new.columns)
    print('++++Datentypen++++')
    print(df_new.dtypes)
    print('+++++Head of Dataset+++++')
    print(df_new.head(10))
    print('Number of duplicated rows:',df_new.duplicated().sum()) #duplicate rows

    #Alle Zeilen löschen mit NA in Mental Health !!wie viele? brauchen wir auch zum argumentieren für die Arbeit!!
    df_cleaned = df_new.dropna(subset=['Mental_health_status'])
    print('Number of deleted rows:', len(df_new)-len(df_cleaned))
    print("CIGO",df_cleaned["CIGOFRSM"].unique())
    missing_values_per_column = df_cleaned.isna().sum().sort_values(ascending=False)
    print('++++Number of missing values per Column++++')
    print(missing_values_per_column)

    num_columns_not_zero = sum([1 for value in missing_values_per_column if value > 0])
    print("Number of columns with missing values:", num_columns_not_zero) #100 von 1756 Spalten enthalten leere Zeilen 
    #df_cleaned=df_cleaned.dropna(axis=1) #Drop alle Spalten, in denen Werte fehlen
    df_cleaned= df_cleaned.fillna(0)
    
    print('Are there still missing values:', df_cleaned.isna().any().any()) 
    
    #alle Spalten löschen die mehr als  2/3 Blank angaben haben
    df_cleaned=blank(df_cleaned)
    #allen skip angaben die Nummer 99 zuordnern
    df_cleaned=skip99(df_cleaned)
    
    #hier noch drüber nachdenken ob NANs und blanks dasselbe sind und ob man das so löschen möchte

    print('+++++Shape finales Dataset+++++')
    print(df_cleaned.shape) #(42739, 1632) verbliebenes Dataset davor (56136, 1756): 13397 rows and 124 columns deleted

    return df_cleaned

def replace_data(data_name):
    data_name['HALLUCEVR']=data_name['HALLUCEVR'].replace(91,2)
    data_name['INHALEVER']=data_name['INHALEVER'].replace(91,2)
    data_name['CRKEVER']=data_name['CRKEVER'].replace(91,2)
    data_name['PNRANYLIF']=data_name['PNRANYLIF'].replace(5,1)
    data_name['TRQANYLIF']=data_name['TRQANYLIF'].replace(5,1)
    data_name['STMANYLIF']=data_name['STMANYLIF'].replace(5,1)
    data_name['SEDANYLIF']=data_name['SEDANYLIF'].replace(5,1)
    data_name['PNRNMLIF']=data_name['PNRNMLIF'].replace(5,1)
    data_name['TRQNMLIF']=data_name['TRQNMLIF'].replace(5,1)
    data_name['STMNMLIF']=data_name['STMNMLIF'].replace(5,1)
    data_name['SEDNMLIF']=data_name['SEDNMLIF'].replace(5,1)
    data_name['PNRNMLIF']=data_name['PNRNMLIF'].replace(91,2)
    data_name['TRQNMLIF']=data_name['TRQNMLIF'].replace(91,2)
    data_name['STMNMLIF']=data_name['STMNMLIF'].replace(91,2)
    data_name['SEDNMLIF']=data_name['SEDNMLIF'].replace(91,2)
    
data=pd.read_csv('../data/NSDUH-2019.tsv', sep='\t')
df_cleaned=clean_data(data)

#Visualisieren
#Ob jemals Drogen konsumiert wurden bzw. Abhänigkeit von Drogen
drug_data=df_cleaned[['CIGEVER','ALCEVER','MJEVER','COCEVER','CRKEVER','HEREVER','HALLUCEVR','INHALEVER','METHAMEVR','PNRANYLIF','TRQANYLIF','STMANYLIF','SEDANYLIF','PNRNMLIF','TRQNMLIF','STMNMLIF','SEDNMLIF']]
replace_data(drug_data)
drug_data_with_ment = df_cleaned[['CIGEVER','ALCEVER','MJEVER','COCEVER','CRKEVER','HEREVER','HALLUCEVR','INHALEVER','METHAMEVR','PNRANYLIF','TRQANYLIF','STMANYLIF','SEDANYLIF','PNRNMLIF','TRQNMLIF','STMNMLIF','SEDNMLIF','Mental_health_status']] 
replace_data(drug_data_with_ment) 

melted_drug_data=drug_data.melt(var_name='Column', value_name='Value')
print(len(melted_drug_data))
filtered_drug_data=melted_drug_data[melted_drug_data['Value'].isin([1,2])] #gibt hier auch noch andere wie don't know und refused
print(len(filtered_drug_data))

drug_dependet=data[['DNICNSP', 'DEPNDALC', 'DEPNDMRJ', 'DEPNDCOC', 'DEPNDHER', 'DEPNDPYHAL','DEPNDPYINH','DEPNDPYMTH', 'DEPNDPYPNR','DEPNDPYTRQ','DEPNDPYSTM','DEPNDPYSED', 'DEPNDPYPSY']]
drug_dependet=drug_dependet.replace(0,2)
melted_drug_depended=drug_dependet.melt(var_name='Column', value_name='Value')
filtered_drug_depended= melted_drug_depended[melted_drug_depended['Value'].isin([1,2])]

list_mean_mental_health_yes = []
list_mean_mental_health_no = []
count=0
for col in drug_data_with_ment.iloc[:, :-1]:
    mean_mental_health_yes = drug_data_with_ment.loc[drug_data_with_ment[col].isin([1]), 'Mental_health_status'].mean()
    mean_mental_health_no = drug_data_with_ment.loc[drug_data_with_ment[col].isin([2]), 'Mental_health_status'].mean()
    list_mean_mental_health_yes.append(mean_mental_health_yes)
    list_mean_mental_health_no.append(mean_mental_health_no)

ment_drug_yes = pd.DataFrame({'means':list_mean_mental_health_yes,'drugs':['CIGEVER','ALCEVER','MJEVER','COCEVER','CRKEVER','HEREVER','HALLUCEVR','INHALEVER','METHAMEVR','PNRANYLIF','TRQANYLIF','STMANYLIF','SEDANYLIF','PNRNMLIF','TRQNMLIF','STMNMLIF','SEDNMLIF']})
ment_drug_no = pd.DataFrame({'means':list_mean_mental_health_no,'drugs':['CIGEVER','ALCEVER','MJEVER','COCEVER','CRKEVER','HEREVER','HALLUCEVR','INHALEVER','METHAMEVR','PNRANYLIF','TRQANYLIF','STMANYLIF','SEDANYLIF','PNRNMLIF','TRQNMLIF','STMNMLIF','SEDNMLIF']})

#Plot allgemein Druge Usage
fig1= sns.histplot(data=filtered_drug_data, x='Column', hue='Value', multiple="stack") #wär hier auch noch cool vielleicht dont know und so zu sehen also vielleicht eher melted_drug_data statt die filtered version
plt.xticks(rotation=45)
plt.legend(title='Have you ever used...', labels=['Yes','No']) #hier irgendwie vertauscht siehe plot (alkohol macht zb keinen sinn dass ja und nein so)
plt.xlabel('Different drugs')
plt.ylabel('Number of people')
plt.tight_layout()
plt.savefig('../output/drug_use.png')
plt.close()

#Plot dependency
fig2= sns.countplot(data=filtered_drug_depended, x='Column',) #stimmt irgendwas nicht kann nicht sein dass alle leute drogen nehmen
plt.xticks(rotation=45)
plt.xlabel('Different drugs')
plt.ylabel('Number of people')
plt.tight_layout()
plt.savefig('../output/drug_depended_use.png')
plt.close()

#Plot Menatl Health
fig3= sns.countplot(data=df_cleaned, x='Mental_health_status')
#plt.xticks(rotation=45)
plt.xlabel('Mental Health Status')
plt.ylabel('Number of people')
plt.tight_layout()
plt.savefig('../output/mental_health_status.png')

#Plot mental health per drug
fig4, axs = plt.subplots(2)
sns.barplot(data=ment_drug_no,x='drugs',y='means', ax=axs[0]) #hier stimmt auch irgendwas nicht da der durchschnitt über 4 ist dabei geht die scala nur bis 3
axs[0].set_title('Mental Health if no drugs')
axs[0].tick_params(axis='x', rotation=45)


sns.barplot(data=ment_drug_yes,x='drugs',y='means', ax=axs[1]) #hier anders falsch
axs[1].set_title('Mental Health if drug taken')
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig('../output/mental_health_status_vs_drugtypes.png')
plt.close()



'''
**************************************************************************************************************************************
Feature Selection
**************************************************************************************************************************************
'''
#################################### CHI2 Test ###################################################
#chi test geht nicht wegen enormer sample size --> gibt unglaublich kleine p-values
def contingencytable(effect, intervention):
    global counter
    counter+=1
    contab=pd.crosstab(intervention, effect)
    return contab
def chisquare(df, values, label):
    d={}
    for var in values:
        table=contingencytable(df[label], df[var])
        stats=[sts.chi2_contingency(table).pvalue, sts.chi2_contingency(table).statistic]
        d[var]=stats #dictionary with p-value, statistic
    return d
def featureanalysis(d):
    SignificantFeatures=[]
    for key, value in d.items():
        if value[0]<SignificanceLevel:
            SignificantFeatures.append(key)
    return SignificantFeatures

#################################### Calculation of Mutual Information ###################################################
def mutualinfo(values, label):
    mutualinfoscore=mutual_info_classif(values,label, discrete_features=True, random_state=42)
    #d=dict(zip(values.columns, mutualinfoscore))
    return mutualinfoscore
'''
#hab was ausprobiert geht aber ewig und macht eig das selbe
def select_features(X_train, y_train, X_test):
    fs = SelectKBest(score_func=mutual_info_classif, k=15) # auch methode um mutual info zu berechnen dauert nur 5 min
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs

def sorted_features(fs, values):
    d = {}
    for var in values:
        d[var]=fs.scores_
    return d
'''

################################### feature selection based on common sense ##############################################
counter=0
for var in df_cleaned.columns:
    if len(df_cleaned[var].unique())<2:
        del df_cleaned[var]
        counter+=1
print(counter, "columns lost their meaning because of the deletion of rows with NANs in Mental health status")

"""
#split age in groups 
def split_age_groups(drug, df_cleaned):
    df_cleaned.loc[df_cleaned[drug] <= 21, drug] = 1
    df_cleaned.loc[(21<= df_cleaned[drug])&(df_cleaned[drug] <= 50), drug] = 2
    df_cleaned.loc[df_cleaned[drug] >50, drug] = 3
    return df_cleaned

#Age/Year/Month when first smoked a cigarette --> redundant data with cigever
df_cleaned=split_age_groups('CIGTRY', df_cleaned)
df_cleaned.drop(['CIGYFU', 'CIGMFU'], axis=1)
#CIGREC is redundant due to CIG30USE and CG30ST due to CIG30AV and CIG30BR2 due to CIG30TPE ...
df_cleaned.drop(['CIGREC', 'CG30EST','CIGAGE','CIGDLYFU','CIGDLMFU','SMKLSSYFU','SMKLSSMFU', 'SMKLSSREC', 'SMKLSS30N', 'CIGARYFU', 'CIGARMFU', 'CIGARREC', 'CGR30USE'], axis=1)
df_cleaned=split_age_groups('CIGARTRY', df_cleaned)
df_cleaned=split_age_groups('SMKLSSTRY', df_cleaned)
#Alcohol
df_cleaned=split_age_groups('ALCTRY', df_cleaned)
df_cleaned.drop(['ALCYFU', 'ALCMFU', 'ALCYRTOT'], axis=1)
#ALBSTWAY feature engineering
df_cleaned.loc[df_cleaned['ALBSTWAY'] == 11, 'ALBSTWAY'] = 1
df_cleaned.loc[df_cleaned['ALBSTWAY'] == 12, 'ALBSTWAY'] = 2
df_cleaned.loc[df_cleaned['ALBSTWAY'] == 13, 'ALBSTWAY'] = 3
print(df_cleaned['ALBSTWAY'].unique())
df_cleaned.drop(df_cleaned.loc[:,'ALDAYPYR':'ALCUS30D'].columns, axis=1)
#ALCBNG30D feature engineering in Are there days with 5 drinks or more in the past 30 days
df_cleaned.loc[df_cleaned['ALCBNG30D'] < 31, 'ALCBNG30D'] = 1
df_cleaned.loc[df_cleaned['ALCBNG30D'] == 80, 'ALCBNG30D'] = 2
#Marijuana
df_cleaned=split_age_groups('MJAGE', df_cleaned)
df_cleaned.drop(df_cleaned.loc[:,'MJYFU':'MJMFU'].columns, axis=1)
df_cleaned.drop(df_cleaned.loc[:,'MJYRTOT':'MJFQFLG'].columns, axis=1)
df_cleaned.loc[df_cleaned['MRBSTWAY'] == 11, 'MRBSTWAY'] = 1
df_cleaned.loc[df_cleaned['MRBSTWAY'] == 12, 'MRBSTWAY'] = 2
df_cleaned.loc[df_cleaned['MRBSTWAY'] == 13, 'MRBSTWAY'] = 3
df_cleaned.drop(df_cleaned.loc[:,'MRDAYPYR':'MJDAY30A'].columns, axis=1)
#Cocaine
df_cleaned=split_age_groups('COCAGE', df_cleaned)
df_cleaned.drop(df_cleaned.loc['COCYFU':'CCFQFLG'].columns, axis=1)
df_cleaned.loc[df_cleaned['CCBSTWAY'] == 11, 'CCBSTWAY'] = 1
df_cleaned.loc[df_cleaned['CCBSTWAY'] == 12, 'CCBSTWAY'] = 2
df_cleaned.loc[df_cleaned['CCBSTWAY'] == 13, 'CCBSTWAY'] = 3
df_cleaned.loc[df_cleaned['CCBSTWAY'] == 21, 'CCBSTWAY'] = 1
df_cleaned.loc[df_cleaned['CCBSTWAY'] == 22, 'CCBSTWAY'] = 2
df_cleaned.loc[df_cleaned['CCBSTWAY'] == 23, 'CCBSTWAY'] = 3
df_cleaned.drop(df_cleaned.loc['CCDAYPYR':'CCDAYPWK'].columns, axis=1)

df_cleaned.loc[df_cleaned['COCUS30A'] < 31, 'COCUS30A'] = 1 #COCUS30A changed in consumed in last 30 days

df_cleaned.drop('CC30EST')
#Crack
df_cleaned=split_age_groups('CRKAGE', df_cleaned)
df_cleaned.drop(df_cleaned.loc['CRKYFU':'CRKMFU'].columns, axis=1)
df_cleaned.drop(df_cleaned.loc['CRKYRTOT':'CRFQFLG'].columns, axis=1)
df_cleaned.drop(df_cleaned['CRDAYPYR':'CRDAYPWK'].columns, axis=1)
df_cleaned.loc[df_cleaned['CRKUS30A'] < 31, 'CRKUS30A'] = 1 #changed in consumed in last 30 days
"""


################################### feature selection based on statistics ################################################
X  = df_cleaned.loc[:,'IRCIGRC':'SRCCLFRSTM']
Y  = df_cleaned['Mental_health_status']

SignificanceLevel=0.05 # brauchts dann auch nich mehr wegen chisquare
#valuesofinterest=X.columns # mit split nicht mehr gebraucht
#ChiSquare implementation: ist müll weil dataset viel zu gross
#chi2=chisquare(df_cleaned, valuesofinterest, 'Mental_health_status')
#relevfeatures=featureanalysis(chi2)
################################### Splitting in Test and Train Set ######################################################
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1) # split bei so grossem dataset gebraucht (siehe lesezeichen chrome) #was macht random state hier? wieder 42? checke leider nicht was die zahl genau macht

################################### Cross Validation ######################################################
#hab ich von HW5 übernommen und auf unseres überführt
n_splits = 10 #bei so grossem dataset sollte man mehr splits verwenden um overfitting und bias zu vermeiden
skf      = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state=0)

mutinfolist = []
for train_i, test_i in skf.split(X,Y):
    X_train, X_test = X.iloc[train_i], X.iloc[test_i]
    y_train, y_test = Y.iloc[train_i], Y.iloc[test_i]
    mutinfo=mutualinfo(X_train, y_train)
    mutinfolist.append(mutinfo) # liste wo mutual info gespreichert wird


mutinfomatrix = np.array(mutinfolist) #macht nen numpy array draus
std = np.std(mutinformatrix, axis=0) #berechnung standardabweichung über folds
average_mutinfo = np.mean(mutinfomatrix, axis=0) #mean über folds
'''
for i in range(len(mutinfomatrix[1])): #berechnung standardabweichung über folds
    std.append(np.std(mutinfomatrix[:, i]))
'''

feature_importance_df = pd.DataFrame({ #dataframe zum benutzen für sorting und plot
    'Feature': X_train.columns,
    'Average Coefficient': average_mutinfo,
    'Error': std
})

#das was ich ausprobiert hab und ewig geht, können wa sonst auch wieder löschen
"""X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)

d_sorted = dict(sorted(d.items(), key=lambda item: item[1]))
sorted = sorted_features(fs, valuesofinterest)
top_15 = list(sorted.items())[:15]
for i in range(len(top_15)):
    print('Feature %d: %f' % (i, top_15[i]))"""



#sortieren von values in absteigend reihenfolge
feature_importance_df.sort_values(by='Average Coefficient', ascending=False, inplace=True) #value sort
feature_importance_df_top = feature_importance_df.head(20) #top 20 features
'''
feature_coeff_list=list(feature_importance_df['Average Coefficient'])
diff=0
index=0
secdiff=0
tridiff=0
fodiff=0
foind=0

for i in range(3, len(feature_coeff_list)):
    mdiff=feature_coeff_list[i-1]-feature_coeff_list[i]
    if mdiff>diff:
        secdiff=diff
        diff=mdiff
        index=i
    elif mdiff>secdiff:
        tridiff=secdiff
        secdiff=mdiff
        secind=i
    elif mdiff>tridiff:
        tridiff=mdiff
        triind=i
    elif mdiff>fodiff:
        fodiff=mdiff
        foind=i

print(diff, secdiff, tridiff, fodiff, index, secind, triind, foind)
'''

    


#sortierung mit dictionary (is ohne folds, können wa wenn wir lieber das benutzen möchten anpassen)
"""multiinfsorted=dict(sorted(mutinfo.items(), key=lambda x:x[1], reverse=True))
multkey=list(multiinfsorted.keys())[:30]
multval=list(multiinfsorted.values())[:30]"""

#plot mit dataframe erstellt
plt.close()
figure = plt.barh(
    feature_importance_df_top['Feature'], 
    feature_importance_df_top['Average Coefficient'], 
    xerr = feature_importance_df_top['Error']
    )

plt.xlabel("Averaged Mutual Information")
plt.ylabel("Top features")
plt.title("Feature importance of top 20 features normalized across 10 folds")
plt.tight_layout()
plt.savefig('../output/importance.png')
'''
**************************************************************************************************************************************
Handling imbalanced data
**************************************************************************************************************************************
'''
################################### random oversampling ################################################
def randomoversampling(x,y,sampling_strategy):
    resample=SMOTE(sampling_strategy=sampling_strategy)
    x_resampled, y_resampled = resample.fit_resample(x, y)
    return x_resampled, y_resampled
################################### random undersampling ################################################
def randomundersampling(x,y,sampstrat):
    undersampler = RandomUnderSampler(sampling_strategy=sampstrat, random_state=42)
    x_resampled, y_resampled = undersampler.fit_resample(x, y)
    return x_resampled, y_resampled

d={0.0 : y_train.value_counts()[0.0], 1.0:20000, 2.0:18000, 3.0:19000}
x_resampled, y_resampled=randomoversampling(X_train, y_train, d) #majority class bleibt gleich nur alle anderen werden mehr
x_resampled, y_resampled=randomundersampling(x_resampled, y_resampled, "auto") #bisschen was von majority class wird von 28746 auf etwa 17500 reduziert


plt.close()
fig= sns.countplot( x=y_resampled)
#plt.xticks(rotation=45)
plt.xlabel('Mental Health Status balanced')
plt.ylabel('Number of people')
plt.tight_layout()
plt.savefig('../output/mental_health_status_balanced.png')


print('Fertig')
