import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as sts
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.model_selection import StratifiedKFold, train_test_split
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler 
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
#ihr müsst evt. noch imblearn runterladen (pip install imbalanced-learn)
import warnings


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
    
    #Alle recorded Fragen zu Drogenkonsum
    df_imputed_drug_use = data.loc[:,'IRCIGRC':'SRCCLFRSED']
    df_recorded_special_drug = data.loc[:,'ANYNDLREC':'GHBMONR']
    df_recorded_risk = data.loc[:,'GRSKCIGPKD':'APPDRGMON2']
    df_recorded_drug_dependence = data.loc[:,'IRCGIRTB':'AUDNODUD']
    df_recorded_drug_treatment = data.loc[:,'TXEVRRCVD2':'NDTRNMIMPT']
    df_recorded_alcohol = data.loc[:,'UADPEOP':'KRATMON']

    df_new=pd.concat([df_imputed_drug_use, df_recorded_special_drug, df_recorded_risk, df_recorded_drug_dependence, df_recorded_drug_treatment, df_recorded_alcohol], axis=1)
    
    #Alles inklusive Mental Health Einteilung
    df_new['Mental_health_status'] = data['MI_CAT_U'] 

    #Alle Spalten woher die Antworten kommen (questionair, imputed,...) droppen (114 Spalten) 
    #ANYDLREC gibts nicht
    df_new.drop(columns=[
        'IICIGRC', 'II2CIGRC', 'IICGRRC', 'II2CGRRC', 'IIPIPLF', 'IIPIPMN', 'IISMKLSSREC', 'IIALCRC', 'II2ALCRC', 'IIMJRC', 'II2MJRC', 'IICOCRC', 
        'II2COCRC', 'IICRKRC', 'II2CRKRC', 'IIHERRC', 'II2HERRC', 'IIHALLUCREC', 'IILSDRC', 'II2LSDRC', 'IIPCPRC', 'II2PCPRC', 'IIECSTMOREC', 'IIDAMTFXREC', 
        'IISALVIAREC', 'IIINHALREC', 'IIMETHAMREC', 'IIPNRANYREC', 'IIOXCNANYYR', 'IITRQANYREC', 'IISTMANYREC', 'IISEDANYREC', 'IIPNRNMREC', 'IIOXCNNMYR', 
        'IITRQNMREC', 'IISTMNMREC', 'IISEDNMREC', 'IIALCFY', 'II2ALCFY', 'IIMJFY',  'II2MJFY', 'IICOCFY', 'II2COCFY', 'IICRKFY', 'II2CRKFY', 'IIHERFY', 'II2HERFY',
        'IIHALLUCYFQ', 'IIINHALYFQ', 'IIMETHAMYFQ', 'IICIGFM', 'II2CIGFM', 'IICGRFM', 'II2CGRFM', 'IISMKLSS30N', 'IIALCFM', 'II2ALCFM', 'IIALCBNG30D', 'IIMJFM', 
        'II2MJFM', 'IICOCFM', 'II2COCFM', 'IICRKFM', 'II2CRKFM', 'IIHERFM', 'II2HERFM', 'IIHALLUC30N', 'IIINHAL30N', 'IIMETHAM30N', 'IIPNRNM30FQ', 'IITRQNM30FQ',
        'IISTMNM30FQ', 'IISEDNM30FQ', 'IICIGAGE', 'IICIGYFU', 'IICDUAGE', 'IICD2YFU', 'IICGRAGE', 'IICGRYFU', 'IISMKLSSTRY', 'IISMKLSSYFU', 'IIALCAGE', 'IIALCYFU',
        'IIMJAGE', 'IIMJYFU', 'IICOCAGE', 'IICOCYFU', 'IICRKAGE', 'IICRKYFU', 'IIHERAGE', 'IIHERYFU', 'IIHALLUCAGE', 'IIHALLUCYFU', 'IILSDAGE', 'IILSDYFU', 'IIPCPAGE',
        'IIPCPYFU', 'IIECSTMOAGE', 'IIECSTMOYFU', 'IIINHALAGE', 'IIINHALYFU', 'IIMETHAMAGE', 'IIMETHAMYFU', 'IIPNRNMINIT', 'IITRQNMINIT', 'IISTMNMINIT', 'IISEDNMINIT',
        'IIPNRNMYFU', 'IIPNRNMAGE', 'IITRQNMYFU', 'IITRQNMAGE', 'IISTMNMYFU', 'IISTMNMAGE', 'IISEDNMYFU', 'IISEDNMAGE', 'CHMNDLREC', 'NDSSANSP', 'UDPYHRPNR',
        'NDTXEFTALC', 'NDTXEFTILL', 'NDTXEFILAL', 'UADPEOP', 'UADOTSP', 'UADPLACE', 'UADCAG', 'UADFWHO', 'UADBUND', 'UADFRD'
        ], inplace=True)


    #Alle leere Einträge füllen die eigentlich Nein sein sollten
    #'CADRKMATH2', 'CAMHRPROB2' gibts auch nicht
    df_new[[
        'CIGAVGD', 'CIGAVGM', 'ALCNUMDKPM', 'SRCPNRNM2', 'SRCSTMNM2', 'SRCSEDNM2', 'SRCFRPNRNM', 'SRCFRTRQNM', 
        'SRCFRSTMNM', 'SRCFRSEDNM', 'SRCCLFRPNR', 'SRCCLFRTRQ', 'SRCCLFRSTM', 'SRCCLFRSED', 'GRSKCIGPKD', 'GRSKMRJMON', 
        'GRSKMRJWK', 'GRSKCOCMON', 'GRSKCOCWK', 'GRSKHERTRY', 'GRSKHERWK', 'GRSKLSDTRY', 'GRSKLSDWK', 'GRSKBNGDLY', 
        'GRSKBNGWK', 'DIFOBTMRJ', 'DIFOBTCOC', 'DIFOBTCRK', 'DIFOBTHER', 'DIFOBTLSD', 'APPDRGMON2', 'NDTRNNOCOV', 'NDTRNNOTPY',
        'NDTRNTSPHR', 'NDTRNWANTD', 'NDTRNNSTOP', 'NDTRNPFULL', 'NDTRNDKWHR', 'NDTRNNBRNG', 'NDTRNJOBNG', 'NDTRNNONED', 'NDTRNHANDL',
        'NDTRNNOHLP', 'NDTRNNTIME', 'NDTRNFNDOU', 'NDTRNMIMPT', 'UADCAR', 'UADHOME', 'UADOTHM', 'UADPUBL', 'UADBAR', 'UADEVNT',
        'UADSCHL', 'UADROTH', 'UADPAID', 'UADMONY', 'UADBWHO', 'CADRKMARJ2', 'CADRKCOCN2', 'CADRKHERN2', 'CADRKHALL2', 'CADRKINHL2',
        'CASUPROB2', 'RCVYSUBPRB', 'RCVYMHPRB', 'ALMEDYR2', 'OPMEDYR2', 'ALOPMEDYR', 'KRATFLG', 'KRATYR', 'KRATMON' 
    ]] = df_new[[
        'CIGAVGD', 'CIGAVGM', 'ALCNUMDKPM', 'SRCPNRNM2', 'SRCSTMNM2', 'SRCSEDNM2', 'SRCFRPNRNM', 'SRCFRTRQNM', 
        'SRCFRSTMNM', 'SRCFRSEDNM', 'SRCCLFRPNR', 'SRCCLFRTRQ', 'SRCCLFRSTM', 'SRCCLFRSED', 'GRSKCIGPKD', 'GRSKMRJMON', 
        'GRSKMRJWK', 'GRSKCOCMON', 'GRSKCOCWK', 'GRSKHERTRY', 'GRSKHERWK', 'GRSKLSDTRY', 'GRSKLSDWK', 'GRSKBNGDLY', 
        'GRSKBNGWK', 'DIFOBTMRJ', 'DIFOBTCOC', 'DIFOBTCRK', 'DIFOBTHER', 'DIFOBTLSD', 'APPDRGMON2', 'NDTRNNOCOV', 'NDTRNNOTPY',
        'NDTRNTSPHR', 'NDTRNWANTD', 'NDTRNNSTOP', 'NDTRNPFULL', 'NDTRNDKWHR', 'NDTRNNBRNG', 'NDTRNJOBNG', 'NDTRNNONED', 'NDTRNHANDL',
        'NDTRNNOHLP', 'NDTRNNTIME', 'NDTRNFNDOU', 'NDTRNMIMPT', 'UADCAR', 'UADHOME', 'UADOTHM', 'UADPUBL', 'UADBAR', 'UADEVNT',
        'UADSCHL', 'UADROTH', 'UADPAID', 'UADMONY', 'UADBWHO', 'CADRKMARJ2', 'CADRKCOCN2', 'CADRKHERN2', 'CADRKHALL2', 'CADRKINHL2',
        'CASUPROB2', 'RCVYSUBPRB', 'RCVYMHPRB', 'ALMEDYR2', 'OPMEDYR2', 'ALOPMEDYR', 'KRATFLG', 'KRATYR', 'KRATMON'
    ]].fillna(0)
    
    #Hier ist 0 schon besetzt deswegen habe ich die Nein einträge provisorisch auf 3 geändert
    df_new['CIG1PACK']=df_new['CIG1PACK'].fillna(3)

    #Die wenigen Einträge mit missing Values droppen (581 Reihen gelöscht, 55555 Reihen übrig)
    df_new =df_new.dropna(subset=['PNRMAINRSN', 'TRQMAINRSN', 'STMMAINRSN', 'SEDMAINRSN'])

    #Alle Zeilen löschen mit NA in Mental Health !!wie viele? brauchen wir auch zum argumentieren für die Arbeit!!
    df_cleaned = df_new.dropna(subset=['Mental_health_status'])
    print('Number of deleted rows:', len(df_new)-len(df_cleaned))

    #Übersicht neues Dataset
    print('Shape of dataset: ',df_new.shape)
    print('++++Columns Index++++')
    print(df_new.columns)
    print('++++Datentypen++++')
    print(df_new.dtypes)
    print('+++++Head of Dataset+++++')
    print(df_new.head(10))
    print('Number of duplicated rows:',df_new.duplicated().sum()) #duplicate rows

    missing_values_per_column = df_cleaned.isna().sum().sort_values(ascending=False)
    print('++++Number of missing values per Column++++')
    print(missing_values_per_column)

    num_columns_not_zero = sum([1 for value in missing_values_per_column if value > 0])
    print("Number of columns with missing values:", num_columns_not_zero) #100 von 1756 Spalten enthalten leere Zeilen 
    #df_cleaned=df_cleaned.dropna(axis=1) #Drop alle Spalten, in denen Werte fehlen
    df_cleaned= df_cleaned.fillna(0)
    
    print('Are there still missing values:', df_cleaned.isna().any().any()) 
    print('+++++Shape finales Dataset+++++')
    print(df_cleaned.shape) #(42739, 1632) verbliebenes Dataset davor (56136, 1756): 13397 rows and 124 columns deleted

    return df_cleaned

def replace_data(data_name):
    data_name[['HALLUCEVR', 'INHALEVER', 'CRKEVER', 'PNRNMLIF', 'TRQNMLIF', 'STMNMLIF', 'SEDNMLIF']]=data_name[['HALLUCEVR', 'INHALEVER', 'CRKEVER', 'PNRNMLIF', 'TRQNMLIF', 'STMNMLIF', 'SEDNMLIF']].replace(91,2)
    data_name[['PNRANYLIF', 'TRQANYLIF', 'STMANYLIF', 'SEDANYLIF','PNRNMLIF', 'TRQNMLIF', 'STMNMLIF', 'SEDNMLIF']]=data_name[['PNRANYLIF', 'TRQANYLIF', 'STMANYLIF', 'SEDANYLIF','PNRNMLIF', 'TRQNMLIF', 'STMNMLIF', 'SEDNMLIF']].replace(5,1)
    #data_name[['NDFLTXILAL', 'NDFLTXILL', 'NDFLTXALC', 'UADCAR', 'UADHOME', 'UADOTHM', 'UADPUBL', 'UADBAR', 'UADEVNT', 'UADSCHL', 'UADROTH']]=data_name[['NDFLTXILAL', 'NDFLTXILL', 'NDFLTXALC', 'UADCAR', 'UADHOME', 'UADOTHM', 'UADPUBL', 'UADBAR', 'UADEVNT', 'UADSCHL', 'UADROTH']].replace(2,0)
    #data_name[['UADPAID', 'UADMONY', 'UADBWHO']]=data_name[['UADPAID', 'UADMONY', 'UADBWHO']].replace(3,2)
    #data_name[['UADPAID', 'UADMONY', 'UADBWHO']]=data_name[['UADPAID', 'UADMONY', 'UADBWHO']].replace(2,0)

data=pd.read_csv('../data/NSDUH-2019.tsv', sep='\t', index_col=0)
df_cleaned=clean_data(data)

'''
**************************************************************************************************************************************
Visualiseren
**************************************************************************************************************************************
'''
#Ob jemals Drogen konsumiert wurden bzw. Abhänigkeit von Drogen
drug_data=data[['CIGEVER','ALCEVER','MJEVER','COCEVER','CRKEVER','HEREVER','HALLUCEVR','INHALEVER','METHAMEVR','PNRANYLIF','TRQANYLIF','STMANYLIF','SEDANYLIF','PNRNMLIF','TRQNMLIF','STMNMLIF','SEDNMLIF']]
replace_data(drug_data)

melted_drug_data=drug_data.melt(var_name='Column', value_name='Value')
filtered_drug_data=melted_drug_data[melted_drug_data['Value'].isin([1,2])] #gibt hier auch noch andere wie don't know und refused

#Plot allgemein Druge Usage
fig1= sns.histplot(data=filtered_drug_data, x='Column', hue='Value', multiple="stack") #wär hier auch noch cool vielleicht dont know und so zu sehen also vielleicht eher melted_drug_data statt die filtered version
plt.xticks(rotation=45)
plt.legend(title='Have you ever used...', labels=['No','Yes']) 
plt.xlabel('Different drugs')
plt.ylabel('Number of people')
plt.tight_layout()
plt.savefig('../output/drug_use.png')
plt.close()

#Plot Menatl Health
category_mapping = {
    0.0: 'No MI',
    1.0: 'Mild MI',
    2.0: 'Moderate MI',
    3.0: 'Serious MI'
}

fig2= sns.countplot(data=df_cleaned, x='Mental_health_status')
fig2.set_xticklabels([category_mapping[float(label.get_text())] for label in fig2.get_xticklabels()])
plt.xlabel('Mental Health Status')
plt.ylabel('Number of people')
plt.tight_layout()
plt.savefig('../output/mental_health_status.png')
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
def featureanalysischi(d):
    SignificantFeatures=[]
    for key, value in d.items():
        if value[0]<SignificanceLevel:
            SignificantFeatures.append(key)
    return SignificantFeatures

#################################### Calculation of Mutual Information ###################################################
def mutualinfo(values, label):
    mutualinfoscore=mutual_info_classif(values,label, discrete_features=True, random_state=42)
    return mutualinfoscore


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
X  = df_cleaned.drop(columns=['Mental_health_status'])
Y  = df_cleaned['Mental_health_status']

''' CHI IMPLEMENTATION
SignificanceLevel=0.05 
valuesofinterest=X.columns 
chi2=chisquare(df_cleaned, valuesofinterest, 'Mental_health_status')
relevfeatures=featureanalysischi(chi2)
'''
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
    mutinfolist.append(mutinfo) 


mutinfomatrix = np.array(mutinfolist) 
std = np.std(mutinfomatrix, axis=0) #berechnung standardabweichung über folds
average_mutinfo = np.mean(mutinfomatrix, axis=0) #mean über folds

feature_importance_df = pd.DataFrame({ 
    'Feature': X_train.columns,
    'Average Mutual Information': average_mutinfo,
    'Error': std
})

#sort
feature_importance_df.sort_values(by='Average Mutual Information', ascending=False, inplace=True) 
feature_importance_df_top = feature_importance_df.head(20) #top 20 features
important_features=feature_importance_df.head(100)
df_features_selected=df_cleaned[list(important_features['Feature'])]

#neuer split
X  = df_features_selected
Y  = df_cleaned['Mental_health_status']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
'''
feature_coeff_list=list(feature_importance_df['Average Mutual Information'])
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

    

'''
#sortierung mit dictionary chi
"""multiinfsorted=dict(sorted(mutinfo.items(), key=lambda x:x[1], reverse=True))
multkey=list(multiinfsorted.keys())[:30]
multval=list(multiinfsorted.values())[:30]"""
'''

#plot mit dataframe erstellt
plt.close()
figure = plt.barh(
    feature_importance_df_top['Feature'], 
    feature_importance_df_top['Average Mutual Information'], 
    xerr = feature_importance_df_top['Error']
    )

plt.xlabel("Averaged Mutual Information")
plt.ylabel("Top features")
plt.title("Feature importance of top 20 features averaged across 10 folds")
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
X_train, y_train=randomundersampling(x_resampled, y_resampled, "auto") #bisschen was von majority class wird von 28746 auf etwa 17500 reduziert


plt.close()
fig= sns.countplot( x=y_resampled)
#plt.xticks(rotation=45)
plt.xlabel('Mental Health Status balanced')
plt.ylabel('Number of people')
plt.tight_layout()
plt.savefig('../output/mental_health_status_balanced.png')


print('Fertig')
