import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as sts
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV, RandomizedSearchCV
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler 
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
#ihr müsst evt. noch imblearn runterladen (pip install imbalanced-learn)
import warnings


warnings.filterwarnings("ignore")

#teilweise hat nans die bedeuten die leute nehmen keine drogen... können wir also doch nicht löschen zB SRCPNRNM2 gibt noch viele dort ZB PNRMAINRSN
#beispiel statt replace df_cleaned.loc[df_cleaned['ALBSTWAY'] == 11, 'ALBSTWAY'] = 1

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
X_org=X
Y_org=Y

''' CHI IMPLEMENTATION
SignificanceLevel=0.05 
valuesofinterest=X.columns 
chi2=chisquare(df_cleaned, valuesofinterest, 'Mental_health_status')
relevfeatures=featureanalysischi(chi2)
'''
################################### Splitting in Test and Train Set ######################################################
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1) # split bei so grossem dataset gebraucht (siehe lesezeichen chrome) #was macht random state hier? wieder 42? checke leider nicht was die zahl genau macht
X_train_org, X_test_org, y_train_org, y_test_org = X_train, X_test, y_train, y_test
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
feature_importance_df = feature_importance_df.drop(feature_importance_df[feature_importance_df['Feature']=='CAMHPROB2'].index)
feature_importance_df = feature_importance_df.drop(feature_importance_df[feature_importance_df['Feature']=='RCVYMHPRB'].index)
feature_importance_df_top = feature_importance_df.head(20) #top 20 features
print(feature_importance_df.head(5))
important_features=feature_importance_df.head(100)
X_train_selected=X_train[list(important_features['Feature'])]
X_test_selected=X_test[list(important_features['Feature'])]
X_train=X_train_selected
X_test=X_test_selected

X  = df_cleaned[list(important_features['Feature'])]

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
plt.title("Top 20 features averaged across 10 folds")
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

X_train_ws, y_train_ws=X_train,y_train

d={0.0 : y_train.value_counts()[0.0], 1.0:20000, 2.0:18000, 3.0:19000}
x_resampled, y_resampled=randomoversampling(X_train, y_train, d) #majority class bleibt gleich nur alle anderen werden mehr
X_train, y_train=randomundersampling(x_resampled, y_resampled, "auto") #bisschen was von majority class wird von 28746 auf etwa 17500 reduziert


plt.close()

category_mapping = {
    0.0: 'No MI',
    1.0: 'Mild MI',
    2.0: 'Moderate MI',
    3.0: 'Serious MI'
}
fig3= sns.countplot( x=y_resampled)
fig3.set_xticklabels([category_mapping[float(label.get_text())] for label in fig3.get_xticklabels()])
#plt.xticks(rotation=45)
plt.xlabel('Mental Health Status balanced')
plt.ylabel('Number of people')
plt.tight_layout()
plt.savefig('../output/mental_health_status_balanced.png')

'''
**************************************************************************************************************************************
Models
**************************************************************************************************************************************
'''

################################### Performance Evaluation ################################################
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report, roc_curve, auc, roc_auc_score)
from sklearn.preprocessing import label_binarize

def eval_Performance(y_eval, X_eval, clf, clf_name='My Classifier'):
    # Vorhersagen
    y_pred = clf.predict(X_eval)
    try:
        y_pred_proba = clf.predict_proba(X_eval)
    except:
        y_pred_proba='none'
    
    # Evaluation
    accuracy = accuracy_score(y_eval, y_pred)
    precision = precision_score(y_eval, y_pred, average='weighted')
    recall = recall_score(y_eval, y_pred, average='weighted')
    f1 = f1_score(y_eval, y_pred, average='weighted')
    
    # ROC AUC für Multiclass
    y_eval_bin = label_binarize(y_eval, classes=range(len(clf.classes_)))
    try: 
        if y_pred_proba =='none':
            roc_auc= 'none'
        else:
            roc_auc = roc_auc_score(y_eval_bin, y_pred_proba, average='weighted', multi_class='ovr')
    except:
        roc_auc = roc_auc_score(y_eval_bin, y_pred_proba, average='weighted', multi_class='ovr')
    
    return accuracy, precision, recall, f1, roc_auc


df_performance = pd.DataFrame(columns = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'] )


################################### Support Vecor Machines################################################
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import Nystroem, RBFSampler, PolynomialCountSketch
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def polynomcountsk(X_train, y_train, X_test, y_test):
    n_runs=5
    N_COMPONENTS = [150, 250, 500] #larger than the number of features --> larger than 100

    for n in N_COMPONENTS:
        accuracysum = 0
        for i in range(n_runs):
            pipeline = make_pipeline(    
                PolynomialCountSketch(n_components=n, degree=4),
                SGDClassifier(loss="hinge", penalty="l2", max_iter=5, class_weight='balanced')
            )
            pipeline.fit(X_train, y_train)
            accuracysum += pipeline.score(X_test, y_test)
        accuracy=accuracysum/n_runs #mean of accuracy over runs
        print(f"Accuracy for {n} components: {accuracy}") #very bad accuracy for polynominal kernal aproximation between 27-36 % depending on dimension

def model(X_train, y_train, X_test, y_test, param_grid, ml, kernelaprox):
    if kernelaprox != 0:
        X_train = kernelaprox.fit_transform(X_train)
        X_test = kernelaprox.transform(X_test)

    # GridSearch with cross-validation
    scorer=make_scorer(scoring)
    cv_strategy = StratifiedKFold(n_splits=2)
    grid_search = GridSearchCV(ml, param_grid, cv=cv_strategy, scoring=scorer, n_jobs=-1)
    grid_search.fit(X_train, y_train)
     #5fold cross validation
    

    best_model = grid_search.best_estimator_
    

    print("Best model parameters:", grid_search.best_params_)
    print("Average of Acc,Pre:", grid_search.score(X_test, y_test))
    print("Average of Acc,Pre:", grid_search.score(X_train, y_train))
    
    return eval_Performance(y_test, X_test, best_model, clf_name='SGD Classifier with {kernelaprox} kernel aproximation'), eval_Performance(y_train, X_train, best_model, clf_name='SGD Classifier with {kernelaprox} kernel aproximation')

def scoring(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    return (accuracy+precision)/2

#scaling the data
sc=StandardScaler()
X_train_sc=sc.fit_transform(X_train_ws)
X_test_sc=sc.transform(X_test)


#linear SVM
param_grid = {
    'alpha': [0.000001, 0.00001, 0.0001, 0.001],
    'max_iter': [2750, 3000, 3500, 4000, 5000],
    'penalty': ['l2']
}
#svc=SVC(class_weight='balanced')
sgd=SGDClassifier(loss="hinge", class_weight='balanced')

print("linear SVM")
df_performance.loc['Linear SVM test',:],df_performance.loc['Linear SVM train',:]=model(X_train_sc, y_train_ws, X_test_sc, y_test, param_grid, sgd, 0)


#Nystroem aprox with hyperparameter tuning for SGD
nystroem = Nystroem(kernel= 'rbf', random_state=1, n_components=1000) #n_components=n_features
param_grid = {
    'alpha': [0.000001, 0.00001, 0.0001, 0.001],
    'max_iter': [2750, 3000, 3500, 4000, 5000],
    'penalty': ['l2']
}
sgd = SGDClassifier(loss="hinge", class_weight='balanced') #early stopping to terminate training when validation score is not improving hängt zusammen it max_iter
print("Nystroem (rbf) SVM")
df_performance.loc['Nystoem (rbf) SVM test',:],df_performance.loc['Nystoem (rbf) SVM  train',:]=model(X_train_sc, y_train_ws, X_test_sc, y_test, param_grid, sgd, nystroem)
#Best model parameters: {'alpha': 0.0001, 'max_iter': 1000, 'penalty': 'l2'}
#Model accuracy: 0.7250470809792844
#etwa 4 min

#RBF aprox with hyperparameter tuning for SGD
kernelaprox=RBFSampler(random_state=1, n_components=100, gamma='scale') 
param_grid = {
    'alpha': [0.000001, 0.00001, 0.0001, 0.001],
    'max_iter': [2750, 3000, 3500, 4000, 5000],
    'penalty': ['l2']
}
sgd = SGDClassifier(loss="hinge") #early stopping to terminate training when validation score is not improving hängt zusammen it max_iter
print("RBF Sampler SVM")
df_performance.loc['RBF Sampler SVM test',:],df_performance.loc['RBF Sampler SVM train',:]=model(X_train_sc, y_train_ws, X_test_sc, y_test, param_grid, sgd, kernelaprox)

#model(X_train_sc, y_train, X_test_sc, y_test, kernelaprox, param_grid, sgd)
#Best model parameters: {'alpha': 0.0001, 'max_iter': 1000, 'penalty': 'l2'}
#Model accuracy: 0.5065913370998116
# etwa 2 Minuten

#kernelaproximation with polynominal count sketch (without hyperparameter tuning)
print("Polynominal SVM")
polynomcountsk(X_train_sc, y_train_ws, X_test_sc, y_test)
#Accuracy for 150 components: 0.30018832391713746
#Accuracy for 250 components: 0.29279661016949154
#Accuracy for 500 components: 0.3126647834274953

print(df_performance)

################################### Logistic regression ################################################
from sklearn.linear_model import LogisticRegression

#scaling the data
sc=StandardScaler()
X_train_org=sc.fit_transform(X_train_org)
X_test_org=sc.transform(X_test_org)

           
#LR mit original Dataset
clf_LR = LogisticRegression(random_state=1)
clf_LR.fit(X_train_org, y_train_org)

#LR mit FS und class balancing
clf_LR_FS_OUS = LogisticRegression(random_state=1)
clf_LR_FS_OUS.fit(X_train_sc, y_train_sc)

#LR mit den vorhandenen Funktionen
clf2_LR = LogisticRegression(multi_class='multinomial', solver='saga', C=0.5, penalty='l1', class_weight='balanced')
clf2_LR.fit(X_train_org, y_train_org)

clf3_LR = LogisticRegression(multi_class='ovr', solver='saga', C=0.5, penalty='l1', class_weight='balanced')
clf3_LR.fit(X_train_org, y_train_org)
 
###################################  Random Forest #######################################################
from sklearn.ensemble import RandomForestClassifier

#hyperparameter tuning with randomSearchCV
param_distributions = {
    'n_estimators': [200, 300, 400, 500],
    'max_depth': [20, 30, 40, 50, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True],
    'max_leaf_nodes': [10, 50, 100, 200, 300, 400, 500]
}

clf_RF = RandomForestClassifier(random_state=0)

random_search = RandomizedSearchCV(
    estimator=clf_RF,
    param_distributions=param_distributions,
    cv=skf,
    n_iter=20,
    random_state=0, 
    n_jobs=-1, #use als CPU-Cores
    verbose=1, # minimal output
    scoring ='accuracy',
)

random_search.fit(X_train, y_train)
best_params = random_search.best_params_
print(f"Best parameters found: {best_params}")

best_rf = RandomForestClassifier(**best_params, random_state=0, n_jobs=-1)
best_rf.fit(X_train, y_train)

##################################### K-nearest neighbour ##############################################################
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

pca = PCA(n_components=0.95) 
X_train_pca = pca.fit_transform(X_train_sc)
X_test_pca = pca.transform(X_test_sc)

para_grid = {'n_neighbors': np.arange(1, 51)}

knn = KNeighborsClassifier()

grid_search= GridSearchCV(knn, para_grid, cv=n_splits)
grid_search.fit(X_train_pca, y_train)

print(f'Best n_neighbors: {grid_search.best_params_["n_neighbors"]}')

best_knn = grid_search.best_estimator_
'''
knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(X_train_sc, y_train)
'''
################################### Analysis Performance Metrics #######################################################
#Random Forest
df_performance.loc['RF (test)',:] = eval_Performance(y_test, X_test, best_rf, clf_name='Random Forest')
df_performance.loc['RF (train)',:] = eval_Performance(y_train, X_train, best_rf, clf_name='Random Forest (train)')

#Logisitc Regression
df_performance.loc['LR (test)',:] = eval_Performance(y_test_org, X_test_org, clf_LR, clf_name = 'LR')
df_performance.loc['LR (train)',:] = eval_Performance(y_train_org, X_train_org, clf_LR, clf_name = 'LR (train)')

df_performance.loc['LR (test, FS, OUS)',:] = eval_Performance(y_test_sc, X_test_sc, clf_LR_FS_OUS, clf_name = 'LR_FS_OUS')
df_performance.loc['LR (train, FS, OUS)',:] = eval_Performance(y_train_sc, X_train_sc, clf_LR_FS_OUS, clf_name = 'LR_FS_OUS (train)')

df_performance.loc['LR2 (test)',:] = eval_Performance(y_test_org, X_test_org, clf2_LR, clf_name = 'LR2')
df_performance.loc['LR2 (train)',:] = eval_Performance(y_train_org, X_train_org, clf2_LR, clf_name = 'LR2 (train)')

df_performance.loc['LR3 (test)',:] = eval_Performance(y_test_org, X_test_org, clf3_LR, clf_name = 'LR3')
df_performance.loc['LR3 (train)',:] = eval_Performance(y_train_org, X_train_org, clf3_LR, clf_name = 'LR3 (train)')

#Knearest Neighbors
df_performance.loc['KNN (test)',:]= eval_Performance(y_test, X_test_pca,knn,clf_name="K-nearest neighbor")
df_performance.loc['KNN (train)',:] = eval_Performance(y_train, X_train_pca,knn,clf_name="K-nearest neighbor (train)")

print(df_performance)

print('Fertig')
