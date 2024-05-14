import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as sts
import warnings

warnings.filterwarnings("ignore")

def cleandata(data):
    #Alle Fragen zu Drogenkonsum
    df_new = data.loc[:,'CIGEVER':'MMFGIVE']
    df_new['Mental_health_status'] = data['MI_CAT_U'] 

    count_skip=df_new.eq(99).sum(axis=1) #blank?, refused?, dont know?
    df_cleaned = df_new[count_skip <= 640] #der count bezieht sich auf wie viele nicht beantwortet wurden (mit 99 im dataset gekennzeichnet), also je kleiner desto besser

    #Alle Zeilen löschen mit NA in Mental Health !!wie viele? brauchen wir auch zum argumentieren für die Arbeit!!
    df_cleaned = df_new.dropna(subset=['Mental_health_status'])
    drug_data=df_cleaned[['CIGEVER','ALCEVER','MJEVER','COCEVER','CRKEVER','HEREVER','HALLUCEVR','INHALEVER','METHAMEVR','PNRANYLIF','TRQANYLIF','STMANYLIF','SEDANYLIF','PNRNMLIF','TRQNMLIF','STMNMLIF','SEDNMLIF']]
    return df_cleaned, drug_data
def contingencytable(effect, intervention):
    contab=pd.crosstab(intervention, effect)
    return contab
def chisquare(df, values, label):
    d={}
    for var in values:
        table=contingencytable(df[label], df[var])
        #print(table)
        stats=[sts.chi2_contingency(table).pvalue, sts.chi2_contingency(table).statistic]
        d[var]=stats
    return d
def featureanalysis(d):
    SignificantFeatures=[]
    for key, value in d.items():
        if value[0]<SignificanceLevel:
            SignificantFeatures.append(key)
    return SignificantFeatures



data=pd.read_csv('../data/NSDUH-2019.tsv', sep='\t')
df, dfdrugdata=cleandata(data)

#feature selection
#X  = df.copy().drop('Mental_health_status', axis = 1)
X  = dfdrugdata
Y  = df['Mental_health_status']
SignificanceLevel=0.05
#
valuesofinterest=X.columns
#dictionary with p-value, statistic
chi2=chisquare(df, valuesofinterest, 'Mental_health_status')
print(featureanalysis(chi2))




