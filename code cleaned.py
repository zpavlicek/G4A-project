import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler 

import warnings
warnings.filterwarnings("ignore")


def clean_data(data):
    
    #all recorded questions to drug use
    df_imputed_drug_use = data.loc[:,'IRCIGRC':'SRCCLFRSED']
    df_recorded_special_drug = data.loc[:,'ANYNDLREC':'GHBMONR']
    df_recorded_risk = data.loc[:,'GRSKCIGPKD':'APPDRGMON2']
    df_recorded_drug_dependence = data.loc[:,'IRCGIRTB':'AUDNODUD']
    df_recorded_drug_treatment = data.loc[:,'TXEVRRCVD2':'NDTRNMIMPT']
    df_recorded_alcohol = data.loc[:,'UADPEOP':'KRATMON']

    df_new=pd.concat([df_imputed_drug_use, df_recorded_special_drug, df_recorded_risk, df_recorded_drug_dependence, df_recorded_drug_treatment, df_recorded_alcohol], axis=1)
    
    #adding mental health status
    df_new['Mental_health_status'] = data['MI_CAT_U'] 

    #dropping irrelevant topics (questionair, imputed,...) (114 columns) 
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
        'NDTXEFTALC', 'NDTXEFTILL', 'NDTXEFILAL', 'UADPEOP', 'UADOTSP', 'UADPLACE', 'UADCAG', 'UADFWHO', 'UADBUND', 'UADFRD', 'CAMHPROB2', 'RCVYMHPRB'
        ], inplace=True)
    #making answers consistent
    df_new[[
        'CIGAVGD', 'CIGAVGM', 'ALCNUMDKPM', 'SRCPNRNM2', 'SRCSTMNM2', 'SRCSEDNM2', 'SRCFRPNRNM', 'SRCFRTRQNM', 
        'SRCFRSTMNM', 'SRCFRSEDNM', 'SRCCLFRPNR', 'SRCCLFRTRQ', 'SRCCLFRSTM', 'SRCCLFRSED', 'GRSKCIGPKD', 'GRSKMRJMON', 
        'GRSKMRJWK', 'GRSKCOCMON', 'GRSKCOCWK', 'GRSKHERTRY', 'GRSKHERWK', 'GRSKLSDTRY', 'GRSKLSDWK', 'GRSKBNGDLY', 
        'GRSKBNGWK', 'DIFOBTMRJ', 'DIFOBTCOC', 'DIFOBTCRK', 'DIFOBTHER', 'DIFOBTLSD', 'APPDRGMON2', 'NDTRNNOCOV', 'NDTRNNOTPY',
        'NDTRNTSPHR', 'NDTRNWANTD', 'NDTRNNSTOP', 'NDTRNPFULL', 'NDTRNDKWHR', 'NDTRNNBRNG', 'NDTRNJOBNG', 'NDTRNNONED', 'NDTRNHANDL',
        'NDTRNNOHLP', 'NDTRNNTIME', 'NDTRNFNDOU', 'NDTRNMIMPT', 'UADCAR', 'UADHOME', 'UADOTHM', 'UADPUBL', 'UADBAR', 'UADEVNT',
        'UADSCHL', 'UADROTH', 'UADPAID', 'UADMONY', 'UADBWHO', 'CADRKMARJ2', 'CADRKCOCN2', 'CADRKHERN2', 'CADRKHALL2', 'CADRKINHL2',
        'CASUPROB2', 'RCVYSUBPRB', 'ALMEDYR2', 'OPMEDYR2', 'ALOPMEDYR', 'KRATFLG', 'KRATYR', 'KRATMON', 'SRCTRQNM2' 
    ]] = df_new[[
        'CIGAVGD', 'CIGAVGM', 'ALCNUMDKPM', 'SRCPNRNM2', 'SRCSTMNM2', 'SRCSEDNM2', 'SRCFRPNRNM', 'SRCFRTRQNM', 
        'SRCFRSTMNM', 'SRCFRSEDNM', 'SRCCLFRPNR', 'SRCCLFRTRQ', 'SRCCLFRSTM', 'SRCCLFRSED', 'GRSKCIGPKD', 'GRSKMRJMON', 
        'GRSKMRJWK', 'GRSKCOCMON', 'GRSKCOCWK', 'GRSKHERTRY', 'GRSKHERWK', 'GRSKLSDTRY', 'GRSKLSDWK', 'GRSKBNGDLY', 
        'GRSKBNGWK', 'DIFOBTMRJ', 'DIFOBTCOC', 'DIFOBTCRK', 'DIFOBTHER', 'DIFOBTLSD', 'APPDRGMON2', 'NDTRNNOCOV', 'NDTRNNOTPY',
        'NDTRNTSPHR', 'NDTRNWANTD', 'NDTRNNSTOP', 'NDTRNPFULL', 'NDTRNDKWHR', 'NDTRNNBRNG', 'NDTRNJOBNG', 'NDTRNNONED', 'NDTRNHANDL',
        'NDTRNNOHLP', 'NDTRNNTIME', 'NDTRNFNDOU', 'NDTRNMIMPT', 'UADCAR', 'UADHOME', 'UADOTHM', 'UADPUBL', 'UADBAR', 'UADEVNT',
        'UADSCHL', 'UADROTH', 'UADPAID', 'UADMONY', 'UADBWHO', 'CADRKMARJ2', 'CADRKCOCN2', 'CADRKHERN2', 'CADRKHALL2', 'CADRKINHL2',
        'CASUPROB2', 'RCVYSUBPRB', 'ALMEDYR2', 'OPMEDYR2', 'ALOPMEDYR', 'KRATFLG', 'KRATYR', 'KRATMON', 'SRCTRQNM2'
    ]].fillna(0)
    
    df_new['CIG1PACK']=df_new['CIG1PACK'].fillna(3)

    #drop rows with missing values (581 rows deleted, 55555 rows left)
    df_new =df_new.dropna(subset=['PNRMAINRSN', 'TRQMAINRSN', 'STMMAINRSN', 'SEDMAINRSN'])

    #drop NA in mental health
    df_cleaned = df_new.dropna(subset=['Mental_health_status'])
    print('Number of deleted rows:', len(df_new)-len(df_cleaned))

    #overview cleaned dataset
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
    print("Number of columns with missing values:", num_columns_not_zero) 
    df_cleaned=df_cleaned.dropna(subset=['CADRKMETH2'])
    
    print('Are there still missing values:', df_cleaned.isna().any().any()) 
    print('+++++Shape finales Dataset+++++')
    print(df_cleaned.shape) 

    return df_cleaned

def replace_data(data_name):
    data_name[['HALLUCEVR', 'INHALEVER', 'CRKEVER', 'PNRNMLIF', 'TRQNMLIF', 'STMNMLIF', 'SEDNMLIF']]=data_name[['HALLUCEVR', 'INHALEVER', 'CRKEVER', 'PNRNMLIF', 'TRQNMLIF', 'STMNMLIF', 'SEDNMLIF']].replace(91,2)
    data_name[['PNRANYLIF', 'TRQANYLIF', 'STMANYLIF', 'SEDANYLIF','PNRNMLIF', 'TRQNMLIF', 'STMNMLIF', 'SEDNMLIF']]=data_name[['PNRANYLIF', 'TRQANYLIF', 'STMANYLIF', 'SEDANYLIF','PNRNMLIF', 'TRQNMLIF', 'STMNMLIF', 'SEDNMLIF']].replace(5,1)


data=pd.read_csv('../data/NSDUH-2019.tsv', sep='\t', index_col=0)
df_cleaned=clean_data(data)

'''
**************************************************************************************************************************************
Visualiseren
**************************************************************************************************************************************
'''
#ever consumed
drug_data=data[['CIGEVER','ALCEVER','MJEVER','COCEVER','CRKEVER','HEREVER','HALLUCEVR','INHALEVER','METHAMEVR','PNRANYLIF','TRQANYLIF','STMANYLIF','SEDANYLIF','PNRNMLIF','TRQNMLIF','STMNMLIF','SEDNMLIF']]
replace_data(drug_data)

melted_drug_data=drug_data.melt(var_name='Column', value_name='Value')
filtered_drug_data=melted_drug_data[melted_drug_data['Value'].isin([1,2])] 

#Plot generel Druge Usage
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

################################### feature selection based on statistics ################################################
X  = df_cleaned.drop(columns=['Mental_health_status'])
Y  = df_cleaned['Mental_health_status']
X_org=X
Y_org=Y

################################### Splitting in Test and Train Set ######################################################
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
X_train_org, X_test_org, y_train_org, y_test_org = X_train, X_test, y_train, y_test
################################### Cross Validation ######################################################
#inspired by Hw 05
n_splits = 10 
skf      = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state=0)

mutinfolist = []
for train_i, test_i in skf.split(X,Y):
    X_train, X_test = X.iloc[train_i], X.iloc[test_i]
    y_train, y_test = Y.iloc[train_i], Y.iloc[test_i]
    
    mutinfo=mutualinfo(X_train, y_train)
    mutinfolist.append(mutinfo) 


mutinfomatrix = np.array(mutinfolist) 
std = np.std(mutinfomatrix, axis=0) #standard diviation
average_mutinfo = np.mean(mutinfomatrix, axis=0) #mean

feature_importance_df = pd.DataFrame({ 
    'Feature': X_train.columns,
    'Average Mutual Information': average_mutinfo,
    'Error': std
})

#sort
feature_importance_df.sort_values(by='Average Mutual Information', ascending=False, inplace=True) 
feature_importance_df_top = feature_importance_df.head(20) #top 20 features
#Top 5 features
print(feature_importance_df.head(5))

important_features=feature_importance_df.head(100)
X_train_selected=X_train[list(important_features['Feature'])]
X_test_selected=X_test[list(important_features['Feature'])]
X_train=X_train_selected
X_test=X_test_selected

X  = df_cleaned[list(important_features['Feature'])]


#plot most important features
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
x_resampled, y_resampled=randomoversampling(X_train, y_train, d) #majority class stays the same the rest increases
X_train, y_train=randomundersampling(x_resampled, y_resampled, "auto") #majority class reduced from 28746 to 17500


plt.close()
#plotting balanced mental health
category_mapping = {
    0.0: 'No MI',
    1.0: 'Mild MI',
    2.0: 'Moderate MI',
    3.0: 'Serious MI'
}
fig3= sns.countplot( x=y_train)
fig3.set_xticklabels([category_mapping[float(label.get_text())] for label in fig3.get_xticklabels()])
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
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.preprocessing import label_binarize

def eval_Performance(y_eval, X_eval, clf, clf_name='My Classifier'):
    y_pred = clf.predict(X_eval)
    try: #needed because of SVM
        y_pred_proba = clf.predict_proba(X_eval)
    except:
        y_pred_proba='none'
    
    # Evaluation
    accuracy = accuracy_score(y_eval, y_pred)
    precision = precision_score(y_eval, y_pred, average='weighted')
    recall = recall_score(y_eval, y_pred, average='weighted')
    f1 = f1_score(y_eval, y_pred, average='weighted')
    
    # ROC AUC for Multiclass
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer

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
    print("Average of Acc,Pre,f1:", grid_search.score(X_test, y_test))
    print("Average of Acc,Pre,f1:", grid_search.score(X_train, y_train))
    
    return eval_Performance(y_test, X_test, best_model, clf_name='SGD Classifier with {kernelaprox} kernel aproximation'), eval_Performance(y_train, X_train, best_model, clf_name='SGD Classifier with {kernelaprox} kernel aproximation')

def scoring(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted')
    return (accuracy+precision+f1)/3

#scaling the data
sc=StandardScaler()
X_train_sc_ws=sc.fit_transform(X_train_ws) #without sampling
X_train_sc=sc.fit_transform(X_train) #with sampling
X_test_sc=sc.transform(X_test)


#linear SVM
param_grid = {
    'alpha': [0.000001, 0.00001, 0.0001, 0.001],
    'max_iter': [2750, 3000, 3500, 4000, 5000],
    'penalty': ['l2']
}
#linear SVM
sgd=SGDClassifier(loss="hinge", class_weight='balanced')
print("linear SVM")
df_performance.loc['Linear SVM test',:],df_performance.loc['Linear SVM train',:]=model(X_train_sc_ws, y_train_ws, X_test_sc, y_test, param_grid, sgd, 0)


#Nystroem aprox SVM with hyperparameter tuning for SGD
nystroem = Nystroem(kernel= 'rbf', random_state=1, n_components=100) #n_components=n_features
param_grid = {
    'alpha': [0.000001, 0.00001, 0.0001, 0.001],
    'max_iter': [2750, 3000, 3500, 4000, 5000],
    'penalty': ['l2']
}
sgd = SGDClassifier(loss="hinge", class_weight='balanced') 
print("Nystroem (rbf) SVM")
df_performance.loc['Nystoem (rbf) SVM test',:],df_performance.loc['Nystoem (rbf) SVM  train',:]=model(X_train_sc_ws, y_train_ws, X_test_sc, y_test, param_grid, sgd, nystroem)


#RBF aprox with hyperparameter tuning for SGD
kernelaprox=RBFSampler(random_state=1, gamma='scale', n_components=100) 
param_grid = {
    'alpha': [0.000001, 0.00001, 0.0001, 0.001],
    'max_iter': [2750, 3000, 3500, 4000, 5000],
    'penalty': ['l2']
}
sgd = SGDClassifier(loss="hinge", class_weight='balanced') 
print("RBF Sampler SVM")
df_performance.loc['RBF Sampler SVM test',:],df_performance.loc['RBF Sampler SVM train',:]=model(X_train_sc_ws, y_train_ws, X_test_sc, y_test, param_grid, sgd, kernelaprox)

#Nystroem aprox with hyperparameter tuning for SGD
degree=[2,3,4]
for n in degree:
    nystroem = Nystroem(kernel= 'poly', random_state=1, degree=n, n_components=100) #n_components=n_features
    param_grid = {
        'alpha': [0.000001, 0.00001, 0.0001, 0.001],
        'max_iter': [2750, 3000, 3500, 4000, 5000],
        'penalty': ['l2']
    }
    sgd = SGDClassifier(loss="hinge", class_weight='balanced') 
    print(f'Nystroem (poly{n}) SVM')
    df_performance.loc[f'Nystoem (poly{n}) SVM test',:],df_performance.loc[f'Nystoem (poly{n}) SVM  train',:]=model(X_train_sc_ws, y_train_ws, X_test_sc, y_test, param_grid, sgd, nystroem)


print(df_performance)

################################### Logistic regression ################################################
from sklearn.linear_model import LogisticRegression

#scaling the data
sc=StandardScaler()
X_train_org=sc.fit_transform(X_train_org)
X_test_org=sc.transform(X_test_org)
   
#LR with original dataset
clf1_LR = LogisticRegression(random_state=1)
clf1_LR.fit(X_train_org, y_train_org)

#LR with FS and class balancing
clf_LR_FS_OUS = LogisticRegression()
clf_LR_FS_OUS.fit(X_train_sc, y_train)

#LR with FS and classweight='balanced
clf_LR_FS=LogisticRegression(class_weight='balanced')
clf_LR_FS.fit(X_train_sc_ws, y_train_ws)

#LR with sklearn functions
clf_LR_Multi = LogisticRegression(multi_class='multinomial', solver='saga', C=0.5, penalty='l1', class_weight='balanced')
clf_LR_Multi.fit(X_train_org, y_train_org)

clf_LR_OVR = LogisticRegression(multi_class='ovr', solver='saga', C=0.5, penalty='l1', class_weight='balanced')
clf_LR_OVR.fit(X_train_org, y_train_org)
 
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
    'max_leaf_nodes': [800, 900, 1000, 1100, 1200]
}

clf_RF = RandomForestClassifier(random_state=0)

def search(clf_RF):
    random_search = RandomizedSearchCV(
        estimator=clf_RF,
        param_distributions=param_distributions,
        cv=skf,
        n_iter=20,
        random_state=0, 
        n_jobs=-1, #use als CPU-Cores
        verbose=1, # minimal output
        scoring='accuracy'
    )
    return random_search

def rf(X_train, y_train, clf_RF):
    random_search = search(clf_RF)
    random_search.fit(X_train, y_train)
    best_params = random_search.best_params_
    return X_train, y_train, best_params

X_train_wsam, y_train_wsam, best_params_wsam = rf(X_train, y_train, clf_RF)

best_rf = RandomForestClassifier(**best_params_wsam, random_state=0, n_jobs=-1)
best_rf.fit(X_train_wsam, y_train_wsam)

print(f"Best parameters found (wsam): {best_params_wsam}")

##################################### K-nearest neighbour ##############################################################
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

sum_sam = df_cleaned.value_counts().sum()
k_max = int(np.sqrt(sum_sam))

pca = PCA(n_components=0.95) 
X_train_pca = pca.fit_transform(X_train_sc)
X_test_pca = pca.transform(X_test_sc)

para_grid = np.arange(1,k_max+1)
best_k = 0
score = 0 
for i in para_grid:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train_pca, y_train)
    y_pred = knn.predict(X_test_pca)
    x = scoring(y_test, y_pred)
    if x > score:
        score = x
        best_k = i 
print("best k =", best_k, "best score: ",score)

best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train_pca, y_train)
################################### Analysis Performance Metrics #######################################################
#Random Forest
df_performance.loc['RF (test), with bal',:] = eval_Performance(y_test, X_test, best_rf, clf_name='Random Forest, with bal')
df_performance.loc['RF (train), with bal',:] = eval_Performance(y_train_wsam, X_train_wsam, best_rf, clf_name='Random Forest (train), with bal')

#Logisitc Regression
df_performance.loc['LR (test)',:] = eval_Performance(y_test_org, X_test_org, clf1_LR, clf_name = 'LR')
df_performance.loc['LR (train)',:] = eval_Performance(y_train_org, X_train_org, clf1_LR, clf_name = 'LR (train)')

df_performance.loc['LR (test, FS, OUS)',:] = eval_Performance(y_test, X_test_sc, clf_LR_FS_OUS, clf_name = 'LR_FS_OUS')
df_performance.loc['LR (train, FS, OUS)',:] = eval_Performance(y_train, X_train_sc, clf_LR_FS_OUS, clf_name = 'LR_FS_OUS (train)')

df_performance.loc['LR (test, FS)',:] = eval_Performance(y_test, X_test_sc, clf_LR_FS, clf_name = 'LR_FS')
df_performance.loc['LR (train, FS)',:] = eval_Performance(y_train_ws, X_train_sc_ws, clf_LR_FS, clf_name = 'LR_FS (train)')

df_performance.loc['LR (test, HP, multi)',:] = eval_Performance(y_test_org, X_test_org, clf_LR_Multi, clf_name = 'LR_Multi')
df_performance.loc['LR (train, HP, multi)',:] = eval_Performance(y_train_org, X_train_org, clf_LR_Multi, clf_name = 'LR_Multi (train)')

df_performance.loc['LR (test, HP, OVR)',:] = eval_Performance(y_test_org, X_test_org, clf_LR_OVR, clf_name = 'LR_OVR')
df_performance.loc['LR (train, HP, OVR)',:] = eval_Performance(y_train_org, X_train_org, clf_LR_OVR, clf_name = 'LR_OVR (train)')

#Knearest Neighbors
df_performance.loc['KNN (test)',:]= eval_Performance(y_test, X_test_pca,best_knn,clf_name="K-nearest neighbor")
df_performance.loc['KNN (train)',:] = eval_Performance(y_train, X_train_pca,best_knn,clf_name="K-nearest neighbor (train)")

print(df_performance)

print('Fertig')
