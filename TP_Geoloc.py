
# coding: utf-8

# # <center> TP Geoloc </center>

# Groupe :
# - Randy Reinette
# - David Tang
# - Valentin Phetchanpheng
# - Pascal Lim

# <strong> Objectif : Prédire la position du message (latitude/longitude) </strong>

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from geopy.distance import vincenty
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', 300)


# In[2]:


# On charge le train and test data
path = "C:/Users/Pascal/Desktop/Telecom ParisTech/Cours/INF730 - Internet des objets BGD/Machine learning for IOT/data/"
path_val = "/Users/valentinphetchanpheng/Documents/MS BGD Telecom ParisTech/Machine Learning IoT/TP_geoloc/"
df_mess_train = pd.read_csv(path+'mess_train_list.csv') # train set
df_mess_test = pd.read_csv(path+'mess_test_list.csv') # test set
pos_train = pd.read_csv(path+'pos_train_list.csv') # position associated to train set


# ## Data Exploration

# ### Visualisation des données

# On affiche les différentes tables:

# In[3]:


# On rassemble la table de features avec la liste de label
df_mess_train[['pos_lat', 'pos_lng']] = pos_train


# In[4]:


# Train set
df_mess_train.head()


# In[5]:


# Test set
df_mess_test.head()


# In[6]:


print("Train Number of Rows : %d, Train Number of columns : %d" %(df_mess_train.shape[0], df_mess_train.shape[1]))
print("Test Number of Rows : %d, Test Number of columns : %d" %(df_mess_test.shape[0], df_mess_test.shape[1]))
print("Data Types :" + "\n")
print(df_mess_train.dtypes)


# On cherche à rentrer plus dans le détail des données et repérant le nombre de valeurs uniques dans les données globales (train+test):

# In[7]:


print("Nombre de messages uniques : %d" %len(df_mess_train.messid.unique()))
print("Nombre de stations uniques : %d" %len(df_mess_train.bsid.unique()))
print("Nombre de devices uniques : %d" %len(df_mess_train.did.unique()))
print("Nombre de valeurs uniques de time_ux : %d" %len(df_mess_train.time_ux.unique()))
print("Nombre de valeurs uniques de rssi : %d" %len(df_mess_train.rssi.unique()))
print("Nombre de valeurs uniques de nseq: %d" %len(df_mess_train.nseq.unique()))


# On va s'intéresser maintenant à la distribution des valeurs dans certaines variables dans l'échantillon train et test:

# In[8]:


bins = np.linspace(0, 2.5, 11)
plt.figure(figsize=(14,4))
plt.subplot(121)
plt.hist(df_mess_train.nseq, bins=bins)
plt.title("Distribution des valeurs de nseq - Train")
plt.subplot(122)
plt.hist(df_mess_test.nseq, bins=bins, color="orange")
plt.title("Distribution des valeurs de nseq - Test")
plt.show()


# On s'aperçoit que les valeurs de nseq sont des valeurs catégorielles. On a une distribution à peu près équivalent pour les dataset train et le dataset test.

# In[9]:


plt.figure(figsize=(14,4))
plt.subplot(121)
sns.distplot(df_mess_train.rssi, bins=300)
plt.title("Distribution des valeurs de rssi - Train")
plt.subplot(122)
sns.distplot(df_mess_test.rssi, bins=300, color='orange')
plt.title("Distribution des valeurs de rssi - Test")
plt.show()


# On a aussi une même distribution pour les valeurs de rssi. Les valeurs sont principalement comprises entre -140 dBm et -100 dBm.

# In[10]:


plt.figure(figsize=(14,4))
plt.subplot(121)
sns.distplot(df_mess_train.time_ux, bins=100, kde=False)
plt.title("Distribution des valeurs de time_ux - Train")
plt.subplot(122)
sns.distplot(df_mess_test.time_ux, bins=100, kde=False, color='orange')
plt.title("Distribution des valeurs de time_ux - Test")
plt.show()


# On remarque que la distribution des données est différente entre les deux cas. Cependant les valeurs sont comprises à peu près entre 1,4625 ms et 1.4825 ms.

# In[11]:


plt.figure(figsize=(14,4))
plt.subplot(121)
sns.distplot(df_mess_train.bsid, bins=100, kde=False)
plt.title("Distribution du nombre de messages par bsid - Train")
plt.subplot(122)
sns.distplot(df_mess_test.bsid, bins=100, kde=False, color='orange')
plt.title("Distribution du nombre de messages par bsid - Test")
plt.show()


# On remarque que certaines base station reçoivent beaucoup plus de messages que les autres. On pourra le prendre en compte lors de  l'analyse des catégories peu représentées. La distribution, cependant, reste à peu près similaire entre les deux dataset.

# In[12]:


plt.figure(figsize=(14,4))
plt.subplot(121)
sns.distplot(df_mess_train.did, bins=100, kde=False)
plt.title("Distribution du nombre de messages par device id - Train")
plt.subplot(122)
sns.distplot(df_mess_test.did, bins=100, kde=False, color='orange')
plt.title("Distribution du nombre de messages par device id - Test")
plt.show()


# On s'aperçoit que les valeurs des device id sont très séparées, on devra jeter un coup d'oeil sur le nombre de messages par device id et analyser les device id peu représentés.

# ### Corrélation entre les variables

# In[13]:


plt.figure(figsize=(14,5), dpi= 80)
plt.subplot(121)
sns.heatmap(df_mess_train.corr(), xticklabels=df_mess_train.corr().columns, yticklabels=df_mess_train.corr().columns, cmap='PuOr', center=0, annot=True)
plt.title('Matrice de corrélation - Train', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.subplot(122)
sns.heatmap(df_mess_test.corr(), xticklabels=df_mess_test.corr().columns, yticklabels=df_mess_test.corr().columns, cmap='PuOr', center=0, annot=True)
plt.title('Matrice de corrélation - Test', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


# On n'observe pas de corrélations importantes entre les variables à part entre la latitute et la longitude des stations et du message émis.

# ### Détection des outliers

# In[14]:


# Plot bs latitude vs bs longitude
plt.figure(figsize=(6,4))
plt.plot(df_mess_train['bs_lat'], df_mess_train['bs_lng'], '.')
plt.title('Position des Base stations')
plt.xlabel('Base Station Lat')
plt.ylabel('Base Station Long')
plt.show()


# On remarque qu'il y a une partie des basestations qui sont éloignées des autres avec une longitude et latitude proche de -70 et 65. 

# In[15]:


# Plot bs latitude vs bs longitude
plt.figure(figsize=(6,4))
plt.plot(df_mess_train.pos_lat, df_mess_train.pos_lng, ".")
plt.title("Position du message")
plt.xlabel("Lattitude")
plt.ylabel("Longitude")
plt.show()


# En observant les positions réelles des messages envoyés par les devices, on remarque qu'il y a aucun message envoyé à des longitudes et latitudes proches de -70 et 65. On conclut qu'il s'agit d'une erreur dans position de ces base stations.

# On essaye de trouver des stations de base qui ont des positions extrêmes par rapport au reste et voir si il est pertinent ou pas de les garder.

# In[16]:


# Base station ayant une latitude > 60
df_mess_train[df_mess_train['bs_lat']>60].head(5)


# In[17]:


list_bs_outlier = df_mess_train[df_mess_train['bs_lat']>60].bsid.unique()
print('Nombre de message associé au bsid :\n', df_mess_train[df_mess_train['bs_lat']>60].groupby('bsid').count()['messid'].sort_values(ascending=True))
print("\nNombre de message total associé aux bsid dont bs_lat > 60 : {} ".format(df_mess_train[df_mess_train['bs_lat']>60].groupby('bsid').count().sum()[0]))


# On a donc identifié ces stations de base, puis on a regardé à combien de messages sont associés ces stations de base.
# Nous remarquons que le nombre de message total associé à ces bsid est assez significatif (environ 4000 / 39 000).
# On décide tout de même de les retirer car cela va induire en erreur notre algorithme.

# In[18]:


# On retire les données liées aux basestation avec latitude >60
index_to_remove1 = df_mess_train[df_mess_train['bs_lat']>60].index.values
df_mess_train = df_mess_train.drop(index_to_remove1)


# ### Détection des classes peu représentées

# On chosit de détecter d'abord les stations qui détectent peu de messages pour potentiellement les enlever du train set et ainsi avoir moins de catégories en entrée de l'algorithme de prédiction par la suite.

# In[19]:


# On repère les base stations qui ne reçoivent pas beaucoup de messages
count_basestation = df_mess_train.groupby('bsid').count()
count_basestation = count_basestation['messid']
mes_limit = 1000
plt.figure(figsize=(8,6))
count_basestation_cum = count_basestation.sort_values(ascending=True).cumsum()
plt.plot(count_basestation_cum.values)
plt.plot([0,230], [mes_limit,mes_limit])
plt.title("Somme cumulée des messages en fonction du nombre de Base stations", size=14)
plt.show()


# In[20]:


print("Nombre de Base stations en dessous de la droite rouge: %d" %len(count_basestation_cum[count_basestation_cum<mes_limit]))


# On observe approximativement qu'à partir de ces 111 Base stations, le nombre de messages augmente considérablement. On décide de retirer ces Base stations de notre étude.

# In[21]:


# Les bsid qui ne reçoivent pas beaucoup de messages sont retirés des données train
bsid_to_remove = count_basestation_cum[count_basestation_cum<mes_limit].index.values
index_to_remove2 = df_mess_train[df_mess_train.bsid.isin(bsid_to_remove)].index.values
df_mess_train = df_mess_train.drop(index_to_remove2)

# Remise des index par défaux pour le train dataframe pour la jointure des dataframes par la suite
n_train = df_mess_train.shape[0]
df_mess_train = df_mess_train.set_index(np.arange(n_train))


# ## Features Engineering

# In[22]:


# On concatène le train et le test set
df_concact = df_mess_train.iloc[:,:-2].append(df_mess_test, sort=False, ignore_index=True)


# ### Catégorisation Base station

# In[23]:


# On détermine tous les Base stations prises en compte dans le train set
# (après suppression de des base station peu représentées) et le test set
listOfBs = np.unique(df_concact.bsid)
listNameBs = ["bs"+str(i+1) for i in range(len(listOfBs))]


# In[24]:


print("Nombre de stations de Base stations prises en compte au final: ", len(listOfBs))


# In[25]:


# OneHotEncoder pour BSID
ohe = OneHotEncoder()
X_bsid = ohe.fit_transform(df_concact[['bsid']]).toarray()
df_bsid_train = pd.DataFrame(X_bsid[:n_train,:], columns = listNameBs)
df_bsid_test = pd.DataFrame(X_bsid[n_train:,:], columns = listNameBs)


# In[26]:


# On rajoute ces colonnes catégorielles à nos dataset Train et Test
df_mess_train[listNameBs] = df_bsid_train
df_mess_test[listNameBs] = df_bsid_test


# ### Catégorisation Device ID

# On veut également catégoriser la feature Device Id pour notre modèle.

# In[27]:


# On détermine tous les Base stations prises en compte dans le train set
# (après suppression de des base station peu représentées) et le test set
list_did = np.unique(df_concact.did)
listNamedid = ["did"+str(i+1) for i in range(len(list_did))]


# In[28]:


print("Nombre de stations de Devices uniques pris en compte au final: ", len(list_did))


# Le nombre de devices uniques est élevé ce qui va créer beaucoup de features et risque d'alourdir notre modèle. On verra lors de la sélection des features celles qui pourraient être supprimées.

# In[29]:


# OneHotEncoder pour DID
ohe = OneHotEncoder()
X_did = ohe.fit_transform(df_concact[['did']]).toarray()
df_did_train = pd.DataFrame(X_did[:n_train,:], columns = listNamedid)
df_did_test = pd.DataFrame(X_did[n_train:,:], columns = listNamedid)


# In[30]:


# On rajoute ces colonnes catégorielles à nos dataset Train et Test
df_mess_train[listNamedid] = df_did_train
df_mess_test[listNamedid] = df_did_test


# ### Groupby par MessID

# In[31]:


# On groupe par messid
df_grouped_train = df_mess_train.groupby(['messid'])
df_grouped_test = df_mess_test.groupby(['messid'])


# #### Groupement de tous les bsid par MessID

# In[32]:


# On récupère tous les basestation concernés par un messid donné
df_bsid_grouped_train = df_grouped_train.sum()[listNameBs]
df_bsid_grouped_test = df_grouped_test.sum()[listNameBs]


# #### Groupement de tous les did par MessID

# In[33]:


# On récupère tous les basestation concernés par un messid donné
df_did_grouped_train = df_grouped_train.mean()[listNamedid]
df_did_grouped_test = df_grouped_test.mean()[listNamedid]


#  #### Nombre de bsid par MessID

# In[34]:


# On crée la variable du nombre total de bsid par message
count_bsid_grouped_train = df_bsid_grouped_train.sum(axis=1).values
count_bsid_grouped_test = df_bsid_grouped_test.sum(axis=1).values


#  #### Moyenne DeviceID par MessID

# C'est une feature qui n'a pas forcément de sens mais a réussi à améliorer notre score.

# In[35]:


# Moyenne des Device ID par messid
did_grouped_train = df_grouped_train.mean()['did'].values
did_grouped_test = df_grouped_test.mean()['did'].values


#  #### Moyenne du RSSI par MessID

# In[36]:


# Moyenne du RSSI par messid
rssi_grouped_train = df_grouped_train.mean()['rssi'].values
rssi_grouped_test = df_grouped_test.mean()['rssi'].values


# #### Moyenne/Moyenne pondérée par le RSSI des lat/long

# In[37]:


# Addition de features latitude et longitude moyenne
lat_grouped_train = df_grouped_train.mean()['bs_lat'].values
lat_grouped_test = df_grouped_test.mean()['bs_lat'].values

lng_grouped_train = df_grouped_train.mean()['bs_lng'].values
lng_grouped_test = df_grouped_test.mean()['bs_lng'].values


# In[38]:


# Addition de features latitude et longitude moyenne pondérées par rssi
lat_rssi_grouped_train = df_grouped_train.apply(lambda x: pd.Series([np.average(x['bs_lat'], weights=x['rssi'])], index=['messid'])).unstack().values
lat_rssi_grouped_test = df_grouped_test.apply(lambda x: pd.Series([np.average(x['bs_lat'], weights=x['rssi'])], index=['messid'])).unstack().values

lng_rssi_grouped_train = df_grouped_train.apply(lambda x: pd.Series([np.average(x['bs_lng'], weights=x['rssi'])], index=['messid'])).unstack().values
lng_rssi_grouped_test = df_grouped_test.apply(lambda x: pd.Series([np.average(x['bs_lng'], weights=x['rssi'])], index=['messid'])).unstack().values


# #### Moyenne/Moyenne pondérée par le time_ux des lat/long

# In[39]:


# Moyenne time_ux par messid
time_ux_grouped_train = df_grouped_train.mean()['time_ux'].values
time_ux_grouped_test = df_grouped_test.mean()['time_ux'].values


# In[40]:


# lat/long pondérées par time_ux
time_ux_lat_grouped_train = df_grouped_train.apply(lambda x: pd.Series([np.average(x['bs_lat'], weights=x['time_ux'])], index=['messid'])).unstack().values
time_ux_lat_grouped_test = df_grouped_test.apply(lambda x: pd.Series([np.average(x['bs_lat'], weights=x['time_ux'])], index=['messid'])).unstack().values

time_ux_lng_grouped_train = df_grouped_train.apply(lambda x: pd.Series([np.average(x['bs_lng'], weights=x['time_ux'])], index=['messid'])).unstack().values
time_ux_lng_grouped_test = df_grouped_test.apply(lambda x: pd.Series([np.average(x['bs_lng'], weights=x['time_ux'])], index=['messid'])).unstack().values


# #### Moyenne pondérée par le nseq des lat/long

# In[41]:


# lat/long pondérées par nseq
nseq_lat_grouped_train = df_grouped_train.apply(lambda x: pd.Series([np.average(x['bs_lat'], weights=x['nseq']+1)], index=['messid'])).unstack().values
nseq_lat_grouped_test = df_grouped_test.apply(lambda x: pd.Series([np.average(x['bs_lat'], weights=x['nseq']+1)], index=['messid'])).unstack().values

nseq_lng_grouped_train = df_grouped_train.apply(lambda x: pd.Series([np.average(x['bs_lng'], weights=x['nseq']+1)], index=['messid'])).unstack().values
nseq_lng_grouped_test = df_grouped_test.apply(lambda x: pd.Series([np.average(x['bs_lng'], weights=x['nseq']+1)], index=['messid'])).unstack().values


# #### Moyenne label

# In[42]:


# Groupby label
pos_lat_grouped_train = df_grouped_train.mean()['pos_lat'].values
pos_lng_grouped_train = df_grouped_train.mean()['pos_lng'].values


# ### Features selection

# In[43]:


# On met en place la combinaison des features qui a permis d'atteindre notre meilleur score
df_train = pd.DataFrame()
df_train[listNameBs] = df_bsid_grouped_train
df_train[listNamedid] = df_did_grouped_train
df_train['mean_rssi'] = rssi_grouped_train
df_train['mean_lat'] = lat_grouped_train
df_train['mean_lng'] = lng_grouped_train
df_train['mean_lat_rssi'] = lat_rssi_grouped_train
df_train['mean_lng_rssi'] = lng_rssi_grouped_train
df_train['mean_time_ux'] = time_ux_grouped_train
df_train['pos_lat'] = pos_lat_grouped_train
df_train['pos_lng'] = pos_lng_grouped_train

df_test = pd.DataFrame()
df_test[listNameBs] = df_bsid_grouped_test
df_test[listNamedid] = df_did_grouped_test
df_test['mean_rssi'] = rssi_grouped_test
df_test['mean_lat'] = lat_grouped_test
df_test['mean_lng'] = lng_grouped_test
df_test['mean_lat_rssi'] = lat_rssi_grouped_test
df_test['mean_lng_rssi'] = lng_rssi_grouped_test
df_test['mean_time_ux'] = time_ux_grouped_test


# On veut avoir un aperçu de l'importance de nos features sur un modèle ExtraTrees Regressor afin d'affiner la sélection de nos variables.

# In[44]:


# Transformation en Numpy array
X_train = df_train.iloc[:,:-2]
y_lat_train = df_train.pos_lat
y_lng_train = df_train.pos_lng

X_test = df_test


# In[45]:


# Mise en place du modèle ExtraTree pour avoir un aper
clf_lat = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
clf_lat.fit(X_train, y_lat_train)
clf_lng = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
clf_lng.fit(X_train, y_lng_train)


# In[46]:


# Calcul des Features Importances pour la longitude
dict_feature_importance_lat = {'feature': X_train.columns.values, 'importance': clf_lat.feature_importances_}
feature_importances_lat = pd.DataFrame(data=dict_feature_importance_lat).sort_values('importance', ascending=False)
dict_feature_importance_lng = {'feature': X_train.columns.values, 'importance': clf_lng.feature_importances_}
feature_importances_lng = pd.DataFrame(data=dict_feature_importance_lng).sort_values('importance', ascending=False)


# In[47]:


# Calcul des Features Importances pour la latitude
plt.figure(figsize=(14,40))
sns.barplot(x="importance",
            y="feature",
            data=feature_importances_lat.sort_values(by="importance",
                                                     ascending=False))
plt.title('Feature Importance Latitude')
plt.tight_layout()


# In[48]:


plt.figure(figsize=(14,40))
sns.barplot(x="importance",
           y="feature",
           data=feature_importances_lng.sort_values(by="importance",
                                          ascending=False))
plt.title('Feature Importance Longitude')
plt.tight_layout()


# On remarque que la majorité de l'importance est captée entre quelques features. On va donc d'abord s'intéresser aux features dont l'importance est inférieure à 0.0001 et les retirer de nos train et test set.

# In[49]:


# On enlève les features avec importance < 0.00025
seuil_importance = 0.000025
index_to_remove3 = list(set(feature_importances_lat[feature_importances_lat.importance < seuil_importance].iloc[:,0]).intersection(set(feature_importances_lng[feature_importances_lng.importance < seuil_importance].iloc[:,0])))
print("%d features ont une importance inférieure à %.6f." %(len(index_to_remove3), seuil_importance))
X_train = X_train.drop(index_to_remove3, axis=1)
X_test = X_test.drop(index_to_remove3, axis=1)


# ## Modèles de prédiction

# In[50]:


# Fonction qui permet d'évaluer nos résultats
def vincenty_vec(vec_coord):
    vin_vec_dist = np.zeros(vec_coord.shape[0])
    if vec_coord.shape[1] !=  4:
        print('ERROR: Bad number of columns (shall be = 4)')
    else:
        vin_vec_dist = [vincenty(vec_coord[m,0:2],vec_coord[m,2:]).meters for m in range(vec_coord.shape[0])]
    return vin_vec_dist


# In[51]:


# Fonction pour évaluer l'erreur en distance
def Eval_geoloc(y_train_lat , y_train_lng, y_pred_lat, y_pred_lng):
    vec_coord = np.array([y_train_lat , y_train_lng, y_pred_lat, y_pred_lng])
    err_vec = vincenty_vec(np.transpose(vec_coord))
    return err_vec


# In[53]:


# On met en place nos arrays pour l'étape de cross-validation
Xtrain_cv, Xtest_cv, y_lat_train_cv, y_lat_test_cv, y_lng_train_cv, y_lng_test_cv = train_test_split(X_train,
                                                                                                     y_lat_train,
                                                                                                     y_lng_train,
                                                                                                     test_size=0.2,
                                                                                                     random_state=42)


# ### RandomForest Regressor

# On cherche à optimiser notre algorithme de RandomForest. Pour cette étude on décide de s'intéresser à la profondeur de l'arbre et à la proportion de features à considérer à chaque split.

# In[54]:


# On évalue à partir de quelle profondeur d'arbre on obtient de très bons résultats sans overfitter
list_max_depth = [20, 25, 30, 35, 40, 45, 50, 55, 60]
list_err80 = []
for max_depth in list_max_depth:
    clf_rf_lat = RandomForestRegressor(n_estimators = 100, max_depth=max_depth, n_jobs=-1)
    clf_rf_lat.fit(Xtrain_cv, y_lat_train_cv)
    y_pred_lat = clf_rf_lat.predict(Xtest_cv) 

    clf_rf_lng = RandomForestRegressor(n_estimators = 100, max_depth=max_depth, n_jobs=-1)
    clf_rf_lng.fit(Xtrain_cv, y_lng_train_cv)
    y_pred_lng = clf_rf_lng.predict(Xtest_cv)
    
    err_vec = Eval_geoloc(y_lat_test_cv , y_lng_test_cv, y_pred_lat, y_pred_lng)
    list_err80.append(np.percentile(err_vec, 80))


# In[55]:


# On affiche l'évolution de l'erreur en distance pour les différentes profondeurs d'arbre évaluées
plt.figure()
plt.plot(list_max_depth, list_err80)
plt.title("Evolution de l'erreur en distance par rapport à la profondeur d'arbre")
plt.show()


# Meilleur résultat pour max_depth = 40 : au-delà on augmente pas le score et on a des risques d'overfitting.

# In[69]:


# On optimise le paramètre max_feature
list_max_features = [0.5, 0.6, 0.7, 0.8, 0.9, None]
list_err_vec = []
list_err80 = []
for max_features in list_max_features:
    clf_rf_lat = RandomForestRegressor(n_estimators = 100, max_features=max_features, max_depth=40, n_jobs=-1)
    clf_rf_lat.fit(Xtrain_cv, y_lat_train_cv)
    y_pred_lat = clf_rf_lat.predict(Xtest_cv) 

    clf_rf_lng = RandomForestRegressor(n_estimators = 100, max_features=max_features, max_depth=40, n_jobs=-1)
    clf_rf_lng.fit(Xtrain_cv, y_lng_train_cv)
    y_pred_lng = clf_rf_lng.predict(Xtest_cv)
    
    err_vec = Eval_geoloc(y_lat_test_cv , y_lng_test_cv, y_pred_lat, y_pred_lng)
    list_err_vec.append(err_vec)
    list_err80.append(np.percentile(err_vec, 80))


# In[70]:


# On affiche l'évolution de l'erreur en distance pour les différentes max_features évaluées
plt.figure()
plt.plot(list_max_features, list_err80)
plt.title("Evolution de l'erreur en distance par rapport à max_features")
plt.show()


# In[71]:


# On entraîne notre modèle RandomForest sur 80% du train set et on valide sur les 20% restants 
clf_rf_lat = RandomForestRegressor(n_estimators = 200, max_features=0.7, max_depth=40, n_jobs=-1)
clf_rf_lat.fit(Xtrain_cv, y_lat_train_cv)
y_pred_lat = clf_rf_lat.predict(Xtest_cv) 

clf_rf_lng = RandomForestRegressor(n_estimators = 200, max_features=0.7, max_depth=40, n_jobs=-1)
clf_rf_lng.fit(Xtrain_cv, y_lng_train_cv)
y_pred_lng = clf_rf_lng.predict(Xtest_cv)

err_vec = Eval_geoloc(y_lat_test_cv , y_lng_test_cv, y_pred_lat, y_pred_lng)
print("Erreur de distance cumulé à 80% : {}" .format((np.percentile(err_vec, 80))))


# In[72]:


# On affiche le graphe des erreurs de distance cumulées
values, base = np.histogram(err_vec, bins=50000)
cumulative = np.cumsum(values) 
plt.figure();
plt.plot(base[:-1]/1000, cumulative / np.float(np.sum(values))  * 100.0, c='blue')
plt.grid(); plt.xlabel('Distance Error (km)'); plt.ylabel('Cum proba (%)'); plt.axis([0, 30, 0, 100]); 
plt.title('Error Cumulative Probability'); plt.legend( ["Opt LLR", "LLR 95", "LLR 99"])
plt.show()


# ### Extratrees Regressor

# On essaye de tester avec un autre algorithme d'ensemble learning très proche du RandomForest : le régresseur Extratrees.
# On décide de garder les mêmes paramètres pour cet algorithme.

# In[103]:


# On entraîne notre modèle Extratrees sur 80% du train set et on valide sur les 20% restants 
clf_lat = ExtraTreesRegressor(n_estimators=200, max_features=0.7, max_depth=40, n_jobs=-1)
clf_lat.fit(Xtrain_cv, y_lat_train_cv)
y_pred_lat = clf_lat.predict(Xtest_cv)
clf_lng = ExtraTreesRegressor(n_estimators=200, max_features=0.7, max_depth=40, n_jobs=-1)
clf_lng.fit(Xtrain_cv, y_lng_train_cv)
y_pred_lng = clf_lng.predict(Xtest_cv)

err_vec = Eval_geoloc(y_lat_test_cv , y_lng_test_cv, y_pred_lat, y_pred_lng)
print("Erreur de distance cumulé à 80% : {}" .format((np.percentile(err_vec, 80))))


# In[104]:


# On affiche le graphe des erreurs de distance cumulées
values, base = np.histogram(err_vec, bins=50000)
cumulative = np.cumsum(values) 
plt.figure();
plt.plot(base[:-1]/1000, cumulative / np.float(np.sum(values))  * 100.0, c='blue')
plt.grid(); plt.xlabel('Distance Error (km)'); plt.ylabel('Cum proba (%)'); plt.axis([0, 30, 0, 100]); 
plt.title('Error Cumulative Probability'); plt.legend( ["Opt LLR", "LLR 95", "LLR 99"])
plt.show()


# On obtient des meilleurs résultats avec l'algorithme d'Extratrees que le RandomForest.
# On décide donc de partir sur cet algorithme pour la prédiction sur le test set.

# ## Construction fichier de prédiction

# In[105]:


# On prédit sur le test set à partir de l'algorithme d'Extratrees entraîné précédemment
y_pred_lat_final = clf_lat.predict(X_test)
y_pred_lng_final = clf_lng.predict(X_test)


# In[106]:


# Construction du fichier de prédiction
test_res = pd.DataFrame(np.array([y_pred_lat_final, y_pred_lng_final]).T, columns = ['lat', 'lng'])
test_res.to_csv('pred_pos_test_list.csv', index=False)
test_res.head()


# ## Prédicteur one device_out

# Pour créer le prédicteur one device out, on repart de notre matrice de features existantes :

# In[92]:


df_train.head(5)


# On décide quand même de supprimer les features les moins importantes sans prendre en compte les variables catégorielles liées au device id.

# In[93]:


# Transformation en Numpy array
X_train2 = df_train.iloc[:,:-2]
y_lat_train2 = df_train.pos_lat
y_lng_train2 = df_train.pos_lng

X_test2 = df_test


# In[94]:


# On enlève les features avec importance < 0.00025
seuil_importance = 0.000025
index_to_remove4 = list(set(feature_importances_lat[feature_importances_lat.importance < seuil_importance].iloc[:,0]).intersection(set(feature_importances_lng[feature_importances_lng.importance < seuil_importance].iloc[:,0])))
index_to_remove4 = list(filter(lambda x : x.find('did'),index_to_remove4))

print("%d features ont une importance inférieure à %.6f." %(len(index_to_remove4), seuil_importance))
X_train2 = X_train2.drop(index_to_remove4, axis=1)
X_test2 = X_test2.drop(index_to_remove4, axis=1)


# On crée un test de validation en retirant un device.
# Pour l'exemple, on enlève le device qui correspond à l'attribut did121.

# In[99]:


device_id_out = 'did121'


# In[100]:


# On récupère les index qui concernent le device qu'on cherche à retirer du dataset
index_to_keep = X_train2[X_train2[device_id_out]==0].index
index_device_out = X_train2[X_train2[device_id_out]==1].index

Xtrain_cv2 = X_train2.loc[index_to_keep]
Xtest_cv2 = X_train2.loc[index_device_out]

y_lat_train_cv2 = y_lat_train2.loc[index_to_keep]
y_lat_test_cv2 = y_lat_train2.loc[index_device_out]

y_lng_train_cv2 = y_lng_train2.loc[index_to_keep]
y_lng_test_cv2 = y_lng_train2.loc[index_device_out]


# On choisit d'entraîner sur le modèle ExtraTrees avec les paramètres de la partie précédente.

# In[101]:


# On entraîne notre modèle Extratrees sur 80% du train set et on valide sur les 20% restants 
clf_lat = ExtraTreesRegressor(n_estimators=200, max_features=0.7, max_depth=40, n_jobs=-1)
clf_lat.fit(Xtrain_cv2, y_lat_train_cv2)
y_pred_lat = clf_lat.predict(Xtest_cv2)
clf_lng = ExtraTreesRegressor(n_estimators=200, max_features=0.7, max_depth=40, n_jobs=-1)
clf_lng.fit(Xtrain_cv2, y_lng_train_cv2)
y_pred_lng = clf_lng.predict(Xtest_cv2)

err_vec = Eval_geoloc(y_lat_test_cv2 , y_lng_test_cv2, y_pred_lat, y_pred_lng)
print("Erreur de distance cumulé à 80% : {}" .format((np.percentile(err_vec, 80))))


# In[102]:


# On affiche le graphe des erreurs de distance cumulées
values, base = np.histogram(err_vec, bins=50000)
cumulative = np.cumsum(values) 
plt.figure();
plt.plot(base[:-1]/1000, cumulative / np.float(np.sum(values))  * 100.0, c='blue')
plt.grid(); plt.xlabel('Distance Error (km)'); plt.ylabel('Cum proba (%)'); plt.axis([0, 30, 0, 100]); 
plt.title('Error Cumulative Probability'); plt.legend( ["Opt LLR", "LLR 95", "LLR 99"])
plt.show()

