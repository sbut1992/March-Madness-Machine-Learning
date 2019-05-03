# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

from google.colab import drive
drive.mount('/content/drive',force_remount=True)

"""I need to generate new (and better) probabilities for the sample submission"""

data = {}
seeds = pd.read_csv('./drive/My Drive/mens-machine-learning-competition-2019/DataFiles/NCAATourneySeeds.csv')
data['samplesubmission'] = pd.read_csv('./drive/My Drive/mens-machine-learning-competition-2019/SampleSubmissionStage1.csv')
data['regseasoncompact'] = pd.read_csv('./drive/My Drive/mens-machine-learning-competition-2019/DataFiles/RegularSeasonCompactResults.csv')
data['regseasondetail'] = pd.read_csv('./drive/My Drive/mens-machine-learning-competition-2019/DataFiles/RegularSeasonDetailedResults.csv')
data['tourneycompact'] = pd.read_csv('./drive/My Drive/mens-machine-learning-competition-2019/DataFiles/NCAATourneyCompactResults.csv')
data['tourneydetail'] = pd.read_csv('./drive/My Drive/mens-machine-learning-competition-2019/DataFiles/NCAATourneyDetailedResults.csv')

"""Need to transform the target"""

data['regseasondetail']['ID'] = data['regseasondetail'][['Season','WTeamID','LTeamID']].apply(lambda x : str(x[0])+'_'+str(x[1])+'_'+str(x[2]),axis=1)
data['tourneydetail']['ID'] = data['tourneydetail'][['Season','WTeamID','LTeamID']].apply(lambda x : str(x[0])+'_'+str(x[1])+'_'+str(x[2]),axis=1)

X = data['regseasondetail'].copy()

t1 = X[X.WTeamID > X.LTeamID].loc[:,'LTeamID']
t2 = X[X.WTeamID < X.LTeamID].loc[:,'WTeamID']
team1 = pd.concat([t1,t2],axis=1).fillna(0).apply(sum,axis=1).astype(int)
team1.name = 'Team1'

t1 = X[X.WTeamID > X.LTeamID].loc[:,'WTeamID']
t2 = X[X.WTeamID < X.LTeamID].loc[:,'LTeamID']
team2 = pd.concat([t1,t2],axis=1).fillna(0).apply(sum,axis=1).astype(int)
team2.name = 'Team2'

tmp = pd.concat([team1,team2],axis=1)

tmp['Season'] = X.Season
tmp['Win'] = (tmp.Team1==X.WTeamID).astype(int)

X_new = tmp.copy()

X = data['tourneydetail'].copy()

t1 = X[X.WTeamID > X.LTeamID].loc[:,'LTeamID']
t2 = X[X.WTeamID < X.LTeamID].loc[:,'WTeamID']
team1 = pd.concat([t1,t2],axis=1).fillna(0).apply(sum,axis=1).astype(int)
team1.name = 'Team1'

t1 = X[X.WTeamID > X.LTeamID].loc[:,'WTeamID']
t2 = X[X.WTeamID < X.LTeamID].loc[:,'LTeamID']
team2 = pd.concat([t1,t2],axis=1).fillna(0).apply(sum,axis=1).astype(int)
team2.name = 'Team2'

tmp = pd.concat([team1,team2],axis=1)

tmp['Win'] = (tmp.Team1==X.WTeamID).astype(int)
tmp['Season'] = X.Season

Y_new = tmp.copy()

"""Now I'm just setting the X and Y data"""

X = data['regseasondetail'].copy()

X['WFGM2'] = X.WFGM - X.WFGM3
X['WFGA2'] = X.WFGA - X.WFGA3
X['LFGM2'] = X.LFGM - X.LFGM3
X['LFGA2'] = X.LFGA - X.LFGA3

Wie = X.apply(lambda row: row.WScore + row.WFGM + row.WFTM - row.WFGA - row.WFTA + row.WDR + (0.5 * row.WOR) + row.WAst + row.WStl + (0.5 * row.WBlk) - row.WPF - row.WTO, axis=1)
Lie = X.apply(lambda row: row.LScore + row.LFGM + row.LFTM - row.LFGA - row.LFTA + row.LDR + (0.5 * row.LOR) + row.LAst + row.LStl + (0.5 * row.LBlk) - row.LPF - row.LTO, axis=1)

# Then divide by the total game statistics (the denominator):
X['Wie'] = Wie / (Wie + Lie) * 100
X['Lie'] = Lie / (Lie + Wie) * 100


# Winner stats related to offensive efficiency:
X['Wposs'] = X.apply(lambda row: row.WFGA + 0.475 * row.WFTA + row.WTO - row.WOR, axis=1)
X['Wshoot_eff'] = X.apply(lambda row: row.WScore / (row.WFGA + 0.475 * row.WFTA), axis=1)
X['Wscore_op'] = X.apply(lambda row: (row.WFGA + 0.475 * row.WFTA) / row.Wposs, axis=1)
X['Woff_rtg'] = X.apply(lambda row: row.WScore/row.Wposs*100, axis=1)
X['Wts_pct'] = X.apply(lambda row: row.WScore / (2  * (row.WFGA + 0.475  * row.WFTA)) * 100, axis=1)
X['Wefg_pct'] = X.apply(lambda row: (row.WFGM2 + 1.5 * row.WFGM3) / row.WFGA, axis=1)
X['Worb_pct'] = X.apply(lambda row: row.WOR / (row.WOR + row.LDR), axis=1)
X['Wdrb_pct'] = X.apply(lambda row: row.WDR / (row.WDR + row.LOR), axis=1)
X['Wreb_pct'] = X.apply(lambda row: (row.Worb_pct + row.Wdrb_pct) / 2, axis=1)
X['Wto_poss'] = X.apply(lambda row: row.WTO / row.Wposs, axis=1)
X['Wft_rate'] = X.apply(lambda row: row.WFTM / row.WFGA, axis=1)
X['Wast_rtio'] = X.apply(lambda row: row.WAst / (row.WFGA + 0.475*row.WFTA + row.WTO + row.WAst) * 100, axis=1)

# Loser stats related to offensive efficiency:
X['Lposs'] = X.apply(lambda row: row.LFGA + 0.475 * row.LFTA + row.LTO - row.LOR, axis=1)
X['Lshoot_eff'] = X.apply(lambda row: row.LScore / (row.LFGA + 0.475 * row.LFTA), axis=1)
X['Lscore_op'] = X.apply(lambda row: (row.LFGA + 0.475 * row.LFTA) / row.Lposs, axis=1)
X['Loff_rtg'] = X.apply(lambda row: row.LScore/row.Lposs*100, axis=1)
X['Lefg_pct'] = X.apply(lambda row: (row.LFGM2 + 1.5 * row.LFGM3) / row.LFGA, axis=1)
X['Lorb_pct'] = X.apply(lambda row: row.LOR / (row.LOR + row.WDR), axis=1)
X['Ldrb_pct'] = X.apply(lambda row: row.LDR / (row.LDR + row.WOR), axis=1)
X['Lreb_pct'] = X.apply(lambda row: (row.Lorb_pct + row.Ldrb_pct) / 2, axis=1)
X['Lto_poss'] = X.apply(lambda row: row.LTO / row.Lposs, axis=1)
X['Lft_rate'] = X.apply(lambda row: row.LFTM / row.LFGA, axis=1)
X['Last_rtio'] = X.apply(lambda row: row.LAst / (row.LFGA + 0.475*row.LFTA + row.LTO + row.LAst) * 100, axis=1)
X['Lblk_pct'] = X.apply(lambda row: row.LBlk / row.WFGA2 * 100, axis=1)
X['Lstl_pct'] = X.apply(lambda row: row.LStl / row.Wposs * 100, axis=1)
X['Lts_pct'] = X.apply(lambda row: row.LScore / (2 * (row.LFGA + 0.475 * row.LFTA)) * 100, axis=1)
X['Lefg_pct'] = X.apply(lambda row: (row.LFGM2 + 1.5 * row.LFGM3) / row.LFGA, axis=1)
X['Lreb_pct'] = X.apply(lambda row: (row.Lorb_pct + row.Ldrb_pct) / 2, axis=1)

X['Wblk_pct'] = X.apply(lambda row: row.WBlk / row.LFGA2 * 100, axis=1)
X['Wstl_pct'] = X.apply(lambda row: row.WStl / row.Lposs * 100, axis=1)
X['Lorb_pct'] = X.apply(lambda row: row.LOR / (row.LOR + row.WDR), axis=1)
X['Ldrb_pct'] = X.apply(lambda row: row.LDR / (row.LDR + row.WOR), axis=1)
X['Ldef_rtg'] = X.apply(lambda row: row.Woff_rtg, axis=1)
X['Lsos'] = X.apply(lambda row: row.Loff_rtg - row.Woff_rtg, axis=1)
X['Wdef_rtg'] = X.apply(lambda row: row.Loff_rtg, axis=1)
X['Wsos'] = X.apply(lambda row: row.Woff_rtg - row.Loff_rtg, axis=1)

X.head()

data = X_new.set_index(['Season','Team2'])
display(data.head())

agg_cols = ['LTeamID', 'LScore','LFGM', 'LFGA', 'LFGM3', 'LFGA3',
       'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF',
       'LFGM2', 'LFGA2','Lposs', 'Lshoot_eff', 'Lscore_op', 'Loff_rtg','Ldef_rtg', 'Lsos', 'Lts_pct', 'Lefg_pct', 'Lorb_pct',
       'Ldrb_pct', 'Lreb_pct', 'Lto_poss', 'Lft_rate', 'Last_rtio', 'Lblk_pct',
       'Lstl_pct']
t2 = X.groupby(['Season','LTeamID'])[agg_cols].agg('mean').reindex(data.index)
display(t2.head())

rename_cols = {'LTeamID':'Team2_TeamID',
'LScore':'Team2_Score',
'LFGM':'Team2_FGM',
'LFGA':'Team2_FGA',
'LFGM3':'Team2_FGM3',
'LFGA3':'Team2_FGA3',
'LFTM':'Team2_FTM',
'LFTA':'Team2_FTA',
'LOR':'Team2_OR',
'LDR':'Team2_DR',
'LAst':'Team2_Ast',
'LTO':'Team2_TO',
'LStl':'Team2_Stl',
'LBlk':'Team2_Blk',
'LPF':'Team2_PF',
'LFGM2':'Team2_FGM2',
'LFGA2':'Team2_FGA2',
'Lposs':'Team2_poss',
'Lshoot_eff':'Team2_shoot_eff',
'Lscore_op':'Team2_score_op',
'Loff_rtg':'Team2_off_rtg',
'Ldef_rtg':'Team2_def_rtg',
'Lsos':'Team2_sos',
'Lts_pct':'Team2_ts_pct',
'Lefg_pct':'Team2_efg_pct',
'Lorb_pct':'Team2_orb_pct',
'Ldrb_pct':'Team2_drb_pct',
'Lreb_pct':'Team2_reb_pct',
'Lto_poss':'Team2_to_poss',
'Lft_rate':'Team2_ft_rate',
'Last_rtio':'Team2_ast_rtio',
'Lblk_pct':'Team2_blk_pct',
'Lstl_pct':'Team2_stl_pct'}
tmp = data.join(t2).rename(columns=rename_cols).reset_index()
display(tmp.head())

data = X_new.set_index(['Season','Team1'])
display(data.head())

agg_cols = ['WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR',
       'WAst', 'WTO', 'WStl', 'WBlk', 'WPF','WFGM2', 'WFGA2', 'Wposs', 'Wshoot_eff', 'Wscore_op',
       'Woff_rtg','Wdef_rtg',
       'Wsos','Wie', 'Wts_pct', 'Wefg_pct', 'Worb_pct', 'Wdrb_pct',
       'Wreb_pct', 'Wto_poss', 'Wft_rate', 'Wast_rtio', 'Wblk_pct',
       'Wstl_pct']
t1 = X.groupby(['Season','WTeamID'])[agg_cols].agg('mean').reindex(data.index)
display(t1.head())

rename_cols = {'WFGM':'Team1_FGM',
'WFGA':'Team1_FGA',
'WFGM3':'Team1_FGM3',
'WFGA3':'Team1_FGA3',
'WFTM':'Team1_FTM',
'WFTA':'Team1_FTA',
'WOR':'Team1_OR',
'WDR':'Team1_DR',
'WAst':'Team1_Ast',
'WTO':'Team1_TO',
'WStl':'Team1_Stl',
'WBlk':'Team1_Blk',
'WPF':'Team1_PF',
'WFGM2':'Team1_FGM2',
'WFGA2':'Team1_FGA2',
'Wposs':'Team1_poss',
'Wshoot_eff':'Team1_shoot_eff',
'Wscore_op':'Team1_score_op',
'Woff_rtg':'Team1_off_rtg',
'Wdef_rtg':'Team1_def_rtg',
'Wsos':'Team1_sos',
'Wie':'Team1_ie',
'Wts_pct':'Team1_ts_pct',
'Wefg_pct':'Team1_efg_pct',
'Worb_pct':'Team1_orb_pct',
'Wdrb_pct':'Team1_drb_pct',
'Wreb_pct':'Team1_reb_pct',
'Wto_poss':'Team1_to_poss',
'Wft_rate':'Team1_ft_rate',
'Wast_rtio':'Team1_ast_rtio',
'Wblk_pct':'Team1_blk_pct',
'Wstl_pct':'Team1_stl_pct'}
tmp2 = data.join(t1).rename(columns=rename_cols).reset_index()
display(tmp2.head())

data = tmp2.dropna().drop_duplicates().reset_index(drop=True)
display(data.head())

def seed_to_int(seed):
    s_int = int(seed[1:3])
    return s_int 

seeds['seed_int'] =  seeds.Seed.apply(seed_to_int)
seeds.drop(labels=['Seed'], inplace=True, axis=1)

seeds.rename(columns={'TeamID':'Team1','seed_int':'Team1_Seed'},inplace=True)

seeds.rename(columns={'TeamID':'Team1','seed_int':'Team1_Seed'},inplace=True)

len(data)

data =pd.merge(data,seeds,how='left',on=['Season','Team1'])

seeds.rename(columns={'Team1':'Team2','Team1_Seed':'Team2_Seed'},inplace=True)

data =pd.merge(data,seeds,how='left',on=['Season','Team2'])

data['Team1_Seed'] = data['Team1_Seed'].fillna(20)
data['Team2_Seed'] = data['Team2_Seed'].fillna(20)

data[['Team1_FGM', 'Team1_FGA', 'Team1_FGM3', 'Team1_FGA3', 'Team1_FTM',
       'Team1_FTA', 'Team1_OR', 'Team1_DR', 'Team1_Ast', 'Team1_TO',
       'Team1_Stl', 'Team1_Blk', 'Team1_PF', 'Team1_FGM2', 'Team1_FGA2',
       'Team1_poss', 'Team1_shoot_eff', 'Team1_score_op', 'Team1_off_rtg',
       'Team1_def_rtg', 'Team1_sos', 'Team1_ie', 'Team1_ts_pct',
       'Team1_efg_pct', 'Team1_orb_pct', 'Team1_drb_pct', 'Team1_reb_pct',
       'Team1_to_poss', 'Team1_ft_rate', 'Team1_ast_rtio', 'Team1_blk_pct',
       'Team1_stl_pct', 'Team1_Seed', 'Team2_Seed']] = data[['Team1_FGM', 'Team1_FGA', 'Team1_FGM3', 'Team1_FGA3', 'Team1_FTM',
       'Team1_FTA', 'Team1_OR', 'Team1_DR', 'Team1_Ast', 'Team1_TO',
       'Team1_Stl', 'Team1_Blk', 'Team1_PF', 'Team1_FGM2', 'Team1_FGA2',
       'Team1_poss', 'Team1_shoot_eff', 'Team1_score_op', 'Team1_off_rtg',
       'Team1_def_rtg', 'Team1_sos', 'Team1_ie', 'Team1_ts_pct',
       'Team1_efg_pct', 'Team1_orb_pct', 'Team1_drb_pct', 'Team1_reb_pct',
       'Team1_to_poss', 'Team1_ft_rate', 'Team1_ast_rtio', 'Team1_blk_pct',
       'Team1_stl_pct', 'Team1_Seed', 'Team2_Seed']].apply(pd.to_numeric)

data['ID'] = data.apply(lambda x: str(x[0].astype(int))+'_'+str(x[1].astype(int))+'_'+str(x[2].astype(int)),axis=1)
data = data.set_index('ID')
display(data.head())

features = ['Team1_Seed','Team2_Seed','Team2_TeamID',
'Team2_Score',
'Team2_FGM',
'Team2_FGA',
'Team2_FGM3',
'Team2_FGA3',
'Team2_FTM',
'Team2_FTA',
'Team2_OR',
'Team2_DR',
'Team2_Ast',
'Team2_TO',
'Team2_Stl',
'Team2_Blk',
'Team2_PF',
'Team2_FGM2',
'Team2_FGA2',
'Team2_poss',
'Team2_shoot_eff',
'Team2_score_op',
'Team2_off_rtg',
'Team2_def_rtg',
'Team2_sos',
'Team2_ts_pct',
'Team2_efg_pct',
'Team2_orb_pct',
'Team2_drb_pct',
'Team2_reb_pct',
'Team2_to_poss',
'Team2_ft_rate',
'Team2_ast_rtio',
'Team2_blk_pct',
'Team2_stl_pct',
'Team1_FGM',
'Team1_FGA',
'Team1_FGM3',
'Team1_FGA3',
'Team1_FTM',
'Team1_FTA',
'Team1_OR',
'Team1_DR',
'Team1_Ast',
'Team1_TO',
'Team1_Stl',
'Team1_Blk',
'Team1_PF',
'Team1_FGM2',
'Team1_FGA2',
'Team1_poss',
'Team1_shoot_eff',
'Team1_score_op',
'Team1_off_rtg',
'Team1_def_rtg',
'Team1_sos',
'Team1_ie',
'Team1_ts_pct',
'Team1_efg_pct',
'Team1_orb_pct',
'Team1_drb_pct',
'Team1_reb_pct',
'Team1_to_poss',
'Team1_ft_rate',
'Team1_ast_rtio',
'Team1_blk_pct',
'Team1_stl_pct']

display(data.head())
X_features = data.loc[:,data.columns.isin(features)]
y_target = data.loc[:,'Win']
display(X_features.head())
display(y_target.head())
display(np.sum(y_target==1))
np.sum(y_target==0)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import log_loss

cv_split = 10
mod = GradientBoostingClassifier()

data = X_features.join(y_target).fillna(0)

pipe = make_pipeline(StandardScaler(),mod)

X_train, X_test, y_train, y_test = train_test_split(data.drop('Win',axis=1),data['Win'], 
                                        test_size=0.1,
                                        random_state=0,
                                        shuffle=True)

fit = cross_validate(pipe,X_train,y_train,cv=cv_split,return_estimator=True)

probas = fit['estimator'][0].fit(X_train,y_train).predict_proba(X_test)
display(log_loss(y_test,probas[:,1]))

probas = fit['estimator'][0].fit(X_train,y_train).predict_proba(data.drop('Win',axis=1))
data['WProb'] = ['{:f}'.format(x) for x in probas[:,1]]

data.sort_values('WProb',ascending=False).head()

def predict(data,features,cv_split=10,mod = GradientBoostingClassifier(),target_name='Win',test_size=.1,seed=42):
    
    X_features = data.loc[:,data.columns.isin(features)]
    y_target = data.loc[:,target_name]
    
    joined = X_features.join(y_target).fillna(0)

    pipe = make_pipeline(StandardScaler(),mod)

    X_train, X_test, y_train, y_test = train_test_split(joined.drop(target_name,axis=1),joined[target_name], 
                                            test_size=test_size,
                                            random_state=seed,
                                            shuffle=True)

    fit = cross_validate(pipe,X_train,y_train,cv=cv_split,return_estimator=True)

    best = np.where(fit['test_score']==fit['test_score'].max())[0][0]
    
    probas = fit['estimator'][best].fit(X_train,y_train).predict_proba(X_test)
    
    lloss = log_loss(y_test,probas[:,1])

    probas = fit['estimator'][best].fit(X_train,y_train).predict_proba(joined.drop(target_name,axis=1))
    joined['WProb'] = ['{:f}'.format(x) for x in probas[:,1]]
    
    return joined.sort_values('WProb',ascending=False), lloss

result, lloss = predict(data,features,cv_split=2)

display(result.head())
lloss

