
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn  as sns
import sklearn
import streamlit.components.v1 as stc 
import pickle
import datetime



st.title("Advanced House Price Prediction App")
st.sidebar.header("Made By Subhadip")

cleaned_df=pd.read_csv(r'cleaned_train_data(house_price_prediction).csv')

html_temp = """
		<div style="background-color:#387b;padding:10px;border-radius:10px;width:400px">
		<h1 style="color:white;text-align:center;">Advanced House Price Prediction App </h1>
		</div>
		"""

        
def main():
    stc.html(html_temp)
    model = pickle.load(open('Advance_house_price_prediction.pkl', 'rb'))
    
    
    col1,col2,col3,col4=st.columns(4)
    
    
    with col1:
    # ------------------MSsubclass----------------1
        mssubclass=st.number_input('Enter MSSubclass value',step=1)
    # ----------------lot area--------------------3
        lotarea=st.number_input('Enter lot area in square foot')
        
        # -------------house style------------6
        house_stype_opt=['SFoyer', '1.5Fin', 'other', '1Story', 'SLvl', '2Story']
        house_style_d={'SFoyer': 0, '1.5Fin': 1, 'other': 2, '1Story': 3, 'SLvl': 4, '2Story': 5}
        house_style_=st.selectbox('Select house style', house_stype_opt)
        house_style=house_style_d[house_style_]
        
        
        
    # ----------------overall quality--------7
        overall_quality=st.number_input('Enter over quality out of 10',step=1,min_value=0,max_value=10)
        
    # ------------------------yearbuilt---------------9
        thisyear=datetime.datetime.today().year
        yearbuilt_=st.number_input('Enter year of house built',min_value=1800,step=1)
        yearbuilt=thisyear-yearbuilt_
        
        # ----------------------year of renovation-------------10
        thisyear=datetime.datetime.today().year
        yearrenew_=st.number_input('Enter year of renovation',min_value=1800,step=1)
        yearrenew=thisyear-yearrenew_
        
    # --------------------BsmtExposure--------------------13
        BsmtExposure_opt=['None', 'No', 'Mn', 'Av', 'Gd']
        BsmtExposure_d={'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
        BsmtExposure_=st.selectbox('select BsmtExposure',BsmtExposure_opt)
        BsmtExposure=BsmtExposure_d[BsmtExposure_]
   # ---------------BsmtFinSF1-------------------14

        BsmtFinSF1=st.number_input('Enter BsmtFinSF1',step=1)
    
       # --------------BsmtFinSF2-----------------15
        BsmtFinSF2=st.number_input('Enter BsmtFinSF2',step=1)
       
   
       # --------------------LowQualFinSF------------21
       
        LowQualFinSF=st.number_input('Low quality finished square feet (all floors)')
       
    with col2:
       # ------------------BsmtFullBath-----------------22
        BsmtFullBath=st.number_input('Enter number of bathrooms in basement',step=1)
       
       # -------------KitchenQual---------------23
        KitchenQual_opt=['Fa', 'TA', 'Gd', 'Ex']
        KitchenQual_d={'Fa': 0, 'TA': 1, 'Gd': 2, 'Ex': 3}
        KitchenQual_=st.selectbox('Select kitchen quality',KitchenQual_opt)
        KitchenQual=KitchenQual_d[KitchenQual_]
       
       
       # -------------------GarageCars---------------30
        GarageCars=st.number_input('Enter size of garage in terms of number of cars')
       
       
       # ---------------WoodDeckSF--------------32
        WoodDeckSF=st.number_input('Enter Wood deck area in square feet')
       
       # ----OpenPorchSF-----------33
        OpenPorchSF=st.number_input('Enter Open porch area in square feet')
       
       
       # ------EnclosedPorch------34
        EnclosedPorch=st.number_input('Enter Enclosed porch area in square feet')
       
       # ---------------3SsnPorch----------35
        SsnPorch=st.number_input('Enter Three season porch area in square feet')
      #-------------------ScreenPorch-----------------36
        ScreenPorch=st.number_input('Enter Screen porch area in square feet')
   
       
      # -------------------PoolArea-----------------37
        PoolArea=st.number_input(' Enter PoolArea')
      
    with col3:

   
        # ------------------mszoning----------2
        mszone_d={'other': 0, 'RM': 1, 'RH': 2, 'RL': 3, 'FV': 4}
        mszoning=st.selectbox('Select MSZoning',['other', 'RM', 'RH', 'RL', 'FV'])
        mszone=mszone_d[mszoning]
        
        # -------------lot configuration----------4
        lot_conf_d={'Inside': 0, 'Corner': 1, 'FR2': 2, 'other': 3, 'CulDSac': 4}
        lotconfig_=st.selectbox('select lot configuration',['Inside','Corner','FR2','CulDSac','other'])
        lotconfig=lot_conf_d[lotconfig_]
        
        # ----------neighborhoods------------------.5
        opt=['IDOTRR', 'MeadowV', 'BrDale', 'BrkSide', 'OldTown', 'Edwards',
       'Sawyer', 'SWISU', 'NAmes', 'Mitchel', 'SawyerW', 'other', 'NWAmes',
       'Gilbert', 'CollgCr', 'Blmngtn', 'Crawfor', 'ClearCr', 'Somerst',
       'Timber', 'StoneBr', 'NridgHt', 'NoRidge']
        neigh_d={'IDOTRR': 0, 'MeadowV': 1, 'BrDale': 2, 'BrkSide': 3, 'OldTown': 4, 'Edwards': 5, 'Sawyer': 6, 'SWISU': 7, 'NAmes': 8, 'Mitchel': 9, 'SawyerW': 10, 'other': 11, 'NWAmes': 12, 'Gilbert': 13, 'CollgCr': 14, 'Blmngtn': 15, 'Crawfor': 16, 'ClearCr': 17, 'Somerst': 18, 'Timber': 19, 'StoneBr': 20, 'NridgHt': 21, 'NoRidge': 22}
        neighbors_=st.selectbox('Select neighbors', opt)
        neighbors=neigh_d[neighbors_]
        
        # ---------------overall condition--------------8
        overall_cond=st.number_input('Enter rank of overall condition of house  out of 10',step=1,min_value=0,max_value=10)
        
        
        # ------------exterior1st--------------------11
        exterior1st_opt=['AsbShng', 'other', 'Wd Sdng', 'WdShing', 'MetalSd', 'Stucco',
       'HdBoard', 'Plywood', 'BrkFace', 'CemntBd', 'VinylSd']
        exterior1st_d={'AsbShng': 0, 'other': 1, 'Wd Sdng': 2, 'WdShing': 3, 'MetalSd': 4, 'Stucco': 5, 'HdBoard': 6, 'Plywood': 7, 'BrkFace': 8, 'CemntBd': 9, 'VinylSd': 10}
        exterior1st_=st.selectbox('select exterior1st',exterior1st_opt)
        exterior1st=exterior1st_d[exterior1st_]
        
        
        # ---------------Foundation-----------------------12
        foundation_opt=['Slab', 'BrkTil', 'CBlock', 'other', 'PConc']
        foundation_d={'Slab': 0, 'BrkTil': 1, 'CBlock': 2, 'other': 3, 'PConc': 4}
        foundation_=st.selectbox('select foundation type', foundation_opt)
        foundation=foundation_d[foundation_]
        
        
        # -------------BsmtUnfSF-------------16
        BsmtUnfSF=st.number_input('Enter BsmtUnfSF',step=1)
    

  
        # ------------------TotalBsmtSF------------17
        TotalBsmtSF=st.number_input('Enter TotalBsmtSF',step=1)
        
        # ----------------HeatingQC----------------18
        HeatingQC_opt=['other', 'Fa', 'TA', 'Gd', 'Ex']
        HeatingQC_d={'other': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
        HeatingQC_=st.selectbox('Select HeatingQC',HeatingQC_opt)
        HeatingQC=HeatingQC_d[HeatingQC_]
    
        # ---------------1stFlrSF---------------19
        first_stFlrSF=st.number_input('Enter area of first floor in square foot')
        
    with col4:
        # -----------------2ndFlrSF-----------------20
        second_ndFlrSF=st.number_input('Enter area of second floor in square foot')
        
        
        # -------------TotRmsAbvGrd------------------24
        TotRmsAbvGrd=st.number_input('Enter Total rooms above grade (does not include bathrooms)',step=1)
        
        # -------------------Functional---------------25
        Functional_opt=['other', 'Min2', 'Mod', 'Min1', 'Typ']
        Functional_d={'other': 0, 'Min2': 1, 'Mod': 2, 'Min1': 3, 'Typ': 4}
        Functional_=st.selectbox('Enter Home functionality rating',Functional_opt)
        Functional=Functional_d[Functional_]
        
        
        # -----------------------FireplaceQu-----------------26
        FireplaceQu_opt=['Po', 'None', 'Fa', 'TA', 'Gd', 'Ex']
        FireplaceQu_d={'Po': 0, 'None': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
        FireplaceQu_=st.selectbox('Select fireplace quality',FireplaceQu_opt)
        FireplaceQu=FireplaceQu_d[FireplaceQu_]
        
        
        # ----------------------------GarageType---------------27
        GarageType_opt=['None', 'Rare_var', 'Detchd', 'Basment', 'Attchd', 'BuiltIn']
        GarageType_d={'None': 0, 'Rare_var': 1, 'Detchd': 2, 'Basment': 3, 'Attchd': 4, 'BuiltIn': 5}
        GarageType_=st.selectbox('Select GarageType',GarageType_opt)
        GarageType=GarageType_d[GarageType_]
        
        # ---------------GarageYrBlt---------28
        thisyear=datetime.datetime.today().year
        GarageYrBlt_=st.number_input('Enter GarageYrBlt',step=1)
        GarageYrBlt=thisyear-GarageYrBlt_
        
        
        
        # ----------------------GarageFinish-------------------29
        GarageFinish_opt=['None', 'Unf', 'RFn', 'Fin']
        GarageFinish_d={'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
        GarageFinish_=st.selectbox('select Interior finish of the garage',GarageFinish_opt)
        GarageFinish=GarageFinish_d[GarageFinish_]
        
        # --------------------GarageArea-----------31-
        GarageArea=st.number_input('Enter GarageArea in square foot')
        
        
        
        # ----------------SaleCondition-------------38
        SaleCondition_opt=['Abnorml', 'Rare_var', 'Family', 'Normal', 'Partial']
        SaleCondition_d={'Abnorml': 0, 'Rare_var': 1, 'Family': 2, 'Normal': 3, 'Partial': 4}
        SaleCondition_=st.selectbox('',SaleCondition_opt)
        SaleCondition=SaleCondition_d[SaleCondition_]
        
        
        
        
    inputs=[mssubclass,mszone,lotarea,lotconfig,neighbors,
                house_style,overall_quality,overall_cond,yearbuilt,yearrenew,
                exterior1st,foundation,BsmtExposure,BsmtFinSF1,BsmtFinSF2,
                BsmtUnfSF,TotalBsmtSF,HeatingQC,first_stFlrSF,second_ndFlrSF,
                LowQualFinSF,BsmtFullBath,KitchenQual,TotRmsAbvGrd,Functional,
                FireplaceQu,GarageType,GarageYrBlt,GarageFinish,GarageCars,
                GarageArea,WoodDeckSF,OpenPorchSF,EnclosedPorch,SsnPorch,
                ScreenPorch,PoolArea,SaleCondition]
        
        
      # scaling of inputs1
      
    if st.button('Predict House price'):
        result=model.predict([inputs])
        st.success('The house price with given feature is {} $'.format(round(result[0][0],2)))
            
    
if __name__=='__main__':
    main()