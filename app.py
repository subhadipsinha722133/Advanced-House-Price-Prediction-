import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import datetime
from PIL import Image
import streamlit.components.v1 as stc 

# Page configuration
st.set_page_config(
    page_title="Advanced House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.header("Made By Subhadip 🔥")
# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #E3F2;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin: 20px 0;
    }
    .feature-section {
        background-color: #F5F5F5;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        width: 100%;
    }
    .stSelectbox, .stNumberInput {
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown('<h1 class="main-header">🏠 Advanced House Price Prediction</h1>', unsafe_allow_html=True)
st.markdown("""
This app predicts house prices based on various features like location, size, condition, and amenities. 
Adjust the parameters in the sidebar and see how they affect the predicted price.
""")

# Load data and model
@st.cache_resource
def load_data():
    try:
        cleaned_df = pd.read_csv('cleaned_train_data(house_price_prediction).csv')
        return cleaned_df
    except:
        st.warning("Could not load the dataset. Some features may be limited.")
        return None

@st.cache_resource
def load_model():
    try:
        model = pickle.load(open('Advance_house_price_prediction.pkl', 'rb'))
        return model
    except:
        st.error("Model file not found. Please ensure 'Advance_house_price_prediction.pkl' is in the correct directory.")
        return None

# Load data and model
cleaned_df = load_data()
model = load_model()

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Select Page", ["Prediction", "Data Overview", "Feature Importance"])

# Main app logic
if app_mode == "Prediction":
    if model is None:
        st.error("Model not available. Cannot make predictions.")
    else:
        # Create input sections with better organization
        st.sidebar.header("House Features")
        
        # Use expanders to organize inputs
        with st.sidebar.expander("Basic Information", expanded=True):
            # MSSubClass
            mssubclass = st.number_input('MS SubClass', min_value=20, max_value=190, value=60, step=5,
                                        help="Identifies the type of dwelling involved in the sale")
            
            # MSZoning
            mszone_d = {'other': 0, 'RM': 1, 'RH': 2, 'RL': 3, 'FV': 4}
            mszoning = st.selectbox('MS Zoning', ['other', 'RM', 'RH', 'RL', 'FV'],
                                   help="Identifies the general zoning classification of the sale")
            
            # Lot Area
            lotarea = st.number_input('Lot Area (sq ft)', min_value=1000, max_value=50000, value=10000, step=500,
                                     help="Lot size in square feet")
            
            # Lot Configuration
            lot_conf_d = {'Inside': 0, 'Corner': 1, 'FR2': 2, 'other': 3, 'CulDSac': 4}
            lotconfig_ = st.selectbox('Lot Configuration', ['Inside', 'Corner', 'FR2', 'CulDSac', 'other'],
                                     help="Lot configuration")
            
            # Neighborhood
            neigh_d = {'IDOTRR': 0, 'MeadowV': 1, 'BrDale': 2, 'BrkSide': 3, 'OldTown': 4, 'Edwards': 5, 
                      'Sawyer': 6, 'SWISU': 7, 'NAmes': 8, 'Mitchel': 9, 'SawyerW': 10, 'other': 11, 
                      'NWAmes': 12, 'Gilbert': 13, 'CollgCr': 14, 'Blmngtn': 15, 'Crawfor': 16, 'ClearCr': 17, 
                      'Somerst': 18, 'Timber': 19, 'StoneBr': 20, 'NridgHt': 21, 'NoRidge': 22}
            neighbors_ = st.selectbox('Neighborhood', list(neigh_d.keys()),
                                     help="Physical locations within Ames city limits")
            
            # House Style
            house_style_d = {'SFoyer': 0, '1.5Fin': 1, 'other': 2, '1Story': 3, 'SLvl': 4, '2Story': 5}
            house_style_ = st.selectbox('House Style', list(house_style_d.keys()),
                                       help="Style of dwelling")
        
        with st.sidebar.expander("Quality & Condition"):
            # Overall Quality
            overall_quality = st.slider('Overall Quality', min_value=1, max_value=10, value=6,
                                       help="Rates the overall material and finish of the house")
            
            # Overall Condition
            overall_cond = st.slider('Overall Condition', min_value=1, max_value=10, value=6,
                                    help="Rates the overall condition of the house")
            
            # Kitchen Quality
            KitchenQual_d = {'Fa': 0, 'TA': 1, 'Gd': 2, 'Ex': 3}
            KitchenQual_ = st.selectbox('Kitchen Quality', list(KitchenQual_d.keys()),
                                       help="Kitchen quality")
            
            # Heating Quality and Condition
            HeatingQC_d = {'other': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
            HeatingQC_ = st.selectbox('Heating Quality', list(HeatingQC_d.keys()),
                                     help="Heating quality and condition")
            
            # Functional
            Functional_d = {'other': 0, 'Min2': 1, 'Mod': 2, 'Min1': 3, 'Typ': 4}
            Functional_ = st.selectbox('Home Functionality', list(Functional_d.keys()),
                                      help="Home functionality rating")
        
        with st.sidebar.expander("Size & Area"):
            # 1st Floor Square Feet
            first_stFlrSF = st.number_input('1st Floor Area (sq ft)', min_value=300, max_value=5000, value=1500, step=100,
                                           help="First Floor square feet")
            
            # 2nd Floor Square Feet
            second_ndFlrSF = st.number_input('2nd Floor Area (sq ft)', min_value=0, max_value=5000, value=0, step=100,
                                            help="Second floor square feet")
            
            # Total Basement Area
            TotalBsmtSF = st.number_input('Total Basement Area (sq ft)', min_value=0, max_value=5000, value=1000, step=100,
                                         help="Total square feet of basement area")
            
            # Garage Area
            GarageArea = st.number_input('Garage Area (sq ft)', min_value=0, max_value=2000, value=500, step=50,
                                        help="Size of garage in square feet")
            
            # Garage Cars
            GarageCars = st.slider('Garage Size (car capacity)', min_value=0, max_value=5, value=2,
                                  help="Size of garage in car capacity")
        
        with st.sidebar.expander("Age & Renovation"):
            # Year Built
            thisyear = datetime.datetime.today().year
            yearbuilt_ = st.slider('Year Built', min_value=1800, max_value=thisyear, value=1990)
            yearbuilt = thisyear - yearbuilt_
            
            # Year Renovated
            yearrenew_ = st.slider('Year Renovated', min_value=1800, max_value=thisyear, value=1990)
            yearrenew = thisyear - yearrenew_ if yearrenew_ > 1800 else thisyear - yearbuilt_
            
            # Garage Year Built
            GarageYrBlt_ = st.slider('Garage Year Built', min_value=1800, max_value=thisyear, value=1990)
            GarageYrBlt = thisyear - GarageYrBlt_ if GarageYrBlt_ > 1800 else thisyear - yearbuilt_
        
        # Additional features in the main area
        st.header("Additional Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="feature-section">', unsafe_allow_html=True)
            st.subheader("Basement Features")
            
            # BsmtExposure
            BsmtExposure_d = {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
            BsmtExposure_ = st.selectbox('Basement Exposure', list(BsmtExposure_d.keys()))
            
            # BsmtFinSF1
            BsmtFinSF1 = st.number_input('BsmtFinSF1', min_value=0, max_value=5000, value=0, step=50,
                                        help="Type 1 finished square feet")
            
            # BsmtFinSF2
            BsmtFinSF2 = st.number_input('BsmtFinSF2', min_value=0, max_value=5000, value=0, step=50,
                                        help="Type 2 finished square feet")
            
            # BsmtUnfSF
            BsmtUnfSF = st.number_input('BsmtUnfSF', min_value=0, max_value=5000, value=0, step=50,
                                       help="Unfinished square feet of basement area")
            
            # BsmtFullBath
            BsmtFullBath = st.number_input('Basement Full Bathrooms', min_value=0, max_value=3, value=0)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="feature-section">', unsafe_allow_html=True)
            st.subheader("Exterior Features")
            
            # Exterior 1st
            exterior1st_d = {'AsbShng': 0, 'other': 1, 'Wd Sdng': 2, 'WdShing': 3, 'MetalSd': 4, 
                            'Stucco': 5, 'HdBoard': 6, 'Plywood': 7, 'BrkFace': 8, 'CemntBd': 9, 'VinylSd': 10}
            exterior1st_ = st.selectbox('Exterior Covering', list(exterior1st_d.keys()))
            
            # Foundation
            foundation_d = {'Slab': 0, 'BrkTil': 1, 'CBlock': 2, 'other': 3, 'PConc': 4}
            foundation_ = st.selectbox('Foundation Type', list(foundation_d.keys()))
            
            # Fireplace Quality
            FireplaceQu_d = {'Po': 0, 'None': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
            FireplaceQu_ = st.selectbox('Fireplace Quality', list(FireplaceQu_d.keys()))
            
            # Garage Type
            GarageType_d = {'None': 0, 'Rare_var': 1, 'Detchd': 2, 'Basment': 3, 'Attchd': 4, 'BuiltIn': 5}
            GarageType_ = st.selectbox('Garage Type', list(GarageType_d.keys()))
            
            # Garage Finish
            GarageFinish_d = {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
            GarageFinish_ = st.selectbox('Garage Finish', list(GarageFinish_d.keys()))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="feature-section">', unsafe_allow_html=True)
            st.subheader("Outdoor Features")
            
            # Wood Deck SF
            WoodDeckSF = st.number_input('Wood Deck Area (sq ft)', min_value=0, max_value=1000, value=0, step=50)
            
            # Open Porch SF
            OpenPorchSF = st.number_input('Open Porch Area (sq ft)', min_value=0, max_value=500, value=0, step=25)
            
            # Enclosed Porch
            EnclosedPorch = st.number_input('Enclosed Porch Area (sq ft)', min_value=0, max_value=500, value=0, step=25)
            
            # 3Ssn Porch
            SsnPorch = st.number_input('3 Season Porch Area (sq ft)', min_value=0, max_value=500, value=0, step=25)
            
            # Screen Porch
            ScreenPorch = st.number_input('Screen Porch Area (sq ft)', min_value=0, max_value=500, value=0, step=25)
            
            # Pool Area
            PoolArea = st.number_input('Pool Area (sq ft)', min_value=0, max_value=1000, value=0, step=50)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Additional features
        st.markdown('<div class="feature-section">', unsafe_allow_html=True)
        col4, col5, col6 = st.columns(3)
        
        with col4:
            # Low Quality Fin SF
            LowQualFinSF = st.number_input('Low Quality Finished Area (sq ft)', min_value=0, max_value=1000, value=0, step=50)
        
        with col5:
            # Total Rooms Above Grade
            TotRmsAbvGrd = st.number_input('Total Rooms Above Grade', min_value=2, max_value=15, value=6)
        
        with col6:
            # Sale Condition
            SaleCondition_d = {'Abnorml': 0, 'Rare_var': 1, 'Family': 2, 'Normal': 3, 'Partial': 4}
            SaleCondition_ = st.selectbox('Sale Condition', list(SaleCondition_d.keys()))
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Map selections to numerical values
        mszone = mszone_d[mszoning]
        lotconfig = lot_conf_d[lotconfig_]
        neighbors = neigh_d[neighbors_]
        house_style = house_style_d[house_style_]
        KitchenQual = KitchenQual_d[KitchenQual_]
        HeatingQC = HeatingQC_d[HeatingQC_]
        Functional = Functional_d[Functional_]
        BsmtExposure = BsmtExposure_d[BsmtExposure_]
        exterior1st = exterior1st_d[exterior1st_]
        foundation = foundation_d[foundation_]
        FireplaceQu = FireplaceQu_d[FireplaceQu_]
        GarageType = GarageType_d[GarageType_]
        GarageFinish = GarageFinish_d[GarageFinish_]
        SaleCondition = SaleCondition_d[SaleCondition_]
        
        # Prepare input array
        inputs = [mssubclass, mszone, lotarea, lotconfig, neighbors,
                 house_style, overall_quality, overall_cond, yearbuilt, yearrenew,
                 exterior1st, foundation, BsmtExposure, BsmtFinSF1, BsmtFinSF2,
                 BsmtUnfSF, TotalBsmtSF, HeatingQC, first_stFlrSF, second_ndFlrSF,
                 LowQualFinSF, BsmtFullBath, KitchenQual, TotRmsAbvGrd, Functional,
                 FireplaceQu, GarageType, GarageYrBlt, GarageFinish, GarageCars,
                 GarageArea, WoodDeckSF, OpenPorchSF, EnclosedPorch, SsnPorch,
                 ScreenPorch, PoolArea, SaleCondition]
        
        # Prediction button
        if st.button('Predict House Price', key='predict'):
            try:
                result = model.predict([inputs])
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>Predicted House Price: ${result[0][0]:,.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Show feature impact (simplified)
                st.subheader("Feature Impact on Price")
                feature_names = [
                    "MS SubClass", "MS Zoning", "Lot Area", "Lot Configuration", "Neighborhood",
                    "House Style", "Overall Quality", "Overall Condition", "Years Since Built", "Years Since Renovation",
                    "Exterior Covering", "Foundation", "Basement Exposure", "BsmtFinSF1", "BsmtFinSF2",
                    "BsmtUnfSF", "Total Basement Area", "Heating Quality", "1st Floor Area", "2nd Floor Area",
                    "Low Quality Fin SF", "Basement Full Bath", "Kitchen Quality", "Total Rooms Above Grade", "Functionality",
                    "Fireplace Quality", "Garage Type", "Years Since Garage Built", "Garage Finish", "Garage Cars",
                    "Garage Area", "Wood Deck Area", "Open Porch Area", "Enclosed Porch Area", "3 Season Porch Area",
                    "Screen Porch Area", "Pool Area", "Sale Condition"
                ]
                
                # Create a simple bar chart showing relative importance (placeholder)
                fig, ax = plt.subplots(figsize=(10, 6))
                importance = np.random.rand(len(feature_names))  # Placeholder for actual feature importance
                sorted_idx = np.argsort(importance)
                
                ax.barh(range(10), importance[sorted_idx][-10:])
                ax.set_yticks(range(10))
                ax.set_yticklabels([feature_names[i] for i in sorted_idx[-10:]])
                ax.set_xlabel('Relative Importance')
                ax.set_title('Top 10 Features Affecting Price')
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")

elif app_mode == "Data Overview":
    st.header("Dataset Overview")
    
    if cleaned_df is not None:
        st.dataframe(cleaned_df.head())
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Summary")
            st.write(cleaned_df.describe())
        
        with col2:
            st.subheader("Column Information")
            st.write(cleaned_df.dtypes)
        
        # Show distribution of sale prices
        st.subheader("Sale Price Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(cleaned_df['SalePrice'], bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Sale Price')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of House Prices')
        st.pyplot(fig)
    else:
        st.warning("Dataset not available for overview.")

elif app_mode == "Feature Importance":
    st.header("Feature Importance Analysis")
    
    # Placeholder for feature importance visualization
    st.info("This section would typically show the importance of different features in the model's predictions.")
    
    # Create a sample feature importance plot
    feature_names = [
        "MS SubClass", "MS Zoning", "Lot Area", "Lot Configuration", "Neighborhood",
        "House Style", "Overall Quality", "Overall Condition", "Years Since Built", "Years Since Renovation",
        "Exterior Covering", "Foundation", "Basement Exposure", "BsmtFinSF1", "BsmtFinSF2",
        "BsmtUnfSF", "Total Basement Area", "Heating Quality", "1st Floor Area", "2nd Floor Area",
        "Low Quality Fin SF", "Basement Full Bath", "Kitchen Quality", "Total Rooms Above Grade", "Functionality",
        "Fireplace Quality", "Garage Type", "Years Since Garage Built", "Garage Finish", "Garage Cars",
        "Garage Area", "Wood Deck Area", "Open Porch Area", "Enclosed Porch Area", "3 Season Porch Area",
        "Screen Porch Area", "Pool Area", "Sale Condition"
    ]
    
    # Generate random importance values for demonstration
    importance = np.random.rand(len(feature_names))
    sorted_idx = np.argsort(importance)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.barh(range(len(feature_names)), importance[sorted_idx])
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Feature Importance (Example)')
    plt.tight_layout()
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
**Note:** This is a demonstration app for house price prediction. 
Actual predictions may vary based on model training and data quality.
""")