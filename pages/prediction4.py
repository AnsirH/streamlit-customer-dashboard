import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys

# 경로 설정
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.churn_model import load_xgboost_model2, ChurnPredictor2

st.set_page_config(page_title="고객 이탈 예측", layout="wide")
st.title("\ud83d\udcca \uace0\uac1d \uc774\ud0c8 \uc608\ucc28 \uc2dc\uc2a4\ud15c")

# --------------------------
# 1\ufe0f\ufe0f UI \uc785\ub825 \uc11c\ud551 (3\uc5f4 \xd7 6\uc904 = 18\uac1c)
# --------------------------

st.subheader("1\ufe0f\ufe0f \uace0\uac1d \ub370\uc774\ud130 \uc785\ub825")

row1 = st.columns(3)
row2 = st.columns(3)
row3 = st.columns(3)
row4 = st.columns(3)
row5 = st.columns(3)
row6 = st.columns(3)

# 1~3
tenure         = row1[0].number_input("\uc774\uc6a9 \uae30\uac04 (\uac1c\uc6d4)", min_value=0, value=12)
city_tier      = row1[1].selectbox("\uac70\uc8fc \ub3c4\uc2dc \ub4f1\uae09 (1~3)", [1, 2, 3], index=1)
warehouse_dist = row1[2].number_input("\ucc3d\uace0-\uc9d1 \uac70\ub9ac (km)", min_value=0.0, value=20.0)

# 4~6
app_hour    = row2[0].number_input("\uc571 \uc0ac\uc6a9 \uc2dc\uac04 (\uc2dc\uac04)", min_value=0.0, value=2.5)
num_devices = row2[1].number_input("\ub4f1\ub85d\ub41c \uae30\uae30 \uc218", min_value=0, value=2)
satisfaction= row2[2].slider("\ub9cc\uc871\ub3c4 \uc810\uc218 (1~5)", 1, 5, 3)

# 7~9
num_address = row3[0].number_input("\ubc30\uc1a1\uc9c0 \ub4f1\ub85d \uc218", min_value=0, value=1)
complain    = row3[1].selectbox("\ubd88\ub9cc \uc81c\uae30 \uc720\ubb34", ["\uc608", "\uc544\ub2c8\uc624"])
order_hike  = row3[2].number_input("\uc8fc\ubb38\uae08\uc561 \uc0c1\uc2b9\ub960 (%)", value=10.0)

# 10~12
coupon_used = row4[0].number_input("\ucfe0\ud3f0 \uc0ac\uc6a9 \ud69f\uc218", value=2)
orders      = row4[1].number_input("\uc8fc\ubb38 \ud69f\uc218", value=8)
last_order_days = row4[2].number_input("\ub9c8\uc9c0\ub9c9 \uc8fc\ubb38 \ud6c4 \uac74\uc640\uc77c", value=10)

# 13~15
cashback     = row5[0].number_input("\uce90\uc2dc\ubca1 \uae08\uc561", value=150)
login_device = row5[1].selectbox("\uc120\ud638 \ub85c\uadf8\uc778 \uae30\uae00", ["Mobile Phone", "Phone"])
payment_mode = row5[2].selectbox("\uc120\ud638 \uacb0\uc81c \ubc29\uc2dd", [
    "Credit Card", "Debit Card", "Cash on Delivery", "COD", "E wallet", "UPI"])

# 16~18
gender      = row6[0].selectbox("\uc131\ubcc4", ["Male", "Female"])
order_cat   = row6[1].selectbox("\uc120\ud638 \uc8fc\ubb38 \uce74\ud14c\uace0\ub9ac", [
    "Mobile", "Mobile Phone", "Laptop & Accessory", "Grocery"])
marital     = row6[2].selectbox("\uacb0\ud63c \uc720\ubb34", ["Single", "Married"])

# --------------------------
# 2\ufe0f\ufe0f \uc608\ucc28 \ubc84\ud2bc
# --------------------------

if st.button("\ud83e\uddec \uc774\ud0c8 \uc608\ucc28\ud558\uae30"):

    raw_input = {
        "Tenure": tenure,
        "CityTier": city_tier,
        "WarehouseToHome": warehouse_dist,
        "HourSpendOnApp": app_hour,
        "NumberOfDeviceRegistered": num_devices,
        "SatisfactionScore": satisfaction,
        "NumberOfAddress": num_address,
        "Complain": 1 if complain == "\uc608" else 0,
        "OrderAmountHikeFromlastYear": order_hike,
        "CouponUsed": coupon_used,
        "OrderCount": orders,
        "DaySinceLastOrder": last_order_days,
        "CashbackAmount": cashback,
        "PreferredLoginDevice": login_device,
        "PreferredPaymentMode": payment_mode,
        "Gender": gender,
        "PreferedOrderCat": order_cat,
        "MaritalStatus": marital
    }

    df_input = pd.DataFrame([raw_input])
    one_hot_cols = [
        "PreferredLoginDevice", "PreferredPaymentMode", "Gender",
        "PreferedOrderCat", "MaritalStatus"
    ]
    df_encoded = pd.get_dummies(df_input, columns=one_hot_cols)

    required_features = [
        'Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp',
        'NumberOfDeviceRegistered', 'SatisfactionScore', 'NumberOfAddress',
        'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount',
        'DaySinceLastOrder', 'CashbackAmount',
        'PreferredLoginDevice_Mobile Phone', 'PreferredLoginDevice_Phone',
        'PreferredPaymentMode_COD', 'PreferredPaymentMode_Cash on Delivery',
        'PreferredPaymentMode_Credit Card', 'PreferredPaymentMode_Debit Card',
        'PreferredPaymentMode_E wallet', 'PreferredPaymentMode_UPI',
        'Gender_Male',
        'PreferedOrderCat_Grocery', 'PreferedOrderCat_Laptop & Accessory',
        'PreferedOrderCat_Mobile', 'PreferedOrderCat_Mobile Phone',
        'MaritalStatus_Married', 'MaritalStatus_Single']

    for col in required_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    df_encoded = df_encoded[required_features]

    try:
        model = load_xgboost_model2()
        predictor = ChurnPredictor2(external_model=model)
        y_pred, y_proba = predictor.predict(df_encoded)
        prob_pct = float(y_proba[0]) * 100

        st.header("2\ufe0f\ufe0f \uc774\ud0c8 \ud655\ub960 \uc608\ucc28 \uacb0\uacfc")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_pct,
            number={'suffix': '%'},
            title={"text": "\uc774\ud0c8 \uac00\ub2a5\uc131 (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': 'darkblue'},
                'steps': [
                    {'range': [0, 30], 'color': 'green'},
                    {'range': [30, 70], 'color': 'yellow'},
                    {'range': [70, 100], 'color': 'red'}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        processed = predictor._preprocess_data(df_encoded)
        _ = predictor._compute_feature_importance(processed)
        fi = predictor.get_feature_importance()

        st.header("3\ufe0f\ufe0f \uc608\ucc28\uc5d0 \uc601\ud5a5\uc744 \uc900 \uc8fc\uc694 \uc694\uc778")
        fi_df = pd.DataFrame(fi.items(), columns=["Feature", "Importance"]) \
                 .sort_values("Importance", ascending=False)

        fig_bar = go.Figure(go.Bar(
            x=fi_df["Feature"],
            y=fi_df["Importance"]
        ))
        fig_bar.update_layout(xaxis_title="\uc785\ub825 \ubcc0\uc218", yaxis_title="\uc911\uc694\ub3c4")
        st.plotly_chart(fig_bar, use_container_width=True)

    except Exception as e:
        st.error(f"\u274c \uc608\ucc28 \uc2e4\ud328: {str(e)}")
