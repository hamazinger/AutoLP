import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import streamlit as st

def init_bigquery_client():
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    return bigquery.Client(credentials=credentials)

def load_seminar_data() -> pd.DataFrame:
    client = init_bigquery_client()

    query = """
    SELECT
        Seminar_Title,
        Acquisition_Speed,
        Major_Category,
        Category,
        Total_Participants,
        Action_Response_Count,
        Action_Response_Rate,
        User_Company_Percentage,
        Non_User_Company_Percentage
    FROM `mythical-envoy-386309.majisemi.majisemi_seminar_usukiapi`
    WHERE Seminar_Title IS NOT NULL
    AND Acquisition_Speed IS NOT NULL
    """

    try:
        df = client.query(query).to_dataframe()
        if 'Major_Category' not in df.columns:
            raise ValueError("Major_Categoryカラムが見つかりません")
        return df
    except Exception as e:
        raise Exception(f"データの読み込みでエラーが発生しました: {str(e)}")