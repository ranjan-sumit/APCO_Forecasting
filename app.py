import pandas as pd
from prophet import Prophet
import streamlit as st
import plotly.graph_objs as go

import pandas as pd
from prophet import Prophet
import streamlit as st
import plotly.graph_objs as go

# Set up the title of the app
st.title("APCO Forecasting App")
st.markdown("---")
st.sidebar.title("APCO Forecasting App")
st.sidebar.markdown("---")
st.sidebar.write('This application helps us in forecasting using the media monitoring software Talkwalker, and represents the volume of media mentions.')

# Filepath to your Excel file
file_path = 'Data.xlsx'

# Read all sheets from the Excel file
sheets = pd.read_excel(file_path, sheet_name=None)
sheet_names = list(sheets.keys())

# Dropdown for selecting the sheet
selected_sheet = st.selectbox("Select a sheet to forecast:", sheet_names)

# Process the selected sheet
if selected_sheet:
    data = sheets[selected_sheet]
    
    # Display the selected sheet name
    st.write(f"Processing sheet: **{selected_sheet}**")
    st.markdown("---")
    
    # Assuming the first column is 'Date' and the second column is the data to forecast
    data.columns = ['Date', 'Volume of mentions']
    
    # Prepare data for Prophet
    prophet_data = data.rename(columns={'Date': 'ds', 'Volume of mentions': 'y'})
    
    # Initialize and fit the Prophet model
    model = Prophet()
    model.fit(prophet_data)
    
    # Make future dataframe for 14 days
    future = model.make_future_dataframe(periods=14)
    
    # Forecast
    forecast = model.predict(future)
    
    # Calculate percentage change and moving average
    forecast['pct_change'] = forecast['yhat'].pct_change() * 100
    forecast['moving_avg'] = forecast['yhat'].rolling(window=7).mean()

    # Plot the forecast using Plotly
    fig = go.Figure()

    # Add historical data
    fig.add_trace(go.Scatter(
        x=prophet_data['ds'], 
        y=prophet_data['y'], 
        mode='lines+markers', 
        name='Historical Data',
        line=dict(color='blue')
    ))

    # Add forecast data
    fig.add_trace(go.Scatter(
        x=forecast['ds'], 
        y=forecast['yhat'], 
        mode='lines', 
        name='Forecast', 
        line=dict(color='orange')
    ))

    # Add lower and upper uncertainty intervals
    fig.add_trace(go.Scatter(
        x=forecast['ds'], 
        y=forecast['yhat_lower'], 
        mode='lines', 
        name='Lower Bound', 
        line=dict(color='red', width=0.5, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast['ds'], 
        y=forecast['yhat_upper'], 
        mode='lines', 
        name='Upper Bound', 
        line=dict(color='green', width=0.5, dash='dash')
    ))

    # Update layout
    fig.update_layout(
        title=f'Prophet Forecast for {selected_sheet}',
        xaxis_title='Date',
        yaxis_title='Volume of Mentions',
        legend_title='Legend',
        hovermode='x',
        height=600,
        width=1000,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    # Show the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Detailed Explanation
    st.markdown("""
    ### Forecast Details
    The forecast table below shows the predicted volume of media mentions for the next two weeks along with the lower and upper bounds indicating the uncertainty in the predictions.

    - **Predicted Value (yhat)**: The forecasted volume of mentions.
    - **Lower Bound**: The lower limit of the uncertainty interval.
    - **Upper Bound**: The upper limit of the uncertainty interval.
    - **Percentage Change**: The percentage change in volume from the previous forecast.
    - **Moving Average**: A 7-day moving average of the forecasted values.
    """)

    # Print forecast
    st.write(f"Forecast for the next two weeks in **{selected_sheet}**:")
    forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'pct_change', 'moving_avg']].tail(14)
    st.markdown("---")
    st.dataframe(forecast_table, width=1000, height=400)

# Footer
st.markdown("""
---
*Forecasting App using Prophet*
""")
