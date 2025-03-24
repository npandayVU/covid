# %% [markdown]
# # Imports

# %%
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import sqlite3
from statsmodels.tsa.seasonal import STL
import plotly.express as px
import streamlit as st

# %% [markdown]
# # Classes

# %%
class SIRD:
    def __init__(self, s0, i0, r0, d0):
        self.ts = {
            'Susceptible': [s0],
            'Infected': [i0],
            'Recovered': [r0],
            'Deceased': [d0]
        }
        self.N = s0 + i0 + r0 + d0
        self.len = 1
    
    def step(self, alpha, beta, gamma, mu):
        s = self.ts['Susceptible'][-1]
        i = self.ts['Infected'][-1]
        r = self.ts['Recovered'][-1]
        d = self.ts['Deceased'][-1]

        a = alpha * r
        b = beta * s * i / self.N
        g = gamma * i
        m = mu * i

        self.ts['Susceptible'].append(s + a - b)
        self.ts['Infected'].append(i + b - m - g)
        self.ts['Recovered'].append(r + g - a)
        self.ts['Deceased'].append(d + m)
        self.len += 1

    def simulate(self, alpha, beta, gamma, mu):
        for a, b, g, m in zip(alpha, beta, gamma, mu):
            self.step(a, b, g, m)

# %% [markdown]
# # Functions

# %%
def lineplot(df, title='Title', xlabel=None, ylabel=None, errorbar='ci'):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=df, ax=ax, errorbar=errorbar)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    st.pyplot(fig)

def plot_dates(df, dates, title='Title', ylabel='y'):
    lineplot(df.loc[dates], title=title, xlabel='Date', ylabel=ylabel)

def stl(ts):
    dec = STL(ts)
    res = dec.fit()
    return res

def compare(observed, modeled):
    for obs, mod in zip(observed, modeled):
        df = pd.DataFrame({'Observed': observed[obs], 'Modeled': modeled[mod]})
        lineplot(df, title=mod, xlabel='Date', ylabel='Count')
# %% [markdown]
# # Part 3

# %%
covid_database = sqlite3.connect('covid_database.db')
cursor = covid_database.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
# %% [markdown]
# # Part 4

# %% [markdown]
# ## Data Wrangling

# %%
complete = pd.read_csv('complete.csv')
# %% [markdown]
# Fill missing data

# %%
complete.fillna({
    'Province.State': 'Unknown',
    'Confirmed': 0,
    'Deaths': 0,
    'Recovered': 0,
    'Active': 0
}, inplace=True)
# %% [markdown]
# Drop duplicates

# %%
duplicates = complete.duplicated()
dupe_count = duplicates.sum()
complete.drop_duplicates(inplace=True)

# %% [markdown]
# Fix 'Date' datatype

# %%
complete['Date'] = pd.to_datetime(complete['Date'])
# %%
complete.info()

# %% [markdown]
# ## Observations

# %%
total_population = pd.read_sql_query(f"""
                                SELECT Population FROM worldometer_data;
                                """, covid_database).sum().iloc[0]
# %%
GAMMA = 1/4.5
COUNTRY = 'Italy'

# %%
def plot_cases(df, country):
    daywise = df[df['Country.Region'] == country][['Date','Confirmed','Active','Recovered','Deaths']]\
                .set_index('Date')
    population = pd.read_sql_query(f"""
                                    SELECT Population FROM worldometer_data
                                    WHERE "Country.Region" = '{country}';
                                    """, covid_database).iloc[0, 0]
    lineplot(daywise / population, title=f'COVID-19 Cases - {country}', xlabel='Date', ylabel='Fraction of Population')

# %% [markdown]
# ## Parameter Estimation

# %%
def estimate_params(df, gamma, country):
    # Extract and index
    daywise = df[df['Country.Region'] == country][['Date','Active','Recovered','Deaths']]\
                .set_index('Date').copy()
    daywise = daywise.sort_index()
    
    # Population from worldometer
    population = pd.read_sql_query(f"""
        SELECT Population FROM worldometer_data
        WHERE "Country.Region" = '{country}';
    """, covid_database).iloc[0, 0]

    # Differentials
    delta_I = daywise['Active'].diff()[1:]
    delta_D = daywise['Deaths'].diff()[1:]
    delta_R = daywise['Recovered'].diff()[1:]
    
    # Time-aligned data (drop first row to match diffed values)
    infected = daywise['Active'][:-1].values
    recovered = daywise['Recovered'][:-1].values
    deaths = daywise['Deaths'][:-1].values
    susceptible = population - infected - recovered - deaths
    
    # Estimate parameters
    mu = (delta_D / infected).replace([np.inf, -np.inf], 0).fillna(0)
    alpha = ((gamma * infected - delta_R) / recovered).replace([np.inf, -np.inf], 0).fillna(0)
    beta_num = delta_I + mu * infected + gamma * infected
    beta_den = (susceptible * infected) / population
    beta = (beta_num / beta_den).replace([np.inf, -np.inf], 0).fillna(0)

    params = pd.DataFrame({
        'alpha': alpha,
        'beta': beta,
        'mu': mu
    })
    params['gamma'] = gamma
    params = params.rolling(7, min_periods=1).mean().clip(lower=0)\
                .dropna().astype(float)
    return params

# %% [markdown]
# ## SIRD-Model

# %%
def sird_model(df, gamma, country):
    daywise = df[df['Country.Region'] == country][['Date','Active','Recovered','Deaths']]\
                .set_index('Date').copy()
    daywise = daywise.sort_index()
    population = pd.read_sql_query(f"""
                                    SELECT Population FROM worldometer_data
                                    WHERE "Country.Region" = '{country}';
                                    """, covid_database).iloc[0, 0]
    i0, r0, d0 = daywise[['Active','Recovered','Deaths']].iloc[0]
    s0 = population - i0 - r0 - d0
    sird = SIRD(s0, i0, r0, d0)
    params = estimate_params(df, gamma, country)
    sird.simulate(alpha=params['alpha'], beta=params['beta'], gamma=params['gamma'], mu=params['mu'])
    sird_df = pd.DataFrame(sird.ts, index=daywise.index)
    return sird_df / population

# %% [markdown]
# ## R0

# %%
def calculate_r0(df, gamma, country):
    params = estimate_params(df, gamma, country)
    r0 = params['beta'] / params['gamma']
    return r0
# %% [markdown]
# # Other

# %% [markdown]
# ## Map of Active Cases in Europe

# %%
def plot_chloropleth():
    data = pd.read_sql_query("""
        SELECT "Country.Region" AS Country, ActiveCases, Population
        FROM worldometer_data
        WHERE Continent = 'Europe';
    """, covid_database)

    data['CaseRate'] = data['ActiveCases'] / data['Population']

    fig = px.choropleth(
        data_frame=data,
        locations='Country',
        locationmode='country names',
        color='CaseRate',
        scope='europe',
        title='COVID-19 Active Case Rates in Europe',
        color_continuous_scale='Oranges',
    )

    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)
# %% [markdown]
# ## Estimated average death rate per continent

# %%
def plot_death_rates(selected_continent):
    continents = pd.read_sql_query("""
        SELECT "Country.Region", Continent
        FROM worldometer_data
        WHERE Continent IS NOT NULL;
    """, covid_database)

    mu_data = complete[['Date', 'Country.Region', 'Active', 'Deaths']].copy()
    mu_data = mu_data[mu_data['Active'] > 0]
    mu_data.sort_values(['Country.Region', 'Date'], inplace=True)
    mu_data['Delta_D'] = mu_data.groupby('Country.Region')['Deaths'].diff().fillna(0)
    mu_data = mu_data[mu_data['Delta_D'] >= 0]
    mu_data['mu'] = (mu_data['Delta_D'] / mu_data['Active']).replace([np.inf, -np.inf], 0).fillna(0)
    mean_mu = mu_data.groupby('Country.Region')['mu'].mean().reset_index()
    mu_with_continent = pd.merge(mean_mu, continents, on='Country.Region')
    continent_mu = mu_with_continent.groupby('Continent')['mu'].mean().sort_values(ascending=False)

    st.subheader("Estimated Death Rates by Continent")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=continent_mu.index, y=continent_mu.values, ax=ax)
    ax.set_ylabel("μ (Mortality Rate Estimate)")
    st.pyplot(fig)
    top_mu = mu_with_continent[mu_with_continent['Continent'] == selected_continent]\
        .sort_values('mu', ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=top_mu, x='Country.Region', y='mu', ax=ax)
    ax.set_title(f"Top 10 Countries by Death Rate Estimate (μ) - {selected_continent}")
    st.pyplot(fig)
# %%
# # Dashboard
st.set_page_config(page_title='COVID-19 Dashboard', layout='wide')
tab1, tab2, tab3 = st.tabs(["Global", "Continental", "Countrywise"])

countries = complete['Country.Region'].unique()
continents = complete['WHO.Region'].unique()
dates = complete['Date'].unique()

st.sidebar.title("Dashboard Controls")
selected_continent = st.sidebar.selectbox("Select continent", continents)
selected_country = st.sidebar.selectbox("Select country", countries)
selected_start_date = st.sidebar.selectbox("Select start date", dates)
selected_end_date = st.sidebar.selectbox("Select end date", dates[dates >= selected_start_date])

complete = complete[(complete['Date'] >= selected_start_date) &
                    (complete['Date'] <= selected_end_date)]
with tab1:
    st.subheader("General Statistics")
    stats = {
        'Datapoints': complete.size,
        'Continents': continents.size,
        'Countries': countries.size,
        'Dates': dates[-1] - dates[0],
    }
    st.dataframe(stats)

    st.subheader("Global COVID-19 Progression (as Population Fractions)")
    for col in ['Confirmed', 'Active', 'Recovered', 'Deaths']:
        df = complete[['Date', col]].copy()
        df['Fraction'] = df[col] / total_population

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.lineplot(data=df, x='Date', y='Fraction', ax=ax)
        ax.set_title(f'{col}')
        ax.set_ylabel('Fraction of Population')
        st.pyplot(fig)
with tab2:
    plot_death_rates(selected_continent)
    st.subheader(f"European countries with Active Cases")
    plot_chloropleth()
with tab3:
    st.subheader(f'COVID-19 in {selected_country}')
    plot_cases(complete, selected_country)
    with st.spinner("Estimating parameters..."):
        params = estimate_params(complete, GAMMA, selected_country)
    # params = estimate_params(complete, GAMMA, selected_country)
    lineplot(params, title=f'Estimated SIRD-Model Parameters - {selected_country}', xlabel='Date')
    sird = sird_model(complete, GAMMA, selected_country)
    lineplot(sird, title=f'SIRD-Model - {selected_country}', xlabel='date', ylabel='Fraction of Population')
