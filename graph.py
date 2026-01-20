import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==============================================================================
# INSTELLINGEN & CONFIGURATIE
# ==============================================================================
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 12

# ==============================================================================
# FASE 1: DATA INLADEN EN SCHOONMAKEN (ETL)
# ==============================================================================
def load_and_clean_data():
    """
    Laadt de CSV bestanden en voert initiële schoonmaakacties uit.
    """
    print("--- FASE 1: Data Inladen en Opschonen ---")
    try:
        # 1. Laden van Dataframes
        # ISO-8859-1 encoding is nodig voor speciale karakters in namen
        df_sales = pd.read_csv('Sales.csv')
        df_products = pd.read_csv('Products.csv')
        df_stores = pd.read_csv('Stores.csv')
        df_customers = pd.read_csv('Customers.csv', encoding='ISO-8859-1') 
        df_exchange = pd.read_csv('Exchange_Rates.csv')

        # 2. Datum Conversies (MET FIX VOOR 00/00/0000)
        print("Converteren van datums naar datetime objecten...")
        
        # errors='coerce' zorgt ervoor dat foute datums (zoals 00/00/0000) worden omgezet naar NaT (Not a Time)
        # in plaats van het script te laten crashen.
        df_sales['Order Date'] = pd.to_datetime(df_sales['Order Date'], errors='coerce')
        df_stores['Open Date'] = pd.to_datetime(df_stores['Open Date'], errors='coerce')
        df_customers['Birthday'] = pd.to_datetime(df_customers['Birthday'], errors='coerce')
        df_exchange['Date'] = pd.to_datetime(df_exchange['Date'], errors='coerce')

        # Verwijder rijen waar de datum nu ongeldig (NaT) is geworden in Sales
        # Want een verkoop zonder datum kunnen we niet analyseren.
        initial_sales_count = len(df_sales)
        df_sales = df_sales.dropna(subset=['Order Date'])
        if len(df_sales) < initial_sales_count:
            print(f" -> {initial_sales_count - len(df_sales)} rijen met ongeldige datums (bv. 00/00/0000) verwijderd.")

        # 3. Opschonen Exchange Rates
        df_exchange = df_exchange.dropna(subset=['Date']) # Verwijder ongeldige datums in wisselkoersen
        df_exchange = df_exchange.sort_values(by='Date')
        df_exchange = df_exchange.ffill()
        
        # 4. Filteren van de '1916' datums in Sales (Data Cleaning)
        initial_count = len(df_sales)
        df_sales = df_sales[df_sales['Order Date'].dt.year >= 2000]
        print(f" -> {initial_count - len(df_sales)} records met foutieve historische jaartallen (voor 2000) verwijderd.")

        return df_sales, df_products, df_stores, df_customers, df_exchange

    except FileNotFoundError as e:
        print(f"KRITIEKE FOUT: Bestand niet gevonden: {e}")
        return None, None, None, None, None

# ==============================================================================
# FASE 2: DATA INTEGRATIE EN VERRIJKING
# ==============================================================================
def perform_strategic_analysis(df_sales, df_products, df_stores, df_customers, df_exchange):
    """
    Voert de kernanalyses uit: Merging, Valuta-conversie, KPI berekening.
    """
    print("\n--- FASE 2: Data Integratie en Verrijking ---")

    # --- STAP 1: Data Cleaning van Prijzen ---
    # FIX: We gebruiken r'\$' (raw string) om de SyntaxWarning te voorkomen
    cols_to_clean = ['Unit Cost USD', 'Unit Price USD']
    for col in cols_to_clean:
        if df_products[col].dtype == 'object':
            # Eerst naar string omzetten voor de zekerheid, dan vervangen
            df_products[col] = df_products[col].astype(str).replace({r'\$': '', ',': ''}, regex=True)
            # Dan naar numeriek, errors='coerce' maakt van niet-getallen NaN
            df_products[col] = pd.to_numeric(df_products[col], errors='coerce')

    # --- STAP 2: Data Verrijking (Left Joins) ---
    df_full = pd.merge(df_sales, df_products, on='ProductKey', how='left')
    df_full = pd.merge(df_full, df_stores, on='StoreKey', how='left')
    df_full = pd.merge(df_full, df_customers, on='CustomerKey', how='left')

    # --- STAP 3: Valuta Normalisatie ---
    df_exchange_renamed = df_exchange.rename(columns={'Date': 'Order Date', 'Currency': 'Currency Code', 'Exchange': 'ExchangeRate'})
    
    # Merge Sales met Exchange rates op Datum EN Valuta
    df_full = pd.merge(df_full, df_exchange_renamed, on=['Order Date', 'Currency Code'], how='left')
    
    # Vul ontbrekende wisselkoersen (vaak USD of missende data) met 1.0
    df_full['ExchangeRate'] = df_full['ExchangeRate'].fillna(1.0)

    # --- STAP 4: Berekeningen (KPI's) ---
    df_full['Total Revenue USD'] = df_full['Quantity'] * df_full['Unit Price USD']
    df_full['Total Cost USD'] = df_full['Quantity'] * df_full['Unit Cost USD']
    df_full['Total Profit USD'] = df_full['Total Revenue USD'] - df_full['Total Cost USD']
    
    # Marge %
    df_full['Profit Margin %'] = np.where(
        df_full['Total Revenue USD'] != 0, 
        (df_full['Total Profit USD'] / df_full['Total Revenue USD']) * 100, 
        0
    )

    return df_full

# ==============================================================================
# FASE 3: STRATEGISCHE VISUALISATIES
# ==============================================================================
def generate_visualizations(df_full):
    print("\n--- FASE 3: Strategische Inzichten Genereren ---")
    
    # ---------------------------------------------------------
    # ANALYSE 1: PRODUCT MATRIX (Volume vs Marge)
    # ---------------------------------------------------------
    try:
        cat_stats = df_full.groupby('Category').agg({
            'Total Revenue USD': 'sum',
            'Total Profit USD': 'sum',
            'Quantity': 'sum'
        }).sort_values(by='Total Revenue USD', ascending=False).reset_index()
        
        cat_stats['Margin %'] = (cat_stats['Total Profit USD'] / cat_stats['Total Revenue USD']) * 100
        
        print("\nPrestaties per Categorie (Top 5):")
        print(cat_stats.head(5))

        # Visualisatie: Bar + Line Chart
        fig, ax1 = plt.subplots(figsize=(12, 6))
        sns.barplot(data=cat_stats, x='Category', y='Total Revenue USD', color='skyblue', ax=ax1)
        ax1.set_ylabel('Totale Omzet (USD)', color='blue')
        
        ax2 = ax1.twinx()
        sns.lineplot(data=cat_stats, x='Category', y='Margin %', color='red', marker='o', ax=ax2, linewidth=3)
        ax2.set_ylabel('Marge %', color='red')
        
        plt.title('Strategische Paradox: Omzet vs. Winstmarge per Categorie')
        plt.grid(False)
        plt.show()
    except Exception as e:
        print(f"Kon grafiek 1 niet maken: {e}")

    # ---------------------------------------------------------
    # ANALYSE 2: WINKEL EFFICIËNTIE (Sales Density)
    # ---------------------------------------------------------
    try:
        physical_stores = df_full[(df_full['StoreKey'] != 0) & (df_full['Square Meters'].notna()) & (df_full['Square Meters'] > 0)]
        
        store_kpis = physical_stores.groupby(['StoreKey', 'State', 'Country', 'Square Meters']).agg({
            'Total Revenue USD': 'sum'
        }).reset_index()
        
        store_kpis['SalesPerSqMeter'] = store_kpis['Total Revenue USD'] / store_kpis['Square Meters']
        
        print("\nTop 5 Meest Efficiënte Winkels (Sales Density):")
        print(store_kpis.sort_values(by='SalesPerSqMeter', ascending=False).head(5))

        # Visualisatie: Scatter Plot
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=store_kpis, 
            x='Square Meters', 
            y='SalesPerSqMeter', 
            hue='Country', 
            size='Total Revenue USD', 
            sizes=(50, 500),
            palette='viridis'
        )
        plt.title('Winkelefficiëntie Matrix: Oppervlakte vs. Verkoopdichtheid')
        plt.xlabel('Winkeloppervlakte (m²)')
        plt.ylabel('Omzet per m² (USD)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Kon grafiek 2 niet maken: {e}")

    # ---------------------------------------------------------
    # ANALYSE 3: KLANT GENERATIES (Leeftijd)
    # ---------------------------------------------------------
    try:
        now = pd.Timestamp('now')
        df_full['Age'] = (now - df_full['Birthday']).dt.days // 365
        
        df_clean_age = df_full[(df_full['Age'] > 10) & (df_full['Age'] < 100)].copy()

        bins = [10, 25, 40, 55, 70, 100]
        labels = ['Gen Z (10-25)', 'Millennials (26-40)', 'Gen X (41-55)', 'Boomers (56-70)', 'Silent (70+)']
        df_clean_age['AgeGroup'] = pd.cut(df_clean_age['Age'], bins=bins, labels=labels)
        
        age_revenue = df_clean_age.groupby('AgeGroup', observed=False)['Total Revenue USD'].sum().reset_index()

        # Visualisatie: Bar Chart
        plt.figure(figsize=(10, 6))
        sns.barplot(data=age_revenue, x='AgeGroup', y='Total Revenue USD', palette='pastel')
        plt.title('Omzetbijdrage per Generatie')
        plt.ylabel('Totale Omzet (USD)')
        plt.xlabel('Generatie')
        plt.show()
    except Exception as e:
        print(f"Kon grafiek 3 niet maken: {e}")

# --- HOOFDPROGRAMMA ---
if __name__ == "__main__":
    df_sales, df_products, df_stores, df_customers, df_exchange = load_and_clean_data()
    
    if df_sales is not None:
        print("Data succesvol geladen. Start analyse...")
        df_final = perform_strategic_analysis(df_sales, df_products, df_stores, df_customers, df_exchange)
        generate_visualizations(df_final)
        print("Analyse voltooid.")