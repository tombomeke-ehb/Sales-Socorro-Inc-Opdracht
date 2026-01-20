import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==============================================================================
# CONFIGURATIE & STIJL
# ==============================================================================
sns.set(style="whitegrid")
sns.set_context("talk") # Zorgt voor professionele, leesbare letters
plt.rcParams['figure.figsize'] = (14, 8)

def run_full_audit():
    print("--- 1. DATA LADEN & INTEGRITEITSCHECK... ---")
    try:
        # CSV's inlezen
        df_sales = pd.read_csv('Sales.csv')
        df_products = pd.read_csv('Products.csv')
        df_stores = pd.read_csv('Stores.csv')
        df_customers = pd.read_csv('Customers.csv', encoding='ISO-8859-1') 
        
        # --- CRUCIALE STAP: KOLOMMEN HERNOEMEN OM MERGE FOUTEN TE VOORKOMEN ---
        df_stores = df_stores.rename(columns={'State': 'StoreState', 'Country': 'StoreCountry'})
        df_customers = df_customers.rename(columns={'State': 'CustomerState', 'Country': 'CustomerCountry'})

        # --- DATA CLEANING (ETL) ---
        # 1. Datum fix (1916 fout eruit)
        df_sales['Order Date'] = pd.to_datetime(df_sales['Order Date'], errors='coerce')
        df_sales = df_sales[df_sales['Order Date'].dt.year >= 2000].dropna(subset=['Order Date'])

        # 2. Prijzen fix ($ en , verwijderen)
        for col in ['Unit Cost USD', 'Unit Price USD']:
            if df_products[col].dtype == 'object':
                df_products[col] = df_products[col].astype(str).replace({r'\$': '', ',': ''}, regex=True)
                df_products[col] = pd.to_numeric(df_products[col], errors='coerce')

        # 3. Mergen (Samenvoegen tot één dataset)
        df = df_sales.merge(df_products, on='ProductKey', how='left')
        df = df.merge(df_stores, on='StoreKey', how='left')
        df = df.merge(df_customers, on='CustomerKey', how='left')

        # --- KPI BEREKENINGEN ---
        df['Total Revenue'] = df['Quantity'] * df['Unit Price USD']
        df['Total Cost'] = df['Quantity'] * df['Unit Cost USD']
        df['Total Profit'] = df['Total Revenue'] - df['Total Cost']
        
        # Leeftijd berekenen
        df['Age'] = (pd.Timestamp('now') - pd.to_datetime(df['Birthday'], errors='coerce')).dt.days // 365

        # ==============================================================================
        # BEWIJSVOERING VOOR SLIDE 2 (TERMINAL OUTPUT)
        # ==============================================================================
        total_rev = df['Total Revenue'].sum()
        total_profit = df['Total Profit'].sum()
        avg_margin = (total_profit / total_rev) * 100

        print("\n" + "="*50)
        print("     SOCORRO INC. - AUDIT RAPPORT (2025)")
        print("="*50)
        print(f"BEWIJS VOOR SLIDE 2:")
        print(f" -> Totale Omzet:      ${total_rev:,.2f}")
        print(f" -> Totale Winst:      ${total_profit:,.2f}")
        print(f" -> Bruto Marge:       {avg_margin:.2f}%")
        print(f" -> Aantal Orders:     {len(df):,}")
        print("="*50 + "\n")

        # ==============================================================================
        # DE GRAFIEKEN GENEREREN
        # ==============================================================================
        print("--- 2. GRAFIEKEN GENEREREN... ---")

        # --- GRAFIEK 1: SALES TREND (VOOR SLIDE 3) ---
        monthly_sales = df.set_index('Order Date').resample('M')['Total Revenue'].sum().reset_index()
        plt.figure(figsize=(14, 6))
        sns.lineplot(data=monthly_sales, x='Order Date', y='Total Revenue', color='navy', linewidth=3)
        plt.title('Sales Trend (2016-2021): Seizoensgebonden Groei', fontweight='bold')
        plt.ylabel('Omzet (USD)')
        plt.tight_layout()
        plt.show()

        # --- GRAFIEK 2: DE MATRIX PER CATEGORIE (VOOR SLIDE 4/5) ---
        cat_stats = df.groupby('Category').agg({
            'Total Revenue': 'sum', 'Total Profit': 'sum', 'Quantity': 'sum'
        }).reset_index()
        cat_stats['Margin %'] = (cat_stats['Total Profit'] / cat_stats['Total Revenue']) * 100

        plt.figure(figsize=(14, 9))
        sns.scatterplot(
            data=cat_stats, x='Total Revenue', y='Margin %', size='Total Profit', 
            sizes=(1000, 5000), hue='Category', alpha=0.7, legend=False
        )
        for i in range(cat_stats.shape[0]):
            plt.text(cat_stats['Total Revenue'][i], cat_stats['Margin %'][i], cat_stats['Category'][i], 
                     horizontalalignment='center', size='small', color='black', weight='bold')
        plt.axvline(cat_stats['Total Revenue'].mean(), color='red', linestyle='--', alpha=0.3)
        plt.axhline(cat_stats['Margin %'].mean(), color='red', linestyle='--', alpha=0.3)
        plt.title('STRATEGISCHE MATRIX: Categorie Analyse', fontweight='bold')
        plt.xlabel('Totale Omzet (Volume)')
        plt.ylabel('Winstmarge (%)')
        plt.tight_layout()
        plt.show()

        # --- GRAFIEK 3: TOP 5 CASH COWS (VOOR SLIDE 5) ---
        prod_stats = df.groupby('Product Name').agg({'Total Profit': 'sum', 'Total Revenue': 'sum'}).reset_index()
        top_5 = prod_stats.sort_values('Total Profit', ascending=False).head(5)
        top_5['ShortName'] = top_5['Product Name'].str[:40] + '...'
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=top_5, y='ShortName', x='Total Profit', palette='Greens_r')
        plt.title('TOP 5 WINNAARS (Hoogste Absolute Winst)', fontweight='bold')
        plt.xlabel('Winst (USD)')
        plt.tight_layout()
        plt.show()

        # --- GRAFIEK 4: FLOP 5 BLEEDERS (VOOR SLIDE 6) ---
        # We voegen Marge toe aan prod_stats en filteren op volume
        prod_stats_full = df.groupby(['Product Name']).agg({'Total Profit': 'sum', 'Total Revenue': 'sum', 'Quantity': 'sum'}).reset_index()
        prod_stats_full['Margin %'] = (prod_stats_full['Total Profit'] / prod_stats_full['Total Revenue']) * 100
        flop_5 = prod_stats_full[prod_stats_full['Quantity'] > 100].sort_values('Margin %', ascending=True).head(5)
        flop_5['ShortName'] = flop_5['Product Name'].str[:40] + '...'

        plt.figure(figsize=(12, 6))
        sns.barplot(data=flop_5, y='ShortName', x='Margin %', palette='Reds_r')
        plt.title('TOP 5 VERLIEZERS (Laagste Marge bij Hoog Volume)', fontweight='bold')
        plt.xlabel('Marge %')
        plt.axvline(x=50, color='red', linestyle='--', label='Target')
        plt.tight_layout()
        plt.show()

        # --- GRAFIEK 5: VASTGOED BEWIJS (VOOR SLIDE 7/8) ---
        stores = df[(df['StoreKey'] != 0) & (df['Square Meters'] > 0)]
        store_kpi = stores.groupby(['StoreCountry', 'Square Meters'])['Total Revenue'].sum().reset_index()
        store_kpi['Revenue_Per_SqM'] = store_kpi['Total Revenue'] / store_kpi['Square Meters']

        plt.figure(figsize=(14, 8))
        sns.scatterplot(data=store_kpi, x='Square Meters', y='Revenue_Per_SqM', hue='StoreCountry', s=300, palette='viridis')
        sns.regplot(data=store_kpi, x='Square Meters', y='Revenue_Per_SqM', scatter=False, color='red', line_kws={'linestyle':'--'})
        plt.title('VASTGOED ANALYSE: "Shrink to Grow" Bewijs', fontweight='bold')
        plt.xlabel('Winkelgrootte (m²)')
        plt.ylabel('Efficiëntie ($/m²)')
        plt.tight_layout()
        plt.show()

        # --- GRAFIEK 6: KLANT LEEFTIJD (VOOR SLIDE 8/9) ---
        df_age = df[(df['Age'] > 10) & (df['Age'] < 90)]
        plt.figure(figsize=(12, 6))
        sns.histplot(data=df_age, x='Age', bins=20, kde=True, color='purple')
        plt.title('KLANTPROFIEL: Leeftijdsverdeling', fontweight='bold')
        plt.xlabel('Leeftijd')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Er ging iets mis: {e}")

if __name__ == "__main__":
    run_full_audit()