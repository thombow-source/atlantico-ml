import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# --- SEITEN-KONFIGURATION ---
st.set_page_config(
    page_title="Atlântico Wholesale - Kundensegmentierung",
    page_icon="🛒",
    layout="wide"
)

# --- DATEN & MODELL CACHEN ---
# -- Damit wird vermieden, das beim Reload der App im Browser
# -- das ML-Modell neu geladen wird und die Dataframes berechnet
# -- werden müssen. Dies geschieht, wie unten bei den Funktionen
# -- einmalig je 'Server-Run'
@st.cache_resource
def load_pipeline():
    return joblib.load("segmentation_pipeline.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("data_clustered.csv")

try:
    pipeline = load_pipeline()
    df_clustered = load_data()
except Exception as e:
    st.error(f"Fehler beim Laden der Dateien: {e}")
    st.stop()

spending_cols = pipeline["spending_cols"]

# 1. Deutsche Namen für die Kundensegmente (korrigiert nach deiner Modell-Analyse)
cluster_names = {
    0: "Frische- & Gastro-<br>Spezialisten",
    1: "Standard- /<br>Kleinabnehmer",
    2: "Einzelhandel /<br>Supermärkte"
}

df_clustered["segment_name"] = df_clustered["cluster"].map(cluster_names)

# 2. Deutsche Namen für die Produktkategorien
category_de = {
    "Fresh": "Frischeprodukte",
    "Milk": "Molkereiprodukte",
    "Grocery": "Lebensmittel",
    "Frozen": "Tiefkühlwaren",
    "Detergents_Paper": "Reinigung & Papier",
    "Delicassen": "Feinkost"
}

# --- PLOTLY-FUNKTIONEN CACHEN ---
# -- einmalig je 'Server-Run'
@st.cache_data
def create_plotly_heatmap(_df, _spending_cols, _category_de):
    # Berechnung auf den Originalspalten
    relative_spending = _df.groupby("segment_name")[_spending_cols].mean().div(_df[_spending_cols].mean())
    
    # Umbenennen der Spalten für die deutsche Anzeige
    relative_spending_de = relative_spending.rename(columns=_category_de)
    
    fig = px.imshow(
        relative_spending_de,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=1.0,
        labels=dict(x="Produktkategorie", y="Kundensegment", color="Index (1.0 = Schnitt)"),
        title="Relativer Ausgaben-Index je Segment (1.0 = Gesamtdurchschnitt)"
    )
    # Fette Beschriftung für die Produktkategorien (X-Achse) und Kundensegmente (Y-Achse)
    fig.update_xaxes(tickfont=dict(weight="bold"))
    fig.update_yaxes(tickfont=dict(weight="bold"))
    
    # fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=420)
    # Layout-Optimierung: Ausgewogenere Ränder und Schriftgrößen
    fig.update_layout(
        margin=dict(l=120, r=20, t=50, b=50), # Genug Platz links, ohne die Heatmap zu stauchen
        height=450,
        xaxis=dict(tickangle=-25), # Leicht schräge Kategoriereferenz für bessere Lesbarkeit
        yaxis=dict(autorange="reversed") # Behält die Reihenfolge bei
    )
    return fig

@st.cache_data
def create_plotly_radar(_df, _spending_cols, _category_de):
    rel_mean = _df.groupby("segment_name")[_spending_cols].mean().div(_df[_spending_cols].mean())
    
    # Übersetzte Kategorienamen für die Radar-Achsen
    de_categories = [_category_de[col] for col in _spending_cols]
    
    fig = go.Figure()
    
    for segment, row in rel_mean.iterrows():
        categories = de_categories + [de_categories[0]]
        values = row.tolist() + [row.tolist()[0]]
        
        text_labels = [f"{v:.2f}x" for v in values]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=segment,
            mode='lines+markers+text',
            text=text_labels,
            textposition="top center",
            textfont=dict(color="#006400", size=11, weight="bold"),
            hovertemplate="<b>%{theta}</b><br>Index: %{r:.2f}x Schnitt<extra></extra>"
        ))
        
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, 
                range=[0, max(rel_mean.values.max() * 1.25, 2.0)],
                ticksuffix="x",
                tickfont=dict(color="black")
            ),
            angularaxis=dict(
                tickfont=dict(weight="bold") # <- Fette Kategorienamen außen (Lebensmittel, etc.)
            )
        ),
        showlegend=True,
        title="Gesamtform der Segmente (Relativ zum Schnitt = 1.0x) <BR> [Resize mit Doppelklick]",
        margin=dict(l=60, r=60, t=50, b=50),
        height=450,
        dragmode="zoom"
    )
    return fig

# --- HEADER ---
st.title("🛒 Atlântico Wholesale: Kundensegmentierung")
st.markdown("Interaktive Übersicht zur Analyse von Kundenstrukturen und zur Klassifizierung von Neukunden.")

tab1, tab2 = st.tabs(["📊 Teil 1: Segmente kennenlernen", "🎯 Teil 2: Kunden klassifizieren"])

# ==========================================
# TEIL 1: DIE SEGMENTE KENNENLERNEN
# ==========================================
with tab1:
    st.header("Analyse der bestehenden Kundensegmente")
    
    col1, col2, col3 = st.columns(3)
    sizes = df_clustered["segment_name"].value_counts()
    cols = [col1, col2, col3]
    
    for idx, (seg_name, count) in enumerate(sizes.items()):
        pct = (count / len(df_clustered)) * 100
        if idx < len(cols):
            cols[idx].metric(label=seg_name, value=f"{count} Kunden", delta=f"{pct:.1f}% aller Kunden")

    st.markdown("---")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.plotly_chart(
            create_plotly_heatmap(df_clustered, spending_cols, category_de), 
            use_container_width=True
        )

    with col_right:
        st.plotly_chart(
            create_plotly_radar(df_clustered, spending_cols, category_de), 
            use_container_width=True,
            config={
                'displayModeBar': True,
                'modeBarButtonsToAdd': ['resetScale2d', 'hoverClosestPolar'],
                'toImageButtonOptions': {'format': 'svg'}
            }
        )

    with st.expander("🔍 Unter der Haube: Interaktive PCA-Projektion (2D Scatterplot)"):
        scaler = pipeline["scaler"]
        pca = pipeline["pca"]
        X_scaled = scaler.transform(df_clustered[spending_cols])
        X_pca = pca.transform(X_scaled)
        
        pca_df = pd.DataFrame(X_pca, columns=[f"Hauptkomponente {i+1}" for i in range(X_pca.shape[1])])
        pca_df["Kundensegment"] = df_clustered["segment_name"].values
        
        fig_pca = px.scatter(
            pca_df, x="Hauptkomponente 1", y="Hauptkomponente 2", color="Kundensegment",
            title="2D PCA-Scatterplot (Verteilung der Kunden im reduzierten Raum)",
            hover_data=[pca_df.index]
        )
        st.plotly_chart(fig_pca, use_container_width=True)

# ==========================================
# TEIL 2: EINEN KUNDEN KLASSIFIZIEREN
# ==========================================
with tab2:
    st.header("Neuen Kunden klassifizieren")
    st.write("Geben Sie die geschätzten Jahresausgaben (€) des neuen Kunden ein:")
    
    with st.form("customer_input_form"):
        col_a, col_b = st.columns(2)
        
        with col_a:
            fresh = st.number_input("Frischeprodukte (Fresh €)", min_value=0, value=10000, step=500)
            milk = st.number_input("Molkereiprodukte (Milk €)", min_value=0, value=3000, step=200)
            grocery = st.number_input("Lebensmittel (Grocery €)", min_value=0, value=5000, step=500)
            
        with col_b:
            frozen = st.number_input("Tiefkühlwaren (Frozen €)", min_value=0, value=2000, step=200)
            detergents = st.number_input("Reinigungsmittel & Papier (Detergents_Paper €)", min_value=0, value=1500, step=100)
            delicassen = st.number_input("Feinkost (Delicassen €)", min_value=0, value=1000, step=100)

        submitted = st.form_submit_button("Kunden klassifizieren", type="primary")

    if submitted:
        input_data = pd.DataFrame([{
            "Fresh": fresh,
            "Milk": milk,
            "Grocery": grocery,
            "Frozen": frozen,
            "Detergents_Paper": detergents,
            "Delicassen": delicassen
        }])[spending_cols]

        scaled_input = pipeline["scaler"].transform(input_data)
        pca_input = pipeline["pca"].transform(scaled_input)
        cluster_pred = int(pipeline["kmeans"].predict(pca_input)[0])

        # das BR-Tag für die bessere Darstellung in den Plots müssen wir hier wieder löschen
        assigned_segment = cluster_names.get(cluster_pred, f"Cluster {cluster_pred}").replace('<br>','')

        st.success(f"### Zuordnung: **{assigned_segment}**")
        st.info(f"**Gesamtausgabe des Kunden:** {input_data.sum(axis=1).values[0]:,.0f} €")