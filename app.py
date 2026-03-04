"""
CemenTrack
Dashboard de Control de Niveles de Cemento en Silos
con Pronostico de Consumo (Data Science)

Autor: Desarrollado con vibe coding
Contexto: Maestria en Analitica de Datos - Ing. Industrial
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import io

from data_generator import (
    generar_datos_simulados,
    generar_plantilla_vacia,
    SILOS_CONFIG,
    REFERENCIAS_EMPAQUE,
)

# ===========================================================================
# CONFIGURACION DE PAGINA
# ===========================================================================
st.set_page_config(
    page_title="CemenTrack",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Colores por tipo de cemento
COLORES_CEMENTO = {
    "UG": "#1f77b4",
    "ART": "#2ca02c",
    "Ultra": "#ff7f0e",
    "Transicion": "#9467bd",
}

UMBRAL_CRITICO = 0.20
UMBRAL_BAJO = 0.35


# ===========================================================================
# FUNCIONES AUXILIARES
# ===========================================================================

def calcular_niveles_actuales(df, config):
    """
    Calcula el nivel actual de cada silo basado en entradas - salidas.
    Asume un nivel inicial del 60% de capacidad.
    """
    niveles = []
    for silo, info in config.items():
        capacidad = info["capacidad_ton"]
        nivel_inicial = capacidad * 0.6

        df_silo = df[df["silo"] == silo]
        entradas = df_silo[df_silo["tipo_movimiento"] == "Entrada"]["cantidad_ton"].sum()
        salidas = df_silo[df_silo["tipo_movimiento"] == "Salida"]["cantidad_ton"].sum()

        nivel_actual = nivel_inicial + entradas - salidas
        nivel_actual = max(0, min(nivel_actual, capacidad))

        pct = nivel_actual / capacidad
        if pct <= UMBRAL_CRITICO:
            estado = "🔴 Critico"
        elif pct <= UMBRAL_BAJO:
            estado = "🟡 Bajo"
        else:
            estado = "🟢 Normal"

        niveles.append({
            "silo": silo,
            "tipo_cemento": info["tipo_cemento"],
            "capacidad_ton": capacidad,
            "nivel_actual_ton": round(nivel_actual, 1),
            "porcentaje": round(pct * 100, 1),
            "estado": estado,
        })
    return pd.DataFrame(niveles)


def crear_gauge_silo(nombre, tipo_cemento, nivel_pct, nivel_ton, capacidad):
    """Crea un gauge (indicador circular) para un silo."""
    color = COLORES_CEMENTO.get(tipo_cemento, "#636363")

    if nivel_pct <= UMBRAL_CRITICO * 100:
        bar_color = "#d62728"
    elif nivel_pct <= UMBRAL_BAJO * 100:
        bar_color = "#ffdd57"
    else:
        bar_color = color

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=nivel_pct,
        number={"suffix": "%", "font": {"size": 28}},
        title={
            "text": f"{nombre}<br><span style='font-size:14px;color:{color}'>{tipo_cemento}</span>",
            "font": {"size": 16},
        },
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": bar_color},
            "bgcolor": "white",
            "borderwidth": 2,
            "steps": [
                {"range": [0, UMBRAL_CRITICO * 100], "color": "#ffcccc"},
                {"range": [UMBRAL_CRITICO * 100, UMBRAL_BAJO * 100], "color": "#fff3cd"},
                {"range": [UMBRAL_BAJO * 100, 100], "color": "#d4edda"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 2},
                "thickness": 0.75,
                "value": UMBRAL_CRITICO * 100,
            },
        },
    ))

    fig.add_annotation(
        text=f"{nivel_ton} / {capacidad} ton",
        x=0.5, y=-0.05,
        showarrow=False,
        font=dict(size=12, color="gray"),
    )

    fig.update_layout(height=250, margin=dict(l=20, r=20, t=60, b=30))
    return fig


def crear_grafico_barras_silos(df_niveles):
    """Grafico de barras horizontales con nivel de todos los silos."""
    fig = go.Figure()

    for _, row in df_niveles.iterrows():
        color = COLORES_CEMENTO.get(row["tipo_cemento"], "#636363")
        fig.add_trace(go.Bar(
            y=[row["silo"]],
            x=[row["porcentaje"]],
            orientation="h",
            name=f"{row['silo']} ({row['tipo_cemento']})",
            marker_color=color,
            text=f"{row['porcentaje']}% - {row['nivel_actual_ton']} ton",
            textposition="inside",
            showlegend=False,
        ))

    fig.add_shape(type="line", x0=UMBRAL_CRITICO * 100, x1=UMBRAL_CRITICO * 100,
                  y0=-0.5, y1=len(df_niveles) - 0.5,
                  line=dict(dash="dash", color="red", width=1))
    fig.add_shape(type="line", x0=UMBRAL_BAJO * 100, x1=UMBRAL_BAJO * 100,
                  y0=-0.5, y1=len(df_niveles) - 0.5,
                  line=dict(dash="dash", color="orange", width=1))

    fig.add_annotation(x=UMBRAL_CRITICO * 100, y=len(df_niveles) - 0.5,
                       text="Critico (20%)", showarrow=False, yshift=12,
                       font=dict(color="red", size=10))
    fig.add_annotation(x=UMBRAL_BAJO * 100, y=len(df_niveles) - 0.5,
                       text="Bajo (35%)", showarrow=False, yshift=12,
                       font=dict(color="orange", size=10))

    fig.update_layout(
        title="Nivel de Llenado por Silo",
        xaxis_title="Porcentaje de Llenado (%)",
        xaxis=dict(range=[0, 105]),
        height=350,
        margin=dict(l=20, r=20, t=50, b=30),
    )
    return fig


def pronosticar_consumo(df, tipo_cemento, dias_forecast=14):
    """
    Pronostico de consumo diario por tipo de cemento usando
    Suavizacion Exponencial (Holt-Winters).
    """
    df_tipo = df[
        (df["tipo_cemento"] == tipo_cemento) &
        (df["tipo_movimiento"] == "Salida")
    ].copy()

    if df_tipo.empty:
        return pd.DataFrame()

    consumo_diario = (
        df_tipo.groupby(df_tipo["fecha"].dt.date)["cantidad_ton"]
        .sum().reset_index()
    )
    consumo_diario.columns = ["fecha", "consumo_ton"]
    consumo_diario["fecha"] = pd.to_datetime(consumo_diario["fecha"])
    consumo_diario = consumo_diario.set_index("fecha").asfreq("D", fill_value=0)

    if len(consumo_diario) < 14:
        promedio = consumo_diario["consumo_ton"].mean()
        fechas_futuras = pd.date_range(
            start=consumo_diario.index[-1] + timedelta(days=1),
            periods=dias_forecast, freq="D"
        )
        forecast_df = pd.DataFrame({
            "fecha": fechas_futuras,
            "consumo_pronosticado": [promedio] * dias_forecast,
            "tipo": "Pronostico",
        })
    else:
        try:
            model = ExponentialSmoothing(
                consumo_diario["consumo_ton"],
                trend="add",
                seasonal="add",
                seasonal_periods=7,
            ).fit(optimized=True)
            forecast = model.forecast(dias_forecast)
            forecast = forecast.clip(lower=0)

            forecast_df = pd.DataFrame({
                "fecha": forecast.index,
                "consumo_pronosticado": forecast.values.round(1),
                "tipo": "Pronostico",
            })
        except Exception:
            promedio = consumo_diario["consumo_ton"].rolling(7).mean().iloc[-1]
            fechas_futuras = pd.date_range(
                start=consumo_diario.index[-1] + timedelta(days=1),
                periods=dias_forecast, freq="D"
            )
            forecast_df = pd.DataFrame({
                "fecha": fechas_futuras,
                "consumo_pronosticado": [promedio] * dias_forecast,
                "tipo": "Pronostico",
            })

    historico_df = consumo_diario.reset_index()
    historico_df.columns = ["fecha", "consumo_pronosticado"]
    historico_df["tipo"] = "Historico"

    resultado = pd.concat([historico_df, forecast_df], ignore_index=True)
    resultado["tipo_cemento"] = tipo_cemento
    return resultado


def calcular_dias_restantes(df_niveles, df):
    """Estima dias hasta vaciado de cada silo basado en consumo promedio."""
    resultados = []
    for _, row in df_niveles.iterrows():
        silo = row["silo"]
        df_silo = df[
            (df["silo"] == silo) & (df["tipo_movimiento"] == "Salida")
        ]

        if not df_silo.empty:
            consumo_diario_prom = (
                df_silo.groupby(df_silo["fecha"].dt.date)["cantidad_ton"]
                .sum().mean()
            )
            if consumo_diario_prom > 0:
                dias = row["nivel_actual_ton"] / consumo_diario_prom
            else:
                dias = float("inf")
        else:
            consumo_diario_prom = 0
            dias = float("inf")

        resultados.append({
            "silo": silo,
            "tipo_cemento": row["tipo_cemento"],
            "nivel_actual_ton": row["nivel_actual_ton"],
            "consumo_diario_prom": round(consumo_diario_prom, 1),
            "dias_restantes": round(dias, 1) if dias != float("inf") else "Sin consumo",
            "fecha_estimada_vaciado": (
                (datetime(2026, 3, 3) + timedelta(days=dias)).strftime("%Y-%m-%d")
                if dias != float("inf") and dias < 365
                else "N/A"
            ),
        })
    return pd.DataFrame(resultados)


# ===========================================================================
# SIDEBAR
# ===========================================================================

st.sidebar.title("🏭 CemenTrack")
st.sidebar.markdown("**Control de Niveles de Silos de Cemento**")
st.sidebar.markdown("---")

fuente_datos = st.sidebar.radio(
    "Fuente de datos",
    ["Datos Simulados (Demo)", "Cargar archivo CSV/Excel"],
)

if fuente_datos == "Datos Simulados (Demo)":
    dias_sim = st.sidebar.slider("Dias simulados", 30, 180, 90)
    df = generar_datos_simulados(dias=dias_sim)
    st.sidebar.success(f"{len(df)} movimientos simulados")
else:
    archivo = st.sidebar.file_uploader(
        "Sube tu archivo",
        type=["csv", "xlsx"],
        help="Debe tener las columnas: fecha, tipo_movimiento, tipo_cemento, silo, cantidad_ton"
    )
    if archivo is not None:
        if archivo.name.endswith(".csv"):
            df = pd.read_csv(archivo, parse_dates=["fecha"])
        else:
            df = pd.read_excel(archivo, parse_dates=["fecha"])
        st.sidebar.success(f"{len(df)} registros cargados")
    else:
        st.sidebar.warning("Sube un archivo para comenzar")
        df = None

st.sidebar.markdown("---")
st.sidebar.subheader("Configuracion de Silos")

config_silos = {}
tipos_cemento = ["UG", "ART", "Ultra", "Transicion"]

with st.sidebar.expander("Editar asignacion de silos", expanded=False):
    for silo, info in SILOS_CONFIG.items():
        tipo = st.selectbox(
            f"{silo} - Tipo",
            tipos_cemento,
            index=tipos_cemento.index(info["tipo_cemento"]),
            key=f"tipo_{silo}",
        )
        cap = st.number_input(
            f"{silo} - Capacidad (ton)",
            min_value=100,
            max_value=5000,
            value=info["capacidad_ton"],
            step=50,
            key=f"cap_{silo}",
        )
        config_silos[silo] = {"tipo_cemento": tipo, "capacidad_ton": cap}

if not config_silos:
    config_silos = SILOS_CONFIG.copy()

st.sidebar.markdown("---")
dias_forecast = st.sidebar.slider("Dias de pronostico", 7, 30, 14)


# ===========================================================================
# CONTENIDO PRINCIPAL
# ===========================================================================

st.title("🏭 CemenTrack")
st.caption("Dashboard de Control de Niveles de Cemento en Silos | Pronostico de Consumo")

if df is None:
    st.info("Carga un archivo de datos desde el panel lateral para comenzar.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    "Dashboard de Silos",
    "Pronostico de Consumo",
    "Datos y Analisis",
    "Plantilla Estandar",
])


# ===========================================================================
# TAB 1: DASHBOARD DE SILOS
# ===========================================================================
with tab1:
    st.header("Estado Actual de los Silos")

    df_niveles = calcular_niveles_actuales(df, config_silos)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    total_capacidad = df_niveles["capacidad_ton"].sum()
    total_actual = df_niveles["nivel_actual_ton"].sum()
    silos_criticos = len(df_niveles[df_niveles["estado"].str.contains("Critico")])
    silos_bajos = len(df_niveles[df_niveles["estado"].str.contains("Bajo")])

    col_m1.metric("Capacidad Total", f"{total_capacidad:,.0f} ton")
    col_m2.metric("Inventario Actual", f"{total_actual:,.0f} ton",
                  f"{total_actual/total_capacidad*100:.1f}%")
    col_m3.metric("Silos Criticos", silos_criticos,
                  delta=f"{silos_criticos}" if silos_criticos > 0 else "0",
                  delta_color="inverse")
    col_m4.metric("Silos Bajos", silos_bajos,
                  delta=f"{silos_bajos}" if silos_bajos > 0 else "0",
                  delta_color="inverse")

    st.markdown("---")

    st.subheader("Nivel Individual por Silo")
    cols = st.columns(3)
    for i, (_, row) in enumerate(df_niveles.iterrows()):
        with cols[i % 3]:
            fig = crear_gauge_silo(
                row["silo"], row["tipo_cemento"],
                row["porcentaje"], row["nivel_actual_ton"],
                row["capacidad_ton"]
            )
            st.plotly_chart(fig, key=f"gauge_{i}")

    st.markdown("---")

    st.plotly_chart(
        crear_grafico_barras_silos(df_niveles),
        key="barras_silos",
    )

    st.subheader("Estimacion de Dias Restantes")
    df_dias = calcular_dias_restantes(df_niveles, df)

    for _, row in df_dias.iterrows():
        if row["dias_restantes"] != "Sin consumo" and isinstance(row["dias_restantes"], (int, float)):
            if row["dias_restantes"] <= 3:
                st.error(
                    f"**{row['silo']}** ({row['tipo_cemento']}): "
                    f"Se estima vaciado en **{row['dias_restantes']} dias** "
                    f"({row['fecha_estimada_vaciado']})"
                )
            elif row["dias_restantes"] <= 7:
                st.warning(
                    f"**{row['silo']}** ({row['tipo_cemento']}): "
                    f"Se estima vaciado en **{row['dias_restantes']} dias** "
                    f"({row['fecha_estimada_vaciado']})"
                )

    st.dataframe(df_dias, hide_index=True)


# ===========================================================================
# TAB 2: PRONOSTICO DE CONSUMO
# ===========================================================================
with tab2:
    st.header("Pronostico de Consumo por Tipo de Cemento")
    st.markdown(
        "Modelo: **Suavizacion Exponencial (Holt-Winters)** con estacionalidad semanal. "
        "Se pronostican los proximos dias de consumo por tipo de cemento."
    )

    tipos_disponibles = df["tipo_cemento"].unique().tolist()
    tipo_seleccionado = st.selectbox(
        "Selecciona tipo de cemento",
        ["Todos"] + tipos_disponibles,
    )

    if tipo_seleccionado == "Todos":
        tipos_a_mostrar = tipos_disponibles
    else:
        tipos_a_mostrar = [tipo_seleccionado]

    for tipo in tipos_a_mostrar:
        st.subheader(f"Cemento {tipo}")
        df_forecast = pronosticar_consumo(df, tipo, dias_forecast)

        if df_forecast.empty:
            st.warning(f"No hay datos de consumo para {tipo}")
            continue

        fig = px.line(
            df_forecast,
            x="fecha",
            y="consumo_pronosticado",
            color="tipo",
            title=f"Consumo Diario - Cemento {tipo}",
            labels={
                "consumo_pronosticado": "Consumo (ton)",
                "fecha": "Fecha",
                "tipo": "Tipo de dato",
            },
            color_discrete_map={
                "Historico": COLORES_CEMENTO.get(tipo, "#636363"),
                "Pronostico": "#d62728",
            },
        )

        fig.update_layout(height=400)

        # Linea divisoria entre historico y pronostico
        fecha_division = df_forecast[
            df_forecast["tipo"] == "Pronostico"
        ]["fecha"].min()
        if pd.notna(fecha_division):
            fecha_str = str(fecha_division)
            fig.add_shape(
                type="line",
                x0=fecha_str, x1=fecha_str,
                y0=0, y1=1,
                yref="paper",
                line=dict(dash="dash", color="gray", width=2),
            )
            fig.add_annotation(
                x=fecha_str, y=1, yref="paper",
                text="Inicio pronostico",
                showarrow=False,
                font=dict(color="gray", size=11),
                yshift=10,
            )

        st.plotly_chart(fig, key=f"forecast_{tipo}")

        # Metricas del pronostico
        df_hist = df_forecast[df_forecast["tipo"] == "Historico"]
        df_pred = df_forecast[df_forecast["tipo"] == "Pronostico"]

        if not df_pred.empty and not df_hist.empty:
            col1, col2, col3 = st.columns(3)
            prom_hist = df_hist["consumo_pronosticado"].mean()
            prom_pred = df_pred["consumo_pronosticado"].mean()
            total_pred = df_pred["consumo_pronosticado"].sum()

            col1.metric(
                "Consumo promedio historico",
                f"{prom_hist:.1f} ton/dia",
            )
            col2.metric(
                "Consumo promedio pronosticado",
                f"{prom_pred:.1f} ton/dia",
                f"{((prom_pred - prom_hist) / prom_hist * 100):.1f}%",
            )
            col3.metric(
                f"Total proximos {dias_forecast} dias",
                f"{total_pred:,.0f} ton",
            )

        st.markdown("---")


# ===========================================================================
# TAB 3: DATOS Y ANALISIS
# ===========================================================================
with tab3:
    st.header("Exploracion de Datos")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Movimientos por Tipo de Cemento")
        df_resumen = (
            df.groupby(["tipo_movimiento", "tipo_cemento"])["cantidad_ton"]
            .sum().reset_index()
        )
        fig = px.bar(
            df_resumen,
            x="tipo_cemento",
            y="cantidad_ton",
            color="tipo_movimiento",
            barmode="group",
            title="Entradas vs Salidas por Tipo de Cemento",
            color_discrete_map={"Entrada": "#2ca02c", "Salida": "#d62728"},
            labels={"cantidad_ton": "Toneladas", "tipo_cemento": "Tipo Cemento"},
        )
        st.plotly_chart(fig, key="resumen_tipo")

    with col2:
        st.subheader("Distribucion por Silo")
        df_silo = (
            df[df["tipo_movimiento"] == "Salida"]
            .groupby("silo")["cantidad_ton"]
            .sum().reset_index()
        )
        fig = px.pie(
            df_silo,
            values="cantidad_ton",
            names="silo",
            title="Distribucion de Consumo por Silo",
        )
        st.plotly_chart(fig, key="pie_silo")

    st.subheader("Consumo por Referencia de Empaque")
    df_ref = (
        df[df["referencia_empaque"].notna()]
        .groupby("referencia_empaque")["cantidad_ton"]
        .sum().sort_values(ascending=True).reset_index()
    )
    fig = px.bar(
        df_ref,
        y="referencia_empaque",
        x="cantidad_ton",
        orientation="h",
        title="Consumo Total por Referencia de Empaque",
        labels={"cantidad_ton": "Toneladas", "referencia_empaque": "Referencia"},
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, key="ref_empaque")

    st.subheader("Tendencia de Consumo Diario")
    df_tendencia = (
        df[df["tipo_movimiento"] == "Salida"]
        .groupby([df["fecha"].dt.date, "tipo_cemento"])["cantidad_ton"]
        .sum().reset_index()
    )
    df_tendencia.columns = ["fecha", "tipo_cemento", "consumo_ton"]

    fig = px.line(
        df_tendencia,
        x="fecha",
        y="consumo_ton",
        color="tipo_cemento",
        title="Consumo Diario por Tipo de Cemento",
        color_discrete_map=COLORES_CEMENTO,
        labels={"consumo_ton": "Consumo (ton)", "fecha": "Fecha"},
    )
    st.plotly_chart(fig, key="tendencia")

    with st.expander("Ver datos crudos"):
        st.dataframe(df, height=400)


# ===========================================================================
# TAB 4: PLANTILLA ESTANDAR
# ===========================================================================
with tab4:
    st.header("Plantilla Estandar de Captura")
    st.markdown("""
    Esta plantilla estandariza la captura de movimientos de cemento **por silo individual**, 
    resolviendo el problema de que el ERP trata la bodega de silos como una sola bolsa.
    
    ### Columnas del formato:
    | Columna | Descripcion | Ejemplo |
    |---------|-------------|---------|
    | fecha | Fecha del movimiento | 2026-03-03 |
    | tipo_movimiento | Entrada o Salida | Entrada |
    | tipo_cemento | Tipo de cemento | UG, ART, Ultra, Transicion |
    | silo | Silo especifico | Silo 1, Silo 2, ... |
    | cantidad_ton | Cantidad en toneladas | 75.5 |
    | referencia_empaque | Referencia (solo salidas) | Cemento Gris 50kg |
    | turno | Turno de operacion | Manana, Tarde, Noche |
    """)

    plantilla_ejemplo = pd.DataFrame([
        {
            "fecha": "2026-03-03",
            "tipo_movimiento": "Entrada",
            "tipo_cemento": "UG",
            "silo": "Silo 1",
            "cantidad_ton": 85.0,
            "referencia_empaque": "",
            "turno": "Manana",
        },
        {
            "fecha": "2026-03-03",
            "tipo_movimiento": "Salida",
            "tipo_cemento": "UG",
            "silo": "Silo 1",
            "cantidad_ton": 45.0,
            "referencia_empaque": "Cemento Gris 50kg",
            "turno": "Tarde",
        },
    ])

    st.subheader("Ejemplo de registros:")
    st.dataframe(plantilla_ejemplo, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        plantilla_vacia = generar_plantilla_vacia()
        csv_buffer = plantilla_vacia.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Descargar Plantilla (CSV)",
            csv_buffer,
            file_name="plantilla_cementrack.csv",
            mime="text/csv",
        )

    with col2:
        excel_buffer = io.BytesIO()
        plantilla_vacia.to_excel(excel_buffer, index=False, sheet_name="Movimientos")
        st.download_button(
            "Descargar Plantilla (Excel)",
            excel_buffer.getvalue(),
            file_name="plantilla_cementrack.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    st.markdown("---")
    st.subheader("Descargar Datos de Demo")

    if df is not None:
        col1, col2 = st.columns(2)
        with col1:
            csv_data = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Datos Simulados (CSV)",
                csv_data,
                file_name="datos_demo_cementrack.csv",
                mime="text/csv",
            )
        with col2:
            excel_buffer2 = io.BytesIO()
            df.to_excel(excel_buffer2, index=False, sheet_name="Movimientos")
            st.download_button(
                "Datos Simulados (Excel)",
                excel_buffer2.getvalue(),
                file_name="datos_demo_cementrack.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


# ===========================================================================
# FOOTER
# ===========================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 12px;'>"
    "CemenTrack v1.0 | Dashboard de Control de Silos | "
    "Maestria en Analitica de Datos | 2026"
    "</div>",
    unsafe_allow_html=True,
)
