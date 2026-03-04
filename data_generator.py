"""
CemenTrack - Generador de Datos Simulados
Genera movimientos realistas de cemento en silos para demo.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# ─── Configuración base de silos ───────────────────────────────────────────────
SILOS_CONFIG = {
    "Silo 1": {"tipo_cemento": "UG", "capacidad_ton": 1000},
    "Silo 2": {"tipo_cemento": "UG", "capacidad_ton": 1000},
    "Silo 3": {"tipo_cemento": "ART", "capacidad_ton": 1000},
    "Silo 4": {"tipo_cemento": "Ultra", "capacidad_ton": 950},
    "Silo 5": {"tipo_cemento": "Transicion", "capacidad_ton": 1050},
    "Silo 6": {"tipo_cemento": "ART", "capacidad_ton": 1000},
}

# Referencias de empaque y el tipo de cemento que consumen
REFERENCIAS_EMPAQUE = {
    "Cemento Gris 50kg": ["UG"],
    "Cemento Gris 25kg": ["UG"],
    "Cemento Gris Granel": ["UG"],
    "Cemento ART 42.5kg": ["ART"],
    "Cemento ART 50kg": ["ART"],
    "Cemento ART Granel": ["ART"],
    "Cemento Ultra 50kg": ["Ultra"],
    "Cemento Ultra 25kg": ["Ultra"],
    "Cemento Transicion 50kg": ["Transicion"],
    "Cemento Transicion Granel": ["Transicion"],
    "Cemento Multi 50kg": ["UG", "ART"],  # puede consumir de ambos
}


def generar_datos_simulados(dias: int = 90, seed: int = 42) -> pd.DataFrame:
    """
    Genera datos simulados de movimientos de cemento.

    Parámetros:
        dias: Cantidad de días históricos a simular
        seed: Semilla para reproducibilidad

    Retorna:
        DataFrame con columnas: fecha, tipo_movimiento, tipo_cemento, silo,
        cantidad_ton, referencia_empaque, turno
    """
    np.random.seed(seed)

    fecha_fin = datetime(2026, 3, 3)
    fecha_inicio = fecha_fin - timedelta(days=dias)

    registros = []

    fecha_actual = fecha_inicio
    while fecha_actual <= fecha_fin:
        dia_semana = fecha_actual.weekday()

        # Menos actividad en fines de semana
        factor_dia = 0.4 if dia_semana >= 5 else 1.0

        for silo_nombre, silo_info in SILOS_CONFIG.items():
            tipo_cemento = silo_info["tipo_cemento"]

            # ── ENTRADAS (recepción de cemento del molino) ──
            # Promedio 1-2 entradas por día por silo activo
            n_entradas = np.random.poisson(1.5 * factor_dia)
            for _ in range(n_entradas):
                cantidad = np.random.uniform(30, 120)  # toneladas por carga
                turno = np.random.choice(["Mañana", "Tarde", "Noche"],
                                         p=[0.4, 0.35, 0.25])
                registros.append({
                    "fecha": fecha_actual,
                    "tipo_movimiento": "Entrada",
                    "tipo_cemento": tipo_cemento,
                    "silo": silo_nombre,
                    "cantidad_ton": round(cantidad, 2),
                    "referencia_empaque": None,
                    "turno": turno,
                })

            # ── SALIDAS (consumo para empaque) ──
            # Buscar referencias que consumen este tipo de cemento
            refs_compatibles = [
                ref for ref, tipos in REFERENCIAS_EMPAQUE.items()
                if tipo_cemento in tipos
            ]

            n_salidas = np.random.poisson(2.5 * factor_dia)
            for _ in range(n_salidas):
                cantidad = np.random.uniform(15, 80)  # toneladas consumidas
                referencia = np.random.choice(refs_compatibles)
                turno = np.random.choice(["Mañana", "Tarde", "Noche"],
                                         p=[0.35, 0.40, 0.25])
                registros.append({
                    "fecha": fecha_actual,
                    "tipo_movimiento": "Salida",
                    "tipo_cemento": tipo_cemento,
                    "silo": silo_nombre,
                    "cantidad_ton": round(cantidad, 2),
                    "referencia_empaque": referencia,
                    "turno": turno,
                })

        fecha_actual += timedelta(days=1)

    df = pd.DataFrame(registros)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values(["fecha", "silo"]).reset_index(drop=True)

    return df


def generar_plantilla_vacia() -> pd.DataFrame:
    """Genera una plantilla vacía con el formato estándar."""
    return pd.DataFrame(columns=[
        "fecha", "tipo_movimiento", "tipo_cemento", "silo",
        "cantidad_ton", "referencia_empaque", "turno"
    ])


def guardar_datos(df: pd.DataFrame, ruta: str = "data"):
    """Guarda los datos simulados en CSV y Excel."""
    os.makedirs(ruta, exist_ok=True)

    csv_path = os.path.join(ruta, "movimientos_silos.csv")
    excel_path = os.path.join(ruta, "movimientos_silos.xlsx")
    plantilla_path = os.path.join(ruta, "plantilla_movimientos.xlsx")

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_excel(excel_path, index=False, sheet_name="Movimientos")

    # Plantilla vacía
    plantilla = generar_plantilla_vacia()
    plantilla.to_excel(plantilla_path, index=False, sheet_name="Movimientos")

    print(f"✅ Datos guardados en: {ruta}/")
    print(f"   - {csv_path} ({len(df)} registros)")
    print(f"   - {excel_path}")
    print(f"   - {plantilla_path} (plantilla vacía)")


if __name__ == "__main__":
    print("🏭 Generando datos simulados de CemenTrack...")
    df = generar_datos_simulados(dias=90)
    guardar_datos(df)
    print(f"\n📊 Resumen:")
    print(df.groupby(["tipo_movimiento", "tipo_cemento"])["cantidad_ton"].sum()
          .unstack().round(1))
