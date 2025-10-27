# sim_esperanza.py
# Simulación Monte Carlo de ventas diarias (Panadería La Esperanza)
# + Gráficas profesionales + Comparativa con La Espiga
# Autor: Teddy + GPT-5 Thinking

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import math
import sys

# ==========================
# 1) Configuración general
# ==========================
RUTA_EXCEL = Path(r"C:\Users\ulise\esperanza_panaderia.xlsx")
ITERACIONES = 1000

import time
SEMILLA = int(time.time())
OUTDIR = Path("salidas_sim")

# Estilo gráfico (limpio y consistente)
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.25,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.titlesize": 14,
    "font.size": 10,
})

def fmt_mxn(x, pos):
    return f"${x:,.0f}"

MXN = FuncFormatter(fmt_mxn)

# ==========================
# 2) Lectura y precios
# ==========================
def cargar_catalogo(ruta_excel: Path) -> pd.DataFrame:
    if not ruta_excel.exists():
        sys.exit(f"ERROR: No se encontró el archivo: {ruta_excel}")
    try:
        df = pd.read_excel(ruta_excel)
    except Exception as e:
        sys.exit(f"ERROR al leer el Excel: {e}")
    cols_req = {"producto_nombre", "categoria_nombre", "precio"}
    faltan = cols_req - set(df.columns)
    if faltan:
        sys.exit(f"ERROR: Faltan columnas {faltan}. "
                 "Debes tener: producto_nombre, categoria_nombre, precio")
    return df

def precios_referencia(df: pd.DataFrame) -> dict:
    def med_cat(cat):
        s = df[df["categoria_nombre"] == cat]["precio"].dropna()
        return float(s.median()) if len(s) else np.nan

    def med_name(pat):
        s = df[df["producto_nombre"].str.contains(pat, case=False, na=False)]["precio"]
        return float(s.median()) if len(s) else np.nan

    ref = {
        "pan_dulce": med_cat("Panadería Dulce"),
        "pan_salada": med_cat("Panadería Salada"),
        "bocadillo_cafeteria": med_cat("Cafeteria Alimentos"),
        "bebida_cafe": df[df["categoria_nombre"]=="Cafeteria Bebidas"]["precio"].mean(),
        "pastel": med_cat("Pasteleria"),
        "bolillo": med_name(r"\bbolillo\b"),
        "telera":  med_name(r"\btelera\b"),
        "baguette": med_name(r"\bbaguette\b"),
        "croissant": med_name(r"\bcroissant\b"),
    }

    # Fallbacks razonables si falta info en el Excel
    ref = {
        k: (v if (v == v and np.isfinite(v)) else {
            "pan_dulce": 19.0, "pan_salada": 10.0, "bocadillo_cafeteria": 63.0,
            "bebida_cafe": 33.0, "pastel": 360.0, "bolillo": 3.0, "telera": 3.0,
            "baguette": 24.0, "croissant": 38.0
        }.get(k, np.nan)) for k, v in ref.items()
    }
    return ref

# ==========================
# 3) Supuestos
# ==========================
SUPUESTOS = {
    "piezas_tradicional_dia": {"dist": "normal", "mean": 2000, "sd": 200},  # 70% dulce / 30% salada
    "cafes_dia": {"dist": "normal", "mean": 500, "sd": 60},
    "bocadillos_dia": {"dist": "normal", "mean": 100, "sd": 20},
    "pasteles_por_pedido_dia": {"dist": "poisson", "lam": 3},
    "trays_clientes_dia": {"dist": "normal", "mean": 50, "sd": 10},
    "mix_pan_dulce": 0.70,
}

def ticket_charola(n, rng):
    # Ticket variable entre $50–$300 (log-uniforme)
    u = rng.uniform(np.log(50), np.log(300), size=n)
    return np.exp(u)

def muestra(d, rng):
    if d["dist"] == "normal":
        return max(0, int(rng.normal(d["mean"], d["sd"])))
    if d["dist"] == "poisson":
        return int(rng.poisson(d["lam"]))
    return 0

# ==========================
# 4) Simulación
# ==========================
def simular_un_dia(rng, precios):
    piezas = muestra(SUPUESTOS["piezas_tradicional_dia"], rng)
    cafes = muestra(SUPUESTOS["cafes_dia"], rng)
    bocadillos = muestra(SUPUESTOS["bocadillos_dia"], rng)
    pasteles = muestra(SUPUESTOS["pasteles_por_pedido_dia"], rng)
    trays = muestra(SUPUESTOS["trays_clientes_dia"], rng)

    pan_dulce = int(SUPUESTOS["mix_pan_dulce"] * piezas)
    pan_salada = piezas - pan_dulce

    ingresos = {
        "Pan dulce (pieza)": pan_dulce * precios["pan_dulce"],
        "Pan salado (pieza)": pan_salada * precios["pan_salada"],
        "Cafés/bebidas": cafes * precios["bebida_cafe"],
        "Bocadillos": bocadillos * precios["bocadillo_cafeteria"],
        "Pasteles por pedido": pasteles * precios["pastel"],
        "Charolas (ticket 50-300)": float(ticket_charola(trays, rng).sum()),
    }
    conteos = {
        "piezas_pan_dulce": pan_dulce,
        "piezas_pan_salada": pan_salada,
        "cafes": cafes,
        "bocadillos": bocadillos,
        "pasteles": pasteles,
        "charolas": trays,
    }
    total = sum(ingresos.values())
    return ingresos, conteos, total

def correr_simulacion(precios):
    rng = np.random.default_rng(seed=SEMILLA)
    ingresos_hist, conteos_hist, totales = [], [], []
    for _ in range(ITERACIONES):
        inc, cts, tot = simular_un_dia(rng, precios)
        ingresos_hist.append(inc); conteos_hist.append(cts); totales.append(tot)
    df_ingresos = pd.DataFrame(ingresos_hist)
    df_conteos = pd.DataFrame(conteos_hist)
    serie_total = pd.Series(totales, name="Ingresos diarios (MXN)")
    return df_ingresos, df_conteos, serie_total

# ==========================
# 5) Comparativa Espiga
# ==========================
def comparativa_espiga(precios):
    produccion = {"bolillo_normal":700, "bolillo_pequeno":500, "baguette":200, "telera":200, "bizcocho":250}
    precios_basicos = {
        "bolillo_normal": precios.get("bolillo", 3.0),
        "bolillo_pequeno": 2.0,
        "baguette": precios.get("baguette", 24.0),
        "telera": precios.get("telera", 3.0),
        "bizcocho": precios.get("pan_dulce", 19.0)
    }
    ingresos = {k: produccion[k]*precios_basicos[k] for k in produccion}
    df = pd.DataFrame({
        "unidades": pd.Series(produccion),
        "precio_ref_MXN": pd.Series(precios_basicos),
        "ingreso_estimado_MXN": pd.Series(ingresos)
    })
    df.loc["TOTAL", "ingreso_estimado_MXN"] = df["ingreso_estimado_MXN"].sum()
    return df

# ==========================
# 6) Gráficas profesionales
# ==========================
# === Reemplazo: gráficas premium, bonitas y consistentes ===
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_pdf import PdfPages

def _fmt_mxn(x, pos): return f"${x:,.0f}"
def _fmt_pct(x, pos): return f"{100*x:,.0f}%"
MXN = FuncFormatter(_fmt_mxn)
PCT = FuncFormatter(_fmt_pct)

def graficas_profesionales(outdir: Path, serie_total: pd.Series,
                           ranking_unidades: pd.DataFrame,
                           ranking_ingresos: pd.DataFrame,
                           n_iter: int = ITERACIONES):
    outdir.mkdir(parents=True, exist_ok=True)

    # Estilo sobrio y profesional
    plt.style.use("tableau-colorblind10")
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.25,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.titlesize": 15,
        "font.size": 11,
        "legend.frameon": False,
    })

    figs = []

    # ---------- 1) Histograma + CDF + banda P5–P95 ----------
    xs = np.sort(serie_total.values)
    cdf = np.arange(1, len(xs)+1)/len(xs)
    p5, p50, p95 = np.percentile(xs, [5, 50, 95])

    fig, ax = plt.subplots()
    ax.hist(serie_total.values, bins=40, edgecolor="black", alpha=0.85, label="Frecuencia")
    ax.set_title("Ingresos diarios simulados")
    ax.set_xlabel("MXN por día"); ax.set_ylabel("Frecuencia")
    ax.xaxis.set_major_formatter(MXN)

    # Sombreado entre P5 y P95
    ax.axvspan(p5, p95, alpha=0.15)
    for val, lbl in [(p5,"P5"), (p50,"P50"), (p95,"P95")]:
        ax.axvline(val, linestyle="--", linewidth=1.6)
        ax.text(val, ax.get_ylim()[1]*0.92, f"{lbl}\n${val:,.0f}", ha="center")

    ax2 = ax.twinx()
    ax2.plot(xs, cdf, linewidth=2, label="CDF")
    ax2.set_ylabel("Acumulado"); ax2.set_ylim(0, 1.0)
    ax2.yaxis.set_major_formatter(PCT)

    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(outdir/"01_hist_cdf_ingresos.png", dpi=220)
    fig.savefig(outdir/"01_hist_cdf_ingresos.svg")
    figs.append(fig)

    # ---------- 2) Boxplot (presentación) ----------
    fig, ax = plt.subplots()
    bp = ax.boxplot(serie_total.values, vert=True, widths=0.4, patch_artist=True)
    for box in bp['boxes']: box.set(alpha=0.6)
    ax.set_title("Ingresos diarios — distribución (boxplot)")
    ax.set_ylabel("MXN por día"); ax.yaxis.set_major_formatter(MXN)
    ax.text(1.02, np.median(serie_total.values), f"Mediana: ${np.median(serie_total):,.0f}",
            va="center")
    fig.tight_layout()
    fig.savefig(outdir/"02_boxplot_ingresos.png", dpi=220)
    fig.savefig(outdir/"02_boxplot_ingresos.svg")
    figs.append(fig)

    # ---------- 3) Unidades por línea con IC-95% ----------
    # media ± 1.96*std/sqrt(n)
    medias = ranking_unidades["promedio_diario"]
    # re-calcular std de conteos para IC (usamos muestra de la simulación)
    # (si no tienes df_conteos aquí, aproximamos con Poisson: std ~ sqrt(media))
    approx_std = np.sqrt(medias.clip(lower=1))
    ci = 1.96 * (approx_std / np.sqrt(n_iter))

    fig, ax = plt.subplots()
    ax.bar(medias.index, medias.values, yerr=ci.values, capsize=5)
    ax.set_title("Unidades promedio por línea (IC-95%)")
    ax.set_ylabel("Piezas / día")
    plt.xticks(rotation=25, ha="right")
    for i, v in enumerate(medias.values):
        ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir/"03_bar_unidades_ic95.png", dpi=220)
    fig.savefig(outdir/"03_bar_unidades_ic95.svg")
    figs.append(fig)

    # ---------- 4) Pareto de ingresos ----------
    vin = ranking_ingresos.copy()
    vin["pct"] = vin["ingreso_promedio_diario_MXN"] / vin["ingreso_promedio_diario_MXN"].sum()
    vin = vin.sort_values("ingreso_promedio_diario_MXN", ascending=False)

    fig, ax = plt.subplots()
    ax.bar(vin.index, vin["ingreso_promedio_diario_MXN"])
    ax.set_ylabel("MXN/día"); ax.yaxis.set_major_formatter(MXN)
    ax.set_title("Pareto de ingresos por línea")
    plt.xticks(rotation=25, ha="right")

    ax2 = ax.twinx()
    ax2.plot(vin.index, vin["pct"].cumsum(), marker="o", linewidth=2)
    ax2.set_ylabel("Acumulado"); ax2.set_ylim(0, 1.02); ax2.yaxis.set_major_formatter(PCT)
    ax2.axhline(0.8, ls="--")
    # etiqueta en el cruce del 80%
    cruz = np.argmax(vin["pct"].cumsum().values >= 0.8)
    ax2.annotate("80% acumulado",
                 xy=(cruz, vin["pct"].cumsum().iloc[cruz]),
                 xytext=(cruz, 0.88),
                 arrowprops=dict(arrowstyle="->"))
    fig.tight_layout()
    fig.savefig(outdir/"04_pareto_ingresos.png", dpi=220)
    fig.savefig(outdir/"04_pareto_ingresos.svg")
    figs.append(fig)

    # ---------- 5) Waterfall (cascada) ----------
    fig, ax = plt.subplots()
    orden = vin.index.tolist()
    vals = vin["ingreso_promedio_diario_MXN"].values
    total = vals.sum()

    ax.bar(orden, vals)
    ax.bar("TOTAL", total, alpha=0.85)
    ax.set_title("Cascada: ingreso promedio por línea → TOTAL")
    ax.set_ylabel("MXN/día"); ax.yaxis.set_major_formatter(MXN)
    plt.xticks(rotation=25, ha="right")
    for i, v in enumerate(list(vals) + [total]):
        ax.text(i, v, f"${v:,.0f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir/"05_waterfall_ingresos.png", dpi=220)
    fig.savefig(outdir/"05_waterfall_ingresos.svg")
    figs.append(fig)

    # ---------- 6) Dashboard en PDF ----------
    with PdfPages(outdir/"dashboard_simulacion.pdf") as pdf:
        for f in figs:
            pdf.savefig(f, dpi=220)


# ==========================
# 7) Main
# ==========================
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    df_cat = cargar_catalogo(RUTA_EXCEL)
    precios = precios_referencia(df_cat)

    df_ingresos, df_conteos, serie_total = correr_simulacion(precios)

    # KPIs
    resumen = pd.DataFrame({
        "promedio_MXN": [serie_total.mean()],
        "p5_MXN": [np.percentile(serie_total, 5)],
        "p50_MXN": [np.percentile(serie_total, 50)],
        "p95_MXN": [np.percentile(serie_total, 95)],
        "std_MXN": [serie_total.std()],
    }).round(2)

    ranking_unidades = df_conteos.mean().sort_values(ascending=False).to_frame("promedio_diario").round(1)
    ranking_ingresos = df_ingresos.mean().sort_values(ascending=False).to_frame("ingreso_promedio_diario_MXN").round(2)

    # Guardar tablas
    resumen.to_csv(OUTDIR/"kpis_resumen.csv", index=False)
    ranking_unidades.to_csv(OUTDIR/"ranking_unidades.csv")
    ranking_ingresos.to_csv(OUTDIR/"ranking_ingresos.csv")
    pd.Series(precios, name="precio_MXN").to_frame().to_csv(OUTDIR/"precios_referencia.csv")

    # Comparativa Espiga
    df_espiga = comparativa_espiga(precios)
    df_espiga.to_csv(OUTDIR/"espiga_comparativa.csv")

    # Gráficas PRO
    graficas_profesionales(OUTDIR, serie_total, ranking_unidades, ranking_ingresos)

    # Resumen consola
    print("\n=== KPIs (MXN) ===")
    print(resumen.to_string(index=False))
    print("\n=== TOP INGRESOS (MXN/día) ===")
    print(ranking_ingresos.to_string())
    print("\n=== TOP UNIDADES (pzas/día) ===")
    print(ranking_unidades.to_string())
    print("\nArchivos exportados en:", OUTDIR.resolve())

if __name__ == "__main__":
    main()
