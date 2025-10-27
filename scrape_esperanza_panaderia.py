
"""
Scraper Panadería La Esperanza (Angular/JS) usando Selenium + BeautifulSoup
- Renderiza cada subcategoría en modo headless, hace scroll hasta el final
- Extrae: nombre, precio, moneda, URL e imagen
- Exporta: esperanza_panaderia.csv y esperanza_panaderia.xlsx
"""

import re
import time
import random
import logging
from dataclasses import dataclass, asdict
from typing import List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup as BS

# ---------- Selenium ----------
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.edge.service import Service as EdgeService

from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.microsoft import EdgeChromiumDriverManager

# ---------- Pandas----------
try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

# ---------------- Config ----------------
BASE = "https://esperanza.mx"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/123.0.0.0 Safari/537.36"
}
SLEEP_BETWEEN = (0.8, 1.8)

# Cambia a "edge" si prefieres usar Microsoft Edge:
BROWSER = "chrome"   # "chrome" o "edge"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)

# Lista manual de subcategorias bajo /catalogo-productos/panaderia/...
CATS = [
    {"name": "Leche",               "slug": "leche",               "url": "https://esperanza.mx/catalogo-productos/panaderia/leche"},
    {"name": "Día de Muertos",      "slug": "dia-de-muertos",      "url": "https://esperanza.mx/catalogo-productos/panaderia/dia-de-muertos"},
    {"name": "Panadería Salada",    "slug": "panaderia-salada",    "url": "https://esperanza.mx/catalogo-productos/panaderia/panaderia-salada"},
    {"name": "Panadería Dulce",     "slug": "panaderia-dulce",     "url": "https://esperanza.mx/catalogo-productos/panaderia/panaderia-dulce"},
    {"name": "Bolleria Europea",    "slug": "bolleria-europea",    "url": "https://esperanza.mx/catalogo-productos/panaderia/bolleria-europea"},
    {"name": "Pasteleria",          "slug": "pasteleria",          "url": "https://esperanza.mx/catalogo-productos/panaderia/pasteleria-"},
    {"name": "Tartas",              "slug": "tartas",              "url": "https://esperanza.mx/catalogo-productos/panaderia/tartas"},
    {"name": "Tartas Mini",         "slug": "tartas-mini",         "url": "https://esperanza.mx/catalogo-productos/panaderia/tartas-mini"},
    {"name": "Postres",             "slug": "postres",             "url": "https://esperanza.mx/catalogo-productos/panaderia/postres"},
    {"name": "Productos Empaquetados","slug":"productos-empaquetados","url":"https://esperanza.mx/catalogo-productos/panaderia/productos-empaquetados"},
    {"name": "Helados",             "slug": "helados",             "url": "https://esperanza.mx/catalogo-productos/panaderia/helados"},
    {"name": "Bocadillos",          "slug": "bocadillos",          "url": "https://esperanza.mx/catalogo-productos/panaderia/bocadillos"},
    {"name": "Cafeteria Alimentos", "slug": "cafeteria-alimentos", "url": "https://esperanza.mx/catalogo-productos/panaderia/cafeteria-alimentos"},
    {"name": "Cafeteria Bebidas",   "slug": "cafeteria-bebidas",   "url": "https://esperanza.mx/catalogo-productos/panaderia/cafeteria-bebidas"},
    {"name": "Frappes y Malteadas", "slug": "frappes-y-malteadas", "url": "https://esperanza.mx/catalogo-productos/panaderia/frappes-y-malteadas"},
    {"name": "Congelados",          "slug": "congelados",          "url": "https://esperanza.mx/catalogo-productos/panaderia/congelados"},
    {"name": "Tamales",             "slug": "tamales",             "url": "https://esperanza.mx/catalogo-productos/panaderia/tamales"},
    {"name": "Pizzas",              "slug": "pizzas",              "url": "https://esperanza.mx/catalogo-productos/panaderia/pizzas"},
    {"name": "Rosticería",          "slug": "rosticeria",          "url": "https://esperanza.mx/catalogo-productos/panaderia/rosticeria"},
]

# -------------- Utils -------------------
def jitter():
    time.sleep(random.uniform(*SLEEP_BETWEEN))

def clean_text(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip()

def parse_price(raw: str):
    """Devuelve (precio_float, moneda) a partir de '$24.50' etc."""
    s = clean_text(raw)
    if not s:
        return None, None
    moneda = "MXN" if "$" in s else None
    s2 = re.sub(r"[^\d.,-]", "", s)
    if s2.count(",") == 1 and s2.count(".") == 0:
        s2 = s2.replace(",", ".")
    try:
        return float(s2), moneda or "MXN"
    except Exception:
        return None, moneda

@dataclass
class Product:
    categoria_slug: str
    categoria_nombre: str
    producto_nombre: str
    precio: Optional[float]
    moneda: Optional[str]
    url_producto: str
    url_imagen: Optional[str]

# -------------- Selenium driver -------------------
_driver = None

def get_driver():
    """
    Crea un driver headless para Chrome (por defecto) o Edge (opcional),
    usando Service(...) + webdriver_manager para evitar errores de options.
    """
    global _driver
    if _driver is not None:
        return _driver

    if BROWSER.lower() == "edge":
        opts = EdgeOptions()
        opts.add_argument("--headless=new")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--window-size=1400,3200")
        service = EdgeService(EdgeChromiumDriverManager().install())
        _driver = webdriver.Edge(service=service, options=opts)
    else:
        # CHROME por defecto
        opts = ChromeOptions()
        opts.add_argument("--headless=new")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--window-size=1400,3200")
        service = ChromeService(ChromeDriverManager().install())
        _driver = webdriver.Chrome(service=service, options=opts)

    return _driver

def load_full_category(url: str) -> str:
    """
    Abre la subcategoría, espera la grilla .productRow y hace scroll
    hasta que deje de cargar. Devuelve HTML renderizado.
    """
    d = get_driver()
    d.get(url)

    # esperar a que Angular monte la grilla
    WebDriverWait(d, 25).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "section .productRow"))
    )

    last_h = 0
    last_n = 0
    while True:
        cards = d.find_elements(By.CSS_SELECTOR, "section .productRow > div")
        d.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1.0)
        h = d.execute_script("return document.body.scrollHeight")
        n = len(cards)
        if h == last_h and n == last_n:
            break
        last_h, last_n = h, n
    return d.page_source

# -------------- Extraction -------------------
def extract_products_from_rendered_html(html_src: str,
                                        categoria_slug: str,
                                        categoria_nombre: str,
                                        base_url: str) -> List[Product]:
    soup = BS(html_src, "lxml")
    out: List[Product] = []

    # Cada tarjeta suele estar como <div> dentro de .productRow
    cards = soup.select("section .productRow > div")
    for card in cards:
        # Nombre
        name_el = card.select_one("div.product-info div.text-ellipsis.truncate")
        if not name_el:
            name_el = card.select_one("div.product-info div")
        name = clean_text(name_el.get_text()) if name_el else ""

        # Precio (derecha dentro de product-info)
        price_el = card.select_one("div.product-info div.flex-row.justify-between div:last-child")
        price_raw = clean_text(price_el.get_text()) if price_el else ""
        if "$" not in price_raw:
            cand = card.find(text=re.compile(r"\$[\s]*\d"))
            price_raw = clean_text(cand) if cand else price_raw
        price_val, currency = parse_price(price_raw)

        # Enlace del producto (si existe)
        url_producto = ""
        a = card.find("a", href=True)
        if a:
            href = a["href"]
            url_producto = href if href.startswith("http") else urljoin(base_url, href)

        # Imagen
        img_url = None
        img = card.find("img")
        if img:
            src = img.get("data-src") or img.get("src")
            if src:
                img_url = src if src.startswith("http") else urljoin(base_url, src)

        # Filtra tarjetas vacías
        if name or price_val is not None:
            out.append(Product(
                categoria_slug=categoria_slug,
                categoria_nombre=categoria_nombre,
                producto_nombre=name,
                precio=price_val,
                moneda=currency,
                url_producto=url_producto,
                url_imagen=img_url
            ))

    return out

# -------------- Export -------------------
def export_results(products: List[Product], base_filename: str = "esperanza_panaderia"):
    rows = [asdict(p) for p in products]
    if not rows:
        logging.warning("No se extrajeron productos; no se crearán archivos.")
        return

    # CSV
    import csv
    csv_path = f"{base_filename}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logging.info("CSV creado: %s", csv_path)

    # Excel
    if HAS_PANDAS:
        df = pd.DataFrame(rows)
        xlsx_path = f"{base_filename}.xlsx"
        df.to_excel(xlsx_path, index=False)
        logging.info("Excel creado: %s", xlsx_path)

# -------------- Pipeline -------------------
def scrape_panaderia() -> List[Product]:
    logging.info("Subcategorías: %s", [c['name'] for c in CATS])

    # (opcional) filtrar las que devuelven 200
    valid = []
    for c in CATS:
        try:
            r = requests.get(c["url"], headers=HEADERS, timeout=15)
            if r.status_code == 200:
                valid.append(c)
            else:
                logging.warning("Descartada %s (HTTP %s)", c["url"], r.status_code)
        except Exception as e:
            logging.warning("Descartada por error (%s): %s", e, c["url"])
    cats = valid if valid else CATS

    all_products: List[Product] = []

    for c in cats:
        cat_name = c["name"]
        cat_slug = c["slug"]
        url = c["url"]

        logging.info("Scrapeando subcategoría: %s (%s)", cat_name, url)
        try:
            html_src = load_full_category(url)
        except Exception as e:
            logging.error("Error cargando %s: %s", url, e)
            continue

        prods = extract_products_from_rendered_html(html_src, cat_slug, cat_name, url)
        logging.info("Render: %d productos", len(prods))
        all_products.extend(prods)
        jitter()

    logging.info("Total de productos extraídos: %d", len(all_products))
    return all_products

# -------------- Main -------------------
def main():
    try:
        products = scrape_panaderia()
        export_results(products)
    finally:
        # Cerrar el driver si quedó abierto
        try:
            global _driver
            if _driver:
                _driver.quit()
        except Exception:
            pass

if __name__ == "__main__":
    main()
