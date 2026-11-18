"""
🚢 Titanic · Inferencia con el árbol de decisión del notebook
Ejecutar con:  streamlit run app.py
"""
import base64
import io
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

st.set_page_config(page_title="Titanic · Árbol de decisión", page_icon="🚢", layout="wide")

# ==================================================
# 1. Pipeline del notebook (mismos pasos y parámetros)
# ==================================================
AGE_BINS = [0, 12, 18, 35, 60, float("inf")]
AGE_LABELS = ["Child", "Teenager", "Young Adult", "Adult", "Senior"]
COMMON_TITLES = ["Mr", "Mrs", "Miss", "Master"]
SELECTED_FEATURES = ["Pclass", "Sex", "Age", "AgeGroup", "SibSp", "Parch",
                     "Fare", "Embarked", "FamilySize", "IsAlone", "Title"]
CATEGORICAL = ["AgeGroup", "Embarked", "Title"]


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["AgeGroup"] = pd.cut(df["Age"], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    if "Title" not in df.columns:
        df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")
        df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
        df.loc[~df["Title"].isin(COMMON_TITLES), "Title"] = "Rare"
    return df


def to_model_input(df: pd.DataFrame) -> pd.DataFrame:
    X = df[SELECTED_FEATURES].copy()
    X["Sex"] = X["Sex"].map({"male": 0, "female": 1})
    return X


@st.cache_data
def fare_reference():
    """Cuartiles de la tarifa por clase (para dar contexto al slider)."""
    train = pd.read_csv("data/train.csv")
    return train.groupby("Pclass")["Fare"].quantile([0.25, 0.5, 0.75]).unstack().to_dict("index")


GBP_1912_TO_TODAY = 130  # aprox.: 1 £ de 1912 ≈ 130 £ actuales (inflación acumulada)


@st.cache_resource
def train_model():
    train = pd.read_csv("data/train.csv")
    train_fe = feature_engineering(train)
    X = to_model_input(train_fe)
    y = train_fe["Survived"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="passthrough",
    )
    X_train_p = pd.DataFrame(preprocessor.fit_transform(X_train),
                             columns=preprocessor.get_feature_names_out(), index=X_train.index)
    X_val_p = pd.DataFrame(preprocessor.transform(X_val),
                           columns=preprocessor.get_feature_names_out(), index=X_val.index)

    model = DecisionTreeClassifier(criterion="gini", max_depth=4,
                                   min_samples_leaf=10, random_state=42)
    model.fit(X_train_p, y_train)
    acc = (model.predict(X_val_p) == y_val).mean()
    return model, preprocessor, train_fe, acc


model, preprocessor, passengers, val_acc = train_model()

# ==================================================
# 2. Estado de los inputs (valores por defecto + pasajero aleatorio)
# ==================================================
DEFAULTS = dict(name="", pclass=3, sex="male", age=28.0, age_unknown=False,
                sibsp=0, parch=0, fare=14.45, embarked="S", title="Mr", real=None)
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("result", None)


def load_random_passenger():
    row = passengers.sample(1).iloc[0]
    st.session_state.name = row["Name"]
    st.session_state.pclass = int(row["Pclass"])
    st.session_state.sex = row["Sex"]
    st.session_state.age_unknown = bool(pd.isna(row["Age"]))
    # El slider va de 0,5 en 0,5 (los bebés del dataset tienen edades como 0.42 o 0.83)
    st.session_state.age = 28.0 if pd.isna(row["Age"]) else max(0.5, round(float(row["Age"]) * 2) / 2)
    st.session_state.sibsp = int(row["SibSp"])
    st.session_state.parch = int(row["Parch"])
    st.session_state.fare = float(row["Fare"])
    st.session_state.embarked = "?" if pd.isna(row["Embarked"]) else row["Embarked"]
    st.session_state.title = row["Title"]
    st.session_state.real = int(row["Survived"])
    st.session_state.result = None


def set_fare(value):
    st.session_state.fare = float(value)
    manual_edit()


def manual_edit():
    # Si el usuario toca algo, ya no es el pasajero real
    st.session_state.real = None
    st.session_state.name = ""
    st.session_state.result = None


# ==================================================
# 3. Dibujos: monigote, calavera y salvavidas (SVG)
# ==================================================
def stick_figure_svg(sex, age, pclass, fate=None):
    """Monigote sencillo. Tamaño según edad, falda si es mujer, sombrero según clase."""
    if age is None:
        scale = 1.0
    elif age < 12:
        scale = 0.6
    elif age < 18:
        scale = 0.8
    else:
        scale = 1.0
    senior = age is not None and age >= 60
    color = {None: "#2c3e50", 1: "#1e8449", 0: "#7f8c8d"}[fate]

    # Coordenadas base (figura de 200 px de alto) escaladas desde los pies (y=190)
    def Y(y):
        return 190 - (190 - y) * scale

    def X(x):
        return 100 + (x - 100) * scale

    head_y, neck, hip, foot = Y(40), Y(62), Y(120), Y(185)
    parts = [f'<circle cx="100" cy="{head_y}" r="{20*scale}" fill="#fdebd0" stroke="{color}" stroke-width="4"/>']
    # Cara
    parts.append(f'<circle cx="{X(93)}" cy="{Y(37)}" r="{2.5*scale}" fill="{color}"/>')
    parts.append(f'<circle cx="{X(107)}" cy="{Y(37)}" r="{2.5*scale}" fill="{color}"/>')
    smile = f'M {X(92)} {Y(47)} Q 100 {Y(54)} {X(108)} {Y(47)}' if fate != 0 else \
            f'M {X(92)} {Y(51)} Q 100 {Y(45)} {X(108)} {Y(51)}'
    parts.append(f'<path d="{smile}" stroke="{color}" stroke-width="2.5" fill="none"/>')
    # Pelo
    if sex == "female":
        parts.append(f'<path d="M {X(80)} {Y(40)} Q {X(78)} {Y(10)} 100 {Y(18)} Q {X(122)} {Y(10)} {X(120)} {Y(40)} '
                     f'L {X(124)} {Y(70)} M {X(80)} {Y(40)} L {X(76)} {Y(70)}" stroke="#a04000" stroke-width="5" fill="none"/>')
    # Cuerpo
    if sex == "female":
        parts.append(f'<line x1="100" y1="{neck}" x2="100" y2="{Y(100)}" stroke="{color}" stroke-width="4"/>')
        parts.append(f'<polygon points="100,{Y(95)} {X(72)},{Y(140)} {X(128)},{Y(140)}" fill="#e8a0bf" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<line x1="{X(90)}" y1="{Y(140)}" x2="{X(88)}" y2="{foot}" stroke="{color}" stroke-width="4"/>')
        parts.append(f'<line x1="{X(110)}" y1="{Y(140)}" x2="{X(112)}" y2="{foot}" stroke="{color}" stroke-width="4"/>')
    else:
        parts.append(f'<line x1="100" y1="{neck}" x2="100" y2="{hip}" stroke="{color}" stroke-width="4"/>')
        parts.append(f'<line x1="100" y1="{hip}" x2="{X(80)}" y2="{foot}" stroke="{color}" stroke-width="4"/>')
        parts.append(f'<line x1="100" y1="{hip}" x2="{X(120)}" y2="{foot}" stroke="{color}" stroke-width="4"/>')
    # Brazos (arriba si sobrevive)
    arm_y = Y(55) if fate == 1 else Y(110)
    parts.append(f'<line x1="100" y1="{Y(78)}" x2="{X(70)}" y2="{arm_y}" stroke="{color}" stroke-width="4"/>')
    parts.append(f'<line x1="100" y1="{Y(78)}" x2="{X(130)}" y2="{arm_y}" stroke="{color}" stroke-width="4"/>')
    # Bastón si es mayor
    if senior:
        parts.append(f'<path d="M {X(130)} {arm_y} L {X(134)} {foot} M {X(130)} {arm_y} q 8 -8 12 2" '
                     f'stroke="#6e2c00" stroke-width="4" fill="none"/>')
    # Sombrero según clase
    top = head_y - 20 * scale
    if pclass == 1:  # chistera
        parts.append(f'<rect x="{X(84)}" y="{top-22*scale}" width="{32*scale}" height="{24*scale}" fill="#111"/>')
        parts.append(f'<rect x="{X(76)}" y="{top}" width="{48*scale}" height="{5*scale}" fill="#111"/>')
    elif pclass == 2:  # bombín
        parts.append(f'<path d="M {X(82)} {top+3} Q 100 {top-22*scale} {X(118)} {top+3} Z" fill="#5d4037"/>')
        parts.append(f'<rect x="{X(76)}" y="{top+1}" width="{48*scale}" height="{4*scale}" fill="#5d4037"/>')
    else:  # gorra
        parts.append(f'<path d="M {X(82)} {top+4} Q 100 {top-12*scale} {X(118)} {top+4} Z" fill="#7f8c8d"/>')
        parts.append(f'<rect x="{X(110)}" y="{top+1}" width="{20*scale}" height="{4*scale}" fill="#7f8c8d"/>')
    return f'<svg viewBox="0 0 200 200" width="180" height="180">{"".join(parts)}</svg>'


LIFEBUOY_SVG = """
<svg viewBox="0 0 100 100" width="120" height="120">
  <circle cx="50" cy="50" r="40" fill="none" stroke="#e74c3c" stroke-width="18"/>
  <circle cx="50" cy="50" r="40" fill="none" stroke="white" stroke-width="18"
          stroke-dasharray="31.4 31.4" stroke-dashoffset="15.7"/>
  <circle cx="50" cy="50" r="49" fill="none" stroke="#922b21" stroke-width="1.5"/>
  <circle cx="50" cy="50" r="31" fill="none" stroke="#922b21" stroke-width="1.5"/>
</svg>"""

SKULL_SVG = """
<svg viewBox="0 0 100 100" width="120" height="120">
  <path d="M50 8 C24 8 12 28 14 48 C15 60 22 64 26 68 L26 82 L74 82 L74 68 C78 64 85 60 86 48 C88 28 76 8 50 8 Z"
        fill="#f4f6f7" stroke="#566573" stroke-width="3"/>
  <ellipse cx="35" cy="46" rx="10" ry="11" fill="#2c3e50"/>
  <ellipse cx="65" cy="46" rx="10" ry="11" fill="#2c3e50"/>
  <path d="M50 56 L44 68 L56 68 Z" fill="#2c3e50"/>
  <line x1="40" y1="74" x2="40" y2="82" stroke="#566573" stroke-width="2.5"/>
  <line x1="50" y1="74" x2="50" y2="82" stroke="#566573" stroke-width="2.5"/>
  <line x1="60" y1="74" x2="60" y2="82" stroke="#566573" stroke-width="2.5"/>
  <circle cx="38" cy="43" r="2.5" fill="white"/><circle cx="68" cy="43" r="2.5" fill="white"/>
</svg>"""


# ==================================================
# 4. Árbol (con recorrido opcional), igual que en el notebook
# ==================================================
@st.cache_data(show_spinner=False)
def draw_tree(path_key):
    visited_nodes = list(path_key) if path_key else []
    fig, ax = plt.subplots(figsize=(30, 12))
    # Nombres más legibles (sin los prefijos cat__ / remainder__ del ColumnTransformer)
    clean_names = [n.split("__", 1)[1] for n in preprocessor.get_feature_names_out()]
    clean_names = ["Sex (0=hombre, 1=mujer)" if n == "Sex" else n for n in clean_names]
    artists = plot_tree(
        model,
        feature_names=clean_names,
        class_names=["No sobrevivió", "Sobrevivió"],
        filled=True, rounded=True, node_ids=True, impurity=False, fontsize=12, ax=ax,
    )
    node_artists = {}
    for artist in artists:
        m = re.search(r"node #(\d+)", artist.get_text())
        if m:
            node_artists[int(m.group(1))] = artist

    if visited_nodes:
        # Atenuar los nodos que no se visitan
        for nid, art in node_artists.items():
            if nid not in visited_nodes:
                art.set_alpha(0.35)
                art.get_bbox_patch().set_alpha(0.25)
        for nid in visited_nodes:
            bbox = node_artists[nid].get_bbox_patch()
            bbox.set_edgecolor("red")
            bbox.set_linewidth(4)
        for parent, child in zip(visited_nodes[:-1], visited_nodes[1:]):
            ax.annotate("", xy=node_artists[child].get_position(),
                        xytext=node_artists[parent].get_position(),
                        arrowprops=dict(arrowstyle="->", color="red", linewidth=3,
                                        shrinkA=28, shrinkB=28), zorder=10)
    fig.tight_layout()
    # SVG = gráfico vectorial: se ve nítido a cualquier tamaño/zoom
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ==================================================
# 5. Interfaz
# ==================================================
st.markdown("""
<style>
.big-result {border-radius: 18px; padding: 24px; text-align: center; color: white;}
.big-result h1 {color: white; margin: 6px 0; font-size: 2.6rem;}
.survived {background: linear-gradient(135deg,#1e8449,#58d68d);}
.died {background: linear-gradient(135deg,#34495e,#85929e);}
.tree-box {overflow-x: auto; border: 1px solid #d5dbdb; border-radius: 12px; background: white; padding: 8px;}
.tree-box img {min-width: 1800px; width: 100%; height: auto;}
.fare-ref {font-size: 0.85rem; color: #566573;}
.card {background:#f4f6f7; border-radius:14px; padding:12px; text-align:center;}
</style>
""", unsafe_allow_html=True)

head_l, head_r = st.columns([3, 2], vertical_alignment="center")
with head_l:
    st.title("🚢 ¿Habrías sobrevivido al Titanic?")
    st.markdown(
        "El **10 de abril de 1912** el *RMS Titanic*, el barco más grande y lujoso de su época, "
        "zarpó de **Southampton** (Inglaterra) rumbo a Nueva York, haciendo escala en **Cherbourg** "
        "(Francia) y **Queenstown** (Irlanda). Lo llamaban *\"prácticamente insumergible\"*.\n\n"
        "Cuatro días después, en la noche del **14 al 15 de abril**, chocó contra un iceberg en el "
        "Atlántico Norte y se hundió en menos de tres horas. De las **más de 2.200 personas** a bordo, "
        "murieron unas **1.500**."
    )
with head_r:
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg",
             caption="RMS Titanic, abril de 1912", width="stretch")

intro1, intro2, intro3 = st.columns(3)
with intro1:
    st.markdown(
        "#### 🛶 Botes para la mitad\n"
        "El barco llevaba solo **20 botes salvavidas**, con sitio para unas **1.178 personas**: "
        "apenas la mitad de los que iban a bordo. Cumplía la normativa de la época, que estaba "
        "pensada para barcos mucho más pequeños."
    )
with intro2:
    st.markdown(
        "#### 👩‍👧 «Mujeres y niños primero»\n"
        "Durante la evacuación se siguió (sobre todo) este protocolo. Además, los camarotes de "
        "**1ª clase** estaban más cerca de la cubierta de botes, mientras que muchos pasajeros de "
        "**3ª clase** viajaban en las cubiertas inferiores, lejos de la salida."
    )
with intro3:
    st.markdown(
        "#### 🤖 ¿Qué hace el modelo?\n"
        "Con los datos reales de **891 pasajeros** hemos entrenado un **árbol de decisión** que "
        "aprende qué características (sexo, edad, clase, familia, tarifa...) se asocian a sobrevivir. "
        f"Acierta el **{val_acc:.1%}** de los pasajeros de validación."
    )
st.info(
    "**Cómo usarlo:** rellena los datos de un pasajero (o elige uno al azar del dataset), pulsa "
    "**Ejecutar** y verás la predicción y el camino que sigue el pasajero por el árbol, pregunta a "
    "pregunta, hasta llegar a una hoja.", icon="🧭"
)

st.divider()

form_col, avatar_col = st.columns([3, 1])

with form_col:
    st.subheader("🎫 Datos del pasajero")
    st.button("🎲 Pasajero aleatorio del dataset", on_click=load_random_passenger)
    if st.session_state.name:
        st.caption(f"Pasajero real: **{st.session_state.name}**")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.selectbox("Clase", [1, 2, 3], key="pclass", on_change=manual_edit,
                     format_func=lambda c: {1: "1ª clase", 2: "2ª clase", 3: "3ª clase"}[c])
    with c2:
        st.selectbox("Sexo", ["male", "female"], key="sex", on_change=manual_edit,
                     format_func=lambda s: "Hombre" if s == "male" else "Mujer")
    with c3:
        st.selectbox("Título", ["Mr", "Mrs", "Miss", "Master", "Rare"], key="title", on_change=manual_edit,
                     format_func=lambda t: {"Mr": "Mr (señor)", "Mrs": "Mrs (señora)", "Miss": "Miss (señorita)",
                                            "Master": "Master (niño)", "Rare": "Rare (Dr, Rev, Col...)"}[t])
    with c4:
        st.selectbox("Puerto de embarque", ["S", "C", "Q", "?"], key="embarked", on_change=manual_edit,
                     format_func=lambda e: {"S": "🇬🇧 Southampton", "C": "🇫🇷 Cherbourg",
                                            "Q": "🇮🇪 Queenstown", "?": "Desconocido"}[e])

    c5, c6 = st.columns(2)
    with c5:
        st.number_input("👫 Hermanos / cónyuges a bordo (SibSp)", 0, 8, step=1, key="sibsp", on_change=manual_edit)
    with c6:
        st.number_input("👨‍👩‍👧 Padres / hijos a bordo (Parch)", 0, 6, step=1, key="parch", on_change=manual_edit)

    # ---- Edad ----
    age_l, age_r = st.columns([4, 1], vertical_alignment="bottom")
    with age_l:
        st.slider("🎂 Edad (años)", 0.5, 80.0, step=0.5, key="age", on_change=manual_edit,
                  disabled=st.session_state.age_unknown,
                  help="Rango de edades del dataset: de bebés de pocos meses a 80 años.")
    with age_r:
        st.checkbox("Edad desconocida", key="age_unknown", on_change=manual_edit,
                    help="El 20 % de los pasajeros no tiene edad registrada. El árbol sabe trabajar con ese hueco.")

    # ---- Tarifa ----
    ref = fare_reference()[st.session_state.pclass]
    st.slider("💷 Tarifa del billete (libras de 1912)", 0.0, 515.0, step=0.0001, format="£%.2f",
              key="fare", on_change=manual_edit,
              help="Precio total del billete. Si se compró para un grupo o familia, es el precio de todo el grupo.")
    fare = st.session_state.fare
    if fare < ref[0.25]:
        level = "🟢 barato para su clase"
    elif fare <= ref[0.75]:
        level = "🟡 precio normal para su clase"
    else:
        level = "🔴 caro para su clase"
    st.markdown(
        f'<div class="fare-ref">≈ <b>£{fare * GBP_1912_TO_TODAY:,.0f}</b> de hoy · {level} · '
        f'Precio típico en {st.session_state.pclass}ª clase: <b>£{ref[0.5]:.2f}</b> '
        f'(la mitad pagó entre £{ref[0.25]:.2f} y £{ref[0.75]:.2f})</div>',
        unsafe_allow_html=True,
    )
    b1, b2, b3 = st.columns(3)
    b1.button("Billete barato", on_click=set_fare, args=(ref[0.25],), width="stretch")
    b2.button("Billete típico", on_click=set_fare, args=(ref[0.5],), width="stretch")
    b3.button("Billete caro", on_click=set_fare, args=(ref[0.75],), width="stretch")
    st.write("")

    run = st.button("▶️ Ejecutar predicción", type="primary", width="stretch")

# Construir la instancia con el mismo pipeline que el notebook
age_value = None if st.session_state.age_unknown else st.session_state.age
raw = pd.DataFrame([{
    "Pclass": st.session_state.pclass,
    "Sex": st.session_state.sex,
    "Age": np.nan if age_value is None else age_value,
    "SibSp": st.session_state.sibsp,
    "Parch": st.session_state.parch,
    "Fare": st.session_state.fare,
    "Embarked": np.nan if st.session_state.embarked == "?" else st.session_state.embarked,
    "Title": st.session_state.title,
}])
X_raw = to_model_input(feature_engineering(raw))
X_raw[CATEGORICAL] = X_raw[CATEGORICAL].astype(object)  # evita columnas float si el valor es NaN
sample = pd.DataFrame(preprocessor.transform(X_raw), columns=preprocessor.get_feature_names_out())

if run:
    pred = int(model.predict(sample)[0])
    proba = model.predict_proba(sample)[0]
    path = tuple(int(n) for n in model.decision_path(sample).indices)
    st.session_state.result = dict(pred=pred, proba=proba, path=path)

result = st.session_state.result

with avatar_col:
    st.subheader("🧍 Tu personaje")
    fate = result["pred"] if result else None
    st.markdown(f'<div class="card">{stick_figure_svg(st.session_state.sex, age_value, st.session_state.pclass, fate)}'
                f'<br><small>Familia a bordo: {st.session_state.sibsp + st.session_state.parch} · '
                f'Grupo de edad: {X_raw["AgeGroup"].iloc[0] if pd.notna(X_raw["AgeGroup"].iloc[0]) else "?"}</small></div>',
                unsafe_allow_html=True)

# Resultado grande
if result:
    st.divider()
    survived = result["pred"] == 1
    icon = LIFEBUOY_SVG if survived else SKULL_SVG
    text = "¡SOBREVIVE!" if survived else "NO SOBREVIVE"
    msg = ("Has conseguido un sitio en un bote salvavidas 🛟" if survived
           else "Las frías aguas del Atlántico no perdonan... 🧊")
    prob = result["proba"][1]
    real = st.session_state.real
    real_html = ""
    if real is not None:
        ok = "✅ acierto" if real == result["pred"] else "❌ fallo"
        real_html = (f"<p>Valor real en el dataset: <b>{'Sobrevivió' if real == 1 else 'No sobrevivió'}</b> "
                     f"({ok} del modelo)</p>")
    st.markdown(
        f'<div class="big-result {"survived" if survived else "died"}">{icon}'
        f'<h1>{text}</h1><p style="font-size:1.2rem">{msg}</p>'
        f'<p>Probabilidad de supervivencia en la hoja: <b>{prob:.0%}</b></p>{real_html}</div>',
        unsafe_allow_html=True,
    )

st.divider()
st.subheader("🌳 Árbol de decisión" + (" · recorrido del pasajero" if result else ""))
if not result:
    st.caption("Pulsa **Ejecutar predicción** para ver el camino que sigue el pasajero (en rojo).")
tree_svg = draw_tree(result["path"] if result else None)
st.markdown(
    f'<div class="tree-box"><img src="data:image/svg+xml;base64,{tree_svg}"></div>',
    unsafe_allow_html=True,
)
st.caption("💡 Desliza horizontalmente para recorrer el árbol. Cada nodo muestra: la pregunta, "
           "nº de pasajeros (*samples*), cuántos murieron/sobrevivieron (*value*) y la clase mayoritaria.")

with st.expander("🔍 Ver la instancia tal y como la recibe el modelo"):
    st.dataframe(X_raw, hide_index=True)
    st.dataframe(sample, hide_index=True)
