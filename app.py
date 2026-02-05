import io
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from dateutil import parser as dtparser

# PRO parsers
import gpxpy
from fitparse import FitFile
from lxml import etree

# ============================================================
# Planificador PRO
# - Orientado a ZIP completos (Strava/Garmin/etc.)
# - Si hay FIT/TCX dentro del ZIP, usa datos punto a punto (HR/ritmo/velocidad)
# ============================================================

# -----------------------------
# Tipos de actividad (para Strava CSV cuando exista)
# -----------------------------
RUN_TYPES = {
    "run", "running", "carrera", "trail run", "treadmill", "virtual run",
}
BIKE_TYPES = {
    "ride", "cycling", "bicicleta", "bike", "road", "road cycling", "mountain bike",
    "mtb", "gravel ride", "virtual ride", "e-bike ride", "ebike ride",
}

GOALS = {
    "correr": [
        ("maraton", "Maratón (en ~12 meses)"),
        ("resistencia", "Mejorar resistencia y bajar pulsaciones"),
        ("media", "Media maratón"),
        ("10k", "10K"),
        ("base", "Solo base aeróbica"),
    ],
    "bicicleta": [
        ("gran_fondo", "Gran fondo (larga distancia, en ~12 meses)"),
        ("resistencia", "Mejorar resistencia y bajar pulsaciones"),
        ("puerto", "Subidas / puertos (mejorar umbral en subida)"),
        ("criterium", "Potencia y cambios de ritmo (tipo criterium)"),
        ("base", "Solo base aeróbica"),
    ],
}

# -----------------------------
# Librerías (movilidad / core / fuerza)
# -----------------------------
MOBILITY_LIBRARY = {
    "movilidad_20": [
        "Caderas: 90/90 (2x60s por lado)",
        "Tobillos: rodilla a pared (2x12 por lado)",
        "Isquios: bisagra con banda (2x10)",
        "Columna torácica: rotaciones en cuadrupedia (2x8 por lado)",
        "Glúteo medio: monster walks con minibanda (2x12 pasos por lado)",
    ],
    "movilidad_10": [
        "Tobillos: rodilla a pared (2x10 por lado)",
        "Caderas: 90/90 (1x60s por lado)",
        "Torácica: rotaciones (1x8 por lado)",
    ],
}

CORE_LIBRARY = {
    "core_20": [
        "Plancha frontal 3x40s (20s descanso)",
        "Plancha lateral 3x30s por lado",
        "Dead bug 3x10 por lado",
        "Puente glúteo 3x12",
        "Bird-dog 3x10 por lado",
    ],
    "core_10": [
        "Plancha frontal 2x40s",
        "Dead bug 2x10 por lado",
        "Puente glúteo 2x12",
    ],
}

STRENGTH_LIBRARY = {
    "fuerza_35": [
        "Sentadilla goblet 4x8 (RPE 7)",
        "Peso muerto rumano 4x8 (RPE 7)",
        "Zancadas 3x10 por lado (RPE 7)",
        "Elevación de gemelos 4x12",
        "Remo con mancuerna 3x10 por lado",
    ],
    "fuerza_25": [
        "Sentadilla goblet 3x8 (RPE 7)",
        "Peso muerto rumano 3x8 (RPE 7)",
        "Gemelos 3x12",
        "Remo 2x10 por lado",
    ],
}

# ============================================================
# Utilidades
# ============================================================

def next_monday(d: date) -> date:
    return d + timedelta(days=(7 - d.weekday()) % 7)

def safe_float(x) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None

def parse_time_to_minutes(val) -> Optional[float]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, (int, float)):
        if val > 1000:
            return float(val) / 60.0
        return float(val)
    s = str(val).strip()
    if not s:
        return None
    if ":" in s:
        parts = s.split(":")
        try:
            parts = [int(p) for p in parts]
        except Exception:
            return None
        if len(parts) == 3:
            h, m, sec = parts
            return (h * 3600 + m * 60 + sec) / 60.0
        if len(parts) == 2:
            m, sec = parts
            return (m * 60 + sec) / 60.0
    return safe_float(s)

def estimate_hrmax(age: int) -> int:
    # Estimación general; si conoces HRmáx real, mejor usarlo.
    return int(round(208 - 0.7 * age))

def hr_zones(hrmax: int) -> Dict[str, Tuple[int, int]]:
    return {
        "Z1": (int(0.60 * hrmax), int(0.70 * hrmax)),
        "Z2": (int(0.70 * hrmax), int(0.80 * hrmax)),
        "Z3": (int(0.80 * hrmax), int(0.87 * hrmax)),
        "Z4": (int(0.87 * hrmax), int(0.93 * hrmax)),
        "Z5": (int(0.93 * hrmax), hrmax),
    }

def _to_datetime_safe(s) -> Optional[datetime]:
    try:
        return dtparser.parse(str(s))
    except Exception:
        return None

def _week_start_monday(ts: pd.Timestamp) -> pd.Timestamp:
    d = ts.normalize()
    return d - pd.Timedelta(days=d.weekday())

def _sec_diff_safe(a: pd.Timestamp, b: pd.Timestamp) -> float:
    try:
        return float((b - a).total_seconds())
    except Exception:
        return 0.0

# ============================================================
# Parser PRO: extrae series temporales cuando existen (FIT/TCX)
# ============================================================

def _records_to_timeseries(records: List[dict]) -> pd.DataFrame:
    """
    Records list -> dataframe con columnas:
      - timestamp (datetime)
      - hr (bpm)
      - speed_mps (m/s)
      - distance_m (m)
    """
    if not records:
        return pd.DataFrame(columns=["timestamp", "hr", "speed_mps", "distance_m"])
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df.get("timestamp"), errors="coerce")
    df = df[df["timestamp"].notna()].sort_values("timestamp")
    for c in ["hr", "speed_mps", "distance_m"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _summary_from_timeseries(ts: pd.DataFrame) -> dict:
    if ts.empty:
        return {"moving_s": None, "distance_m": None, "avg_hr": None, "avg_speed_mps": None}

    # Duración por diffs (ignorando gaps enormes)
    dt = ts["timestamp"].diff().dt.total_seconds()
    dt = dt.where((dt.notna()) & (dt >= 0) & (dt <= 60), 0)  # cap gaps
    moving_s = float(dt.sum())

    distance_m = None
    if ts["distance_m"].notna().any():
        # distancia total como último - primero
        d0 = float(ts["distance_m"].dropna().iloc[0])
        d1 = float(ts["distance_m"].dropna().iloc[-1])
        distance_m = max(0.0, d1 - d0)
    elif ts["speed_mps"].notna().any():
        # aproximación integrando velocidad
        distance_m = float((ts["speed_mps"].fillna(0) * dt.fillna(0)).sum())

    avg_hr = float(ts["hr"].dropna().mean()) if ts["hr"].notna().any() else None
    avg_speed = float(ts["speed_mps"].dropna().mean()) if ts["speed_mps"].notna().any() else None

    return {"moving_s": moving_s, "distance_m": distance_m, "avg_hr": avg_hr, "avg_speed_mps": avg_speed}

def _zone_seconds(ts: pd.DataFrame, hrmax: int) -> Dict[str, float]:
    z = hr_zones(hrmax)
    out = {k: 0.0 for k in z.keys()}

    if ts.empty or ts["hr"].notna().sum() < 10:
        return out

    df = ts.copy()
    df["dt"] = df["timestamp"].diff().dt.total_seconds()
    df["dt"] = df["dt"].where((df["dt"].notna()) & (df["dt"] >= 0) & (df["dt"] <= 60), 0)
    hr = df["hr"]

    for zone, (lo, hi) in z.items():
        mask = hr.ge(lo) & hr.lt(hi)
        out[zone] = float(df.loc[mask, "dt"].sum())

    return out

def _hr_drift(ts: pd.DataFrame) -> Optional[float]:
    """
    Deriva cardiaca simple: (HR 2ª mitad - HR 1ª mitad) / HR 1ª mitad
    solo si hay suficientes puntos.
    """
    if ts.empty or ts["hr"].notna().sum() < 60:
        return None
    df = ts.dropna(subset=["hr"]).copy()
    if df.shape[0] < 60:
        return None
    mid = df.shape[0] // 2
    hr1 = float(df.iloc[:mid]["hr"].mean())
    hr2 = float(df.iloc[mid:]["hr"].mean())
    if hr1 <= 0:
        return None
    return (hr2 - hr1) / hr1

def parse_fit_pro(file_bytes: bytes) -> Tuple[dict, pd.DataFrame]:
    """
    Devuelve:
      - summary dict con start_dt, sport, distance_km, moving_min, avg_hr, avg_speed_mps, source
      - timeseries df (timestamp/hr/speed_mps/distance_m)
    """
    fitfile = FitFile(BytesIO(file_bytes))

    # 1) sport/start via session si existe
    sport = "other"
    start_dt = None
    sessions = list(fitfile.get_messages("session"))
    if sessions:
        s = sessions[0]
        fields = {f.name: f.value for f in s}
        start_dt = fields.get("start_time")
        sport_raw = str(fields.get("sport") or "").lower()
        if "run" in sport_raw:
            sport = "run"
        elif "cycling" in sport_raw or "bike" in sport_raw:
            sport = "ride"

    # 2) records para series
    recs = []
    for r in fitfile.get_messages("record"):
        fields = {f.name: f.value for f in r}
        ts = fields.get("timestamp")
        if ts is None:
            continue
        recs.append(
            {
                "timestamp": ts,
                "hr": fields.get("heart_rate"),
                "speed_mps": fields.get("speed"),
                "distance_m": fields.get("distance"),
            }
        )
    ts_df = _records_to_timeseries(recs)

    summ2 = _summary_from_timeseries(ts_df)
    distance_km = (summ2["distance_m"] / 1000.0) if summ2["distance_m"] is not None else None
    moving_min = (summ2["moving_s"] / 60.0) if summ2["moving_s"] is not None else None

    # start_dt fallback: primer timestamp
    if start_dt is None and not ts_df.empty:
        start_dt = ts_df["timestamp"].iloc[0]

    summary = {
        "start_dt": start_dt,
        "sport": sport,
        "distance_km": distance_km,
        "moving_min": moving_min,
        "avg_hr": summ2["avg_hr"],
        "avg_speed_mps": summ2["avg_speed_mps"],
        "source": "fit_pro",
    }
    return summary, ts_df

def parse_tcx_pro(file_bytes: bytes) -> Tuple[dict, pd.DataFrame]:
    """
    TCX: extrae Trackpoints con Time/HR/Speed/DistanceMeters cuando existen.
    """
    root = etree.parse(BytesIO(file_bytes))
    activities = root.xpath("//*[local-name()='Activity']")
    if not activities:
        return (
            {"start_dt": None, "sport": "other", "distance_km": None, "moving_min": None, "avg_hr": None, "avg_speed_mps": None, "source": "tcx_pro"},
            pd.DataFrame(columns=["timestamp", "hr", "speed_mps", "distance_m"]),
        )
    act = activities[0]
    sport_attr = (act.get("Sport") or "").lower()
    sport = "run" if "run" in sport_attr else "ride" if ("bike" in sport_attr or "cycle" in sport_attr) else "other"

    # Trackpoints
    tps = act.xpath(".//*[local-name()='Trackpoint']")
    recs = []
    for tp in tps:
        t = tp.xpath("./*[local-name()='Time']/text()")
        if not t:
            continue
        ts = _to_datetime_safe(t[0])

        hr = tp.xpath(".//*[local-name()='HeartRateBpm']//*[local-name()='Value']/text()")
        hr = float(hr[0]) if hr else None

        dist = tp.xpath("./*[local-name()='DistanceMeters']/text()")
        dist = float(dist[0]) if dist else None

        sp = tp.xpath(".//*[local-name()='Speed']/text()")
        sp = float(sp[0]) if sp else None

        recs.append({"timestamp": ts, "hr": hr, "speed_mps": sp, "distance_m": dist})

    ts_df = _records_to_timeseries(recs)
    summ2 = _summary_from_timeseries(ts_df)

    start_dt = None
    laps = act.xpath(".//*[local-name()='Lap']")
    if laps:
        st_attr = laps[0].get("StartTime")
        if st_attr:
            start_dt = _to_datetime_safe(st_attr)
    if start_dt is None and not ts_df.empty:
        start_dt = ts_df["timestamp"].iloc[0]

    summary = {
        "start_dt": start_dt,
        "sport": sport,
        "distance_km": (summ2["distance_m"] / 1000.0) if summ2["distance_m"] is not None else None,
        "moving_min": (summ2["moving_s"] / 60.0) if summ2["moving_s"] is not None else None,
        "avg_hr": summ2["avg_hr"],
        "avg_speed_mps": summ2["avg_speed_mps"],
        "source": "tcx_pro",
    }
    return summary, ts_df

def parse_gpx_basic(file_bytes: bytes) -> dict:
    gpx = gpxpy.parse(BytesIO(file_bytes))
    # GPX suele no incluir HR/Speed de forma estándar => resumen simple
    total_dist_m = 0.0
    start_dt = None
    end_dt = None
    for track in gpx.tracks:
        for segment in track.segments:
            if segment.points:
                if start_dt is None:
                    start_dt = segment.points[0].time
                end_dt = segment.points[-1].time
            try:
                total_dist_m += float(segment.length_3d())
            except Exception:
                try:
                    total_dist_m += float(segment.length_2d())
                except Exception:
                    pass
    moving_min = None
    if start_dt and end_dt:
        moving_min = (end_dt - start_dt).total_seconds() / 60.0
    return {
        "start_dt": start_dt,
        "sport": "other",
        "distance_km": total_dist_m / 1000.0 if total_dist_m else None,
        "moving_min": moving_min,
        "avg_hr": None,
        "avg_speed_mps": None,
        "source": "gpx_basic",
    }

def parse_strava_csv_basic(file_bytes: bytes) -> pd.DataFrame:
    # Compatibilidad: si en el ZIP viene activities.csv lo leemos para deporte y resumen.
    df_raw = pd.read_csv(BytesIO(file_bytes))
    cols = {c.strip().lower(): c for c in df_raw.columns}
    # nombres típicos
    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None
    c_type = pick("activity type", "type")
    c_date = pick("activity date", "date", "start date")
    c_dist = pick("distance")
    c_move = pick("moving time", "moving_time", "duration", "elapsed time")
    c_hr = pick("average heart rate", "avg heart rate", "avg_hr")

    if not (c_type and c_date and c_dist):
        return pd.DataFrame(columns=["start_dt","sport","distance_km","moving_min","avg_hr","avg_speed_mps","source"])

    df = df_raw.copy()
    df["start_dt"] = pd.to_datetime(df[c_date], errors="coerce")
    df = df[df["start_dt"].notna()]

    def sport_from_type(v):
        v = str(v).strip().lower()
        if v in RUN_TYPES:
            return "run"
        if v in BIKE_TYPES:
            return "ride"
        return "other"

    df["sport"] = df[c_type].apply(sport_from_type)
    dist = pd.to_numeric(df[c_dist], errors="coerce")
    df["distance_km"] = np.where(dist > 200, dist / 1000.0, dist)
    df["moving_min"] = df[c_move].apply(parse_time_to_minutes) if c_move else np.nan
    df["avg_hr"] = pd.to_numeric(df[c_hr], errors="coerce") if c_hr else np.nan
    df["avg_speed_mps"] = np.nan
    df["source"] = "strava_csv_basic"
    return df[["start_dt","sport","distance_km","moving_min","avg_hr","avg_speed_mps","source"]]

def parse_zip_pro(file_bytes: bytes) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Lee ZIP y devuelve:
      - activities_df (resumen por actividad)
      - timeseries_by_key dict: key -> timeseries df (para FIT/TCX)
    """
    z = zipfile.ZipFile(BytesIO(file_bytes))
    activities: List[dict] = []
    ts_map: Dict[str, pd.DataFrame] = {}

    # Si hay activities.csv, lo guardamos para fallback de deporte/fecha
    csv_frames = []

    for name in z.namelist():
        lname = name.lower()
        try:
            data = z.read(name)
        except Exception:
            continue

        if lname.endswith("activities.csv") or lname.endswith("activity.csv"):
            try:
                csv_frames.append(parse_strava_csv_basic(data))
            except Exception:
                pass

        if lname.endswith(".fit"):
            try:
                summ, ts = parse_fit_pro(data)
                key = f"{name}"
                activities.append({**summ, "key": key, "file": name})
                if not ts.empty:
                    ts_map[key] = ts
            except Exception:
                continue
        elif lname.endswith(".tcx"):
            try:
                summ, ts = parse_tcx_pro(data)
                key = f"{name}"
                activities.append({**summ, "key": key, "file": name})
                if not ts.empty:
                    ts_map[key] = ts
            except Exception:
                continue
        elif lname.endswith(".gpx"):
            # GPX básico
            try:
                summ = parse_gpx_basic(data)
                key = f"{name}"
                activities.append({**summ, "key": key, "file": name})
            except Exception:
                continue

    act_df = pd.DataFrame(activities)
    if act_df.empty and csv_frames:
        # Solo CSV
        act_df = pd.concat(csv_frames, ignore_index=True)
        act_df["key"] = act_df.index.astype(str)
        act_df["file"] = "activities.csv"
    elif not act_df.empty and csv_frames:
        # Si tenemos FIT/TCX/GPX y también CSV, podemos intentar rellenar sport faltante (other)
        csv_df = pd.concat(csv_frames, ignore_index=True)
        # emparejar por fecha cercana y distancia aproximada (heurística)
        act_df["start_dt"] = pd.to_datetime(act_df["start_dt"], errors="coerce")
        csv_df["start_dt"] = pd.to_datetime(csv_df["start_dt"], errors="coerce")
        for idx, row in act_df.iterrows():
            if str(row.get("sport")) != "other":
                continue
            if pd.isna(row.get("start_dt")) or pd.isna(row.get("distance_km")):
                continue
            window = csv_df[
                (csv_df["start_dt"].between(row["start_dt"] - pd.Timedelta(hours=6), row["start_dt"] + pd.Timedelta(hours=6)))
            ].copy()
            if window.empty:
                continue
            window["dist_diff"] = (window["distance_km"] - float(row["distance_km"])).abs()
            best = window.sort_values("dist_diff").head(1)
            if not best.empty and float(best["dist_diff"].iloc[0]) <= 0.3:
                act_df.at[idx, "sport"] = best["sport"].iloc[0]

    if act_df.empty:
        act_df = pd.DataFrame(columns=["start_dt","sport","distance_km","moving_min","avg_hr","avg_speed_mps","source","key","file"])

    # Normalizar
    act_df["start_dt"] = pd.to_datetime(act_df.get("start_dt"), errors="coerce")
    for c in ["distance_km","moving_min","avg_hr","avg_speed_mps"]:
        if c not in act_df.columns:
            act_df[c] = np.nan
        act_df[c] = pd.to_numeric(act_df[c], errors="coerce")

    return act_df, ts_map

# ============================================================
# Métricas PRO semanales
# ============================================================

def weekly_zone_distribution(target_df: pd.DataFrame, ts_map: Dict[str, pd.DataFrame], hrmax: int) -> pd.DataFrame:
    """
    Devuelve un df por semana con segundos en Z1..Z5 y % sobre total con HR.
    """
    if target_df.empty:
        return pd.DataFrame(columns=["week_start","Z1_s","Z2_s","Z3_s","Z4_s","Z5_s","hr_seconds"])

    rows = []
    for _, r in target_df.dropna(subset=["start_dt"]).iterrows():
        key = r.get("key")
        ts = ts_map.get(key)
        if ts is None or ts.empty:
            continue
        zs = _zone_seconds(ts, hrmax)
        hr_seconds = float(sum(zs.values()))
        ws = _week_start_monday(pd.Timestamp(r["start_dt"]))
        rows.append({"week_start": ws, **{f"{k}_s": v for k, v in zs.items()}, "hr_seconds": hr_seconds})

    if not rows:
        return pd.DataFrame(columns=["week_start","Z1_s","Z2_s","Z3_s","Z4_s","Z5_s","hr_seconds"])

    df = pd.DataFrame(rows)
    g = df.groupby("week_start").sum(numeric_only=True).reset_index().sort_values("week_start")
    for z in ["Z1","Z2","Z3","Z4","Z5"]:
        g[f"{z}_pct"] = np.where(g["hr_seconds"] > 0, g[f"{z}_s"] / g["hr_seconds"], np.nan)
    return g

def weekly_baseline_simple(target_df: pd.DataFrame) -> Dict[str, float]:
    """
    Similar a tu baseline anterior, pero sin renombrados raros.
    target_df: start_dt + distance_km
    """
    if target_df.empty:
        return {"w_km": 0.0, "long_km": 0.0, "sessions_w": 0.0}

    df = target_df.copy()
    df["date_only"] = pd.to_datetime(df["start_dt"], errors="coerce").dt.date
    df = df[df["date_only"].notna()]

    today = date.today()
    start = today - timedelta(days=42)
    df = df[df["date_only"] >= start]
    if df.empty:
        return {"w_km": 0.0, "long_km": 0.0, "sessions_w": 0.0}

    df["week"] = pd.to_datetime(df["date_only"]).dt.to_period("W-MON")
    w = df.groupby("week")["distance_km"].sum()
    sw = df.groupby("week")["distance_km"].count()
    w_km = float(w.mean()) if len(w) else 0.0
    sessions_w = float(sw.mean()) if len(sw) else 0.0
    long_km = float(df["distance_km"].max()) if df["distance_km"].notna().any() else 0.0

    return {"w_km": w_km, "long_km": long_km, "sessions_w": sessions_w}

def make_weekly_feedback_pro(target_df: pd.DataFrame, ts_map: Dict[str, pd.DataFrame], modality: str, hrmax: int) -> str:
    """
    Feedback PRO:
    - volumen, sesiones, larga
    - distribución de zonas (si hay HR)
    - deriva cardiaca de la sesión larga (si hay HR)
    """
    if target_df.empty or target_df["start_dt"].isna().all():
        return "No hay datos suficientes para generar feedback."

    df = target_df.copy()
    df["start_dt"] = pd.to_datetime(df["start_dt"], errors="coerce")
    df = df[df["start_dt"].notna()].sort_values("start_dt")
    if df.empty:
        return "No hay datos suficientes para generar feedback."

    today = pd.Timestamp(date.today())
    this_week_start = _week_start_monday(today)
    df["week_start"] = df["start_dt"].apply(_week_start_monday)
    complete_weeks = df[df["week_start"] < this_week_start]
    if complete_weeks.empty:
        window_start = today - pd.Timedelta(days=7)
        w = df[df["start_dt"] >= window_start]
        label = f"Últimos 7 días (desde {window_start.date()} hasta {today.date()})"
    else:
        last_week = complete_weeks["week_start"].max()
        w = df[df["week_start"] == last_week]
        label = f"Semana (lunes-domingo) desde {last_week.date()}"

    km = float(w["distance_km"].fillna(0).sum())
    sessions = int(w.shape[0])
    long_idx = w["distance_km"].fillna(0).idxmax() if sessions else None
    long_row = w.loc[long_idx] if long_idx is not None else None
    long_km = float(long_row["distance_km"]) if long_row is not None else 0.0

    minutes = float(w["moving_min"].fillna(0).sum()) if "moving_min" in w.columns else 0.0

    # Zonas para esa semana
    z_week = weekly_zone_distribution(w, ts_map, hrmax)
    z_line = ""
    z_warn = ""
    if not z_week.empty:
        last = z_week.sort_values("week_start").tail(1).iloc[0]
        z2 = last.get("Z2_pct")
        z3 = last.get("Z3_pct")
        z4 = last.get("Z4_pct")
        if pd.notna(z2):
            z_line = f"**Distribución de zonas (con HR):** Z2 {z2*100:.0f}% · Z3 {z3*100:.0f}% · Z4 {z4*100:.0f}%"
            # alerta simple: demasiado Z3+Z4
            if (z3 + z4) > 0.45:
                z_warn = "Estás pasando mucho tiempo en Z3–Z4. Si tu objetivo es base/resistencia, prueba a mover más tiempo a Z2."

    # Deriva de la sesión larga
    drift_line = ""
    drift = None
    if long_row is not None:
        ts_long = ts_map.get(long_row.get("key"))
        if ts_long is not None and not ts_long.empty:
            drift = _hr_drift(ts_long)
            if drift is not None:
                drift_line = f"**Deriva cardiaca (sesión larga):** {drift*100:.1f}% (ideal: baja; si es alta, baja ritmo o mejora recuperación)"

    # Próximos pasos
    actions = []
    if sessions < 3:
        actions.append("Intenta llegar a 3 sesiones esta semana (aunque alguna sea corta y muy suave).")
    if modality == "correr" and long_km < 12:
        actions.append("Añade una tirada larga suave (Z2) semanal y súbela progresivamente.")
    if modality == "bicicleta" and long_km < 50:
        actions.append("Añade una salida larga Z2 y practica hidratación/comida.")
    if z_warn:
        actions.append("Haz los rodajes realmente suaves (Z2) y deja 1 día para calidad (tempo/intervalos).")
    if drift is not None and drift > 0.06:
        actions.append("Tu deriva es algo alta: esta semana prioriza descanso y baja un punto la intensidad de los rodajes.")

    if not actions:
        actions = [
            "Mantén 1 sesión de calidad y el resto fácil.",
            "Asegura 1–2 sesiones de fuerza + core (20–35').",
            "Prioriza sueño e hidratación.",
        ]

    txt = f"""### Feedback semanal PRO

**Periodo analizado:** {label}  
**Hechos:** {sessions} sesiones · {km:.1f} km · {minutes:.0f} min · sesión más larga {long_km:.1f} km
"""
    if z_line:
        txt += f"\n{z_line}\n"
    if z_warn:
        txt += f"\n**Ajuste recomendado:** {z_warn}\n"
    if drift_line:
        txt += f"\n{drift_line}\n"

    txt += "\n**Próximos pasos (1–3)**\n"
    for i, a in enumerate(actions[:3], start=1):
        txt += f"{i}. {a}\n"
    return txt

# ============================================================
# Generación del plan (igual que tu versión anterior)
# ============================================================

@dataclass
class UserProfile:
    age: int
    height_cm: float
    weight_kg: float
    days_per_week: int
    goal: str
    modality: str  # "correr" o "bicicleta"
    start_date: date

def phase_for_week(week_idx: int) -> str:
    if week_idx <= 16:
        return "Base"
    if week_idx <= 32:
        return "Construcción"
    if week_idx <= 46:
        return "Específico"
    return "Afinado"

def weekly_volume_target(profile: UserProfile, week_idx: int, baseline_w: float) -> float:
    deload = (week_idx % 4 == 0)
    growth = 1.0 + 0.012 * week_idx
    if profile.modality == "correr":
        base = max(18.0, baseline_w) if baseline_w > 0 else 22.0
        cap = 70.0 if profile.goal == "maraton" else 55.0 if profile.goal == "media" else 45.0
    else:
        base = max(60.0, baseline_w) if baseline_w > 0 else 90.0
        cap = 300.0 if profile.goal == "gran_fondo" else 240.0 if profile.goal == "puerto" else 220.0 if profile.goal == "criterium" else 200.0
    target = min(base * growth, cap)
    if deload:
        target *= 0.78
    return round(target, 1)

def long_session_target(profile: UserProfile, week_idx: int, baseline_long: float) -> float:
    deload = (week_idx % 4 == 0)
    growth = 1.0 + 0.015 * week_idx
    if profile.modality == "correr":
        base = max(8.0, baseline_long * 0.85) if baseline_long > 0 else 10.0
        cap = 32.0 if profile.goal == "maraton" else 24.0 if profile.goal == "media" else 18.0 if profile.goal == "10k" else 20.0
    else:
        base = max(25.0, baseline_long * 0.85) if baseline_long > 0 else 40.0
        cap = 140.0 if profile.goal == "gran_fondo" else 110.0 if profile.goal == "puerto" else 90.0 if profile.goal == "criterium" else 100.0
    target = min(base * growth, cap)
    if deload:
        target *= 0.75
    return round(target, 1)

def workout_templates(phase: str, goal: str, hrmax: int, modality: str) -> Dict[str, Dict]:
    z = hr_zones(hrmax)
    if modality == "bicicleta":
        return {
            "easy": {"title": "Rodaje Z2 (bici)", "details": f"Z2 aprox {z['Z2'][0]}–{z['Z2'][1]} ppm. Cadencia cómoda. RPE 3–4/10."},
            "tempo": {"title": "Sweet spot (bici)", "details": "Calentar 15'. 3x10' fuerte pero sostenible (RPE 7/10) con 5' suave. Enfriar 10'."},
            "intervals": {"title": "VO2 (bici)", "details": "Calentar 15'. 6x3' muy fuerte (RPE 8–9/10) con 3' suave. Enfriar 10'."},
            "progressive": {"title": "Progresivo (bici)", "details": "60–90' empezando fácil y acabando 15–20' a ritmo sostenido (RPE 7/10)."},
            "long": {"title": "Salida larga (bici)", "details": "Mayormente Z2. Practica hidratación y comida cada 20–30'."},
            "recovery": {"title": "Recuperación (bici)", "details": "30–45' muy suave (Z1–Z2) + movilidad 10'."},
            "strength": {"title": "Fuerza + core", "details": "Fuerza total (pierna + tronco) y core. Cargas moderadas, técnica limpia."},
        }
    return {
        "easy": {"title": "Rodaje suave", "details": f"Z2 (aprox {z['Z2'][0]}–{z['Z2'][1]} ppm). Ritmo cómodo, respiración controlada."},
        "tempo": {"title": "Tempo", "details": f"Calentamiento 15'. 3x8' en Z3 (aprox {z['Z3'][0]}–{z['Z3'][1]} ppm) con 3' suave. Enfriar 10'."},
        "intervals": {"title": "Intervalos", "details": f"Calentamiento 15'. 6x3' en Z4 (aprox {z['Z4'][0]}–{z['Z4'][1]} ppm) con 2' suave. Enfriar 10'."},
        "progressive": {"title": "Progresivo", "details": "45–60' empezando en Z2 y acabando 10–15' en Z3."},
        "long": {"title": "Tirada larga", "details": "Mayormente Z2. Si fase Específico y objetivo maratón: últimos 20' en Z3 si te encuentras bien."},
        "recovery": {"title": "Recuperación", "details": "30–40' muy suave en Z1–Z2 + movilidad 10'."},
        "strength": {"title": "Fuerza + core", "details": "Fuerza total (pierna + tronco) y core. Cargas moderadas, técnica limpia."},
    }

def distribute_week_km(total_km: float, long_km: float, days: int) -> List[float]:
    days = max(3, int(days))
    remain = max(0.0, total_km - long_km)
    if days == 3:
        parts = [remain * 0.45, remain * 0.55]
    elif days == 4:
        parts = [remain * 0.25, remain * 0.30, remain * 0.45]
    elif days == 5:
        parts = [remain * 0.18, remain * 0.20, remain * 0.25, remain * 0.37]
    else:
        parts = [remain * 0.12, remain * 0.15, remain * 0.18, remain * 0.20, remain * 0.35]
    return [round(x, 1) for x in parts] + [round(long_km, 1)]

def build_plan(profile: UserProfile, baseline: Dict[str, float], hrmax: int) -> pd.DataFrame:
    start = next_monday(profile.start_date)
    rows = []
    for w in range(1, 53):
        phase = phase_for_week(w)
        w_km = weekly_volume_target(profile, w, baseline["w_km"])
        l_km = long_session_target(profile, w, baseline["long_km"])
        if l_km > 0.45 * w_km:
            w_km = round(l_km / 0.45, 1)

        km_sessions = distribute_week_km(w_km, l_km, profile.days_per_week)
        templates = workout_templates(phase, profile.goal, hrmax, profile.modality)
        week_start = start + timedelta(weeks=w - 1)

        if profile.days_per_week >= 6:
            dmap = {
                0: ("easy", km_sessions[0]),
                1: ("intervals" if phase in ("Construcción", "Específico") else "progressive", km_sessions[1]),
                2: ("easy", km_sessions[2]),
                3: ("tempo" if phase != "Base" else "progressive", km_sessions[3]),
                4: ("recovery", max(4.0, km_sessions[4])),
                5: ("easy", km_sessions[5] if len(km_sessions) > 5 else max(6.0, (w_km - l_km) * 0.2)),
                6: ("long", km_sessions[-1]),
            }
            strength_days = [2, 4]
        elif profile.days_per_week == 5:
            dmap = {
                0: ("easy", km_sessions[0]),
                1: ("intervals" if phase in ("Construcción", "Específico") else "progressive", km_sessions[1]),
                3: ("tempo" if phase != "Base" else "progressive", km_sessions[2]),
                5: ("easy", km_sessions[3]),
                6: ("long", km_sessions[-1]),
            }
            strength_days = [2, 4]
        elif profile.days_per_week == 4:
            dmap = {
                1: ("intervals" if phase in ("Construcción", "Específico") else "progressive", km_sessions[0]),
                3: ("easy", km_sessions[1]),
                5: ("tempo" if phase != "Base" else "progressive", km_sessions[2]),
                6: ("long", km_sessions[-1]),
            }
            strength_days = [0, 2]
        else:
            dmap = {
                1: ("intervals" if phase in ("Construcción", "Específico") else "progressive", km_sessions[0]),
                3: ("easy", km_sessions[1]),
                6: ("long", km_sessions[-1]),
            }
            strength_days = [0, 4]

        for dow in range(7):
            this_day = week_start + timedelta(days=dow)
            day_name = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"][dow]
            modality_name = "Correr" if profile.modality == "correr" else "Bicicleta"

            if dow in dmap:
                wtype, km = dmap[dow]
                t = templates[wtype]
                extra = ""
                if wtype == "recovery":
                    extra = "Movilidad 10': " + "; ".join(MOBILITY_LIBRARY["movilidad_10"])
                if wtype in ("easy", "long") and phase == "Base" and (w % 2 == 1):
                    extra = (extra + " | " if extra else "") + "Core 10': " + "; ".join(CORE_LIBRARY["core_10"])

                rows.append(
                    {
                        "Fecha": this_day,
                        "Día": day_name,
                        "Modalidad": modality_name,
                        "Fase": phase,
                        "Sesión": t["title"],
                        "Volumen (km)": float(km),
                        "Detalles": t["details"] + ((" | " + extra) if extra else ""),
                    }
                )
            elif dow in strength_days:
                strength_key = "fuerza_35" if phase in ("Construcción", "Específico") else "fuerza_25"
                rows.append(
                    {
                        "Fecha": this_day,
                        "Día": day_name,
                        "Modalidad": modality_name,
                        "Fase": phase,
                        "Sesión": "Fuerza + core",
                        "Volumen (km)": 0.0,
                        "Detalles": "Fuerza: "
                        + "; ".join(STRENGTH_LIBRARY[strength_key])
                        + " | Core: "
                        + "; ".join(CORE_LIBRARY["core_20"]),
                    }
                )
            else:
                rows.append(
                    {
                        "Fecha": this_day,
                        "Día": day_name,
                        "Modalidad": modality_name,
                        "Fase": phase,
                        "Sesión": "Descanso / movilidad",
                        "Volumen (km)": 0.0,
                        "Detalles": "Movilidad 20': " + "; ".join(MOBILITY_LIBRARY["movilidad_20"]),
                    }
                )

    return pd.DataFrame(rows)

def plan_to_excel_bytes(plan: pd.DataFrame) -> bytes:
    plan = plan.copy()
    plan["Fecha"] = pd.to_datetime(plan["Fecha"])
    plan["Año"] = plan["Fecha"].dt.year
    plan["Mes"] = plan["Fecha"].dt.month

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for (y, m), chunk in plan.groupby(["Año", "Mes"], sort=True):
            month_name = chunk["Fecha"].dt.strftime("%B").iloc[0]
            sheet_name = f"{month_name[:28]}"
            chunk = chunk.sort_values("Fecha")[
                ["Fecha", "Día", "Modalidad", "Fase", "Sesión", "Volumen (km)", "Detalles"]
            ]
            chunk.to_excel(writer, index=False, sheet_name=sheet_name)
            ws = writer.book[sheet_name]
            ws.freeze_panes = "A2"
            widths = [12, 12, 12, 14, 22, 12, 80]
            for i, w in enumerate(widths, start=1):
                ws.column_dimensions[chr(64 + i)].width = w

    return output.getvalue()

# ============================================================
# App
# ============================================================

st.set_page_config(page_title="Planificador PRO", layout="wide")
st.title("Planificador PRO (ZIP completo con datos detallados)")
st.info("Sube el ZIP completo (Strava / Garmin / otros). Si dentro hay FIT/TCX, usaré pulsaciones y ritmo/velocidad en el análisis semanal.")

uploaded = st.file_uploader(
    "Sube tu ZIP completo (recomendado)",
    type=["zip"],
)

with st.sidebar:
    st.header("Modalidad")
    modality = st.selectbox("Selecciona modalidad", options=["correr", "bicicleta"])

    st.header("Perfil")
    age = st.number_input("Edad", min_value=12, max_value=90, value=36, step=1)
    height_cm = st.number_input("Altura (cm)", min_value=120, max_value=220, value=180, step=1)
    weight_kg = st.number_input("Peso (kg)", min_value=35.0, max_value=200.0, value=80.0, step=0.5)

    st.header("Objetivo")
    goal = st.selectbox(
        "Selecciona objetivo principal",
        options=GOALS[modality],
        format_func=lambda x: x[1],
    )[0]

    days_per_week = st.slider("Días de entrenamiento por semana", min_value=3, max_value=6, value=5)
    start_date = st.date_input("Fecha de inicio", value=date.today())

if uploaded is None:
    st.stop()

# Parse ZIP PRO
activities_df, ts_map = parse_zip_pro(uploaded.getvalue())
if activities_df.empty:
    st.error("No he podido leer actividades del ZIP. Prueba con un ZIP que contenga FIT/TCX/GPX o activities.csv.")
    st.stop()

# Filtrar por modalidad
sport_needed = "run" if modality == "correr" else "ride"
target = activities_df[activities_df["sport"] == sport_needed].copy()

# Si no hay deporte (todo other), permite asignar
if target.empty and (activities_df["sport"] == "other").any():
    st.warning("No he podido identificar correr/bici en tus archivos (típico si solo hay GPX). Puedes asignarlo.")
    assign_all = st.selectbox("Asigna las actividades como:", options=["correr", "bicicleta"])
    activities_df.loc[activities_df["sport"] == "other", "sport"] = "run" if assign_all == "correr" else "ride"
    target = activities_df[activities_df["sport"] == sport_needed].copy()

hrmax = estimate_hrmax(int(age))
baseline = weekly_baseline_simple(target) if not target.empty else {"w_km": 0.0, "long_km": 0.0, "sessions_w": 0.0}

st.subheader("Lectura rápida del ZIP")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Actividades detectadas (modalidad)", int(target.shape[0]))
c2.metric("Km/semana (media ~6 semanas)", f"{baseline['w_km']:.1f}")
c3.metric("Sesión más larga aprox", f"{baseline['long_km']:.1f} km")
c4.metric("HRmáx estimada (aprox)", f"{hrmax} ppm")

with st.expander("Ver tabla de actividades detectadas"):
    show = target.copy()
    show["start_dt"] = pd.to_datetime(show["start_dt"], errors="coerce")
    show = show.sort_values("start_dt", ascending=False)
    cols = ["start_dt", "distance_km", "moving_min", "avg_hr", "source", "file"]
    st.dataframe(show[cols].head(200), use_container_width=True)

st.subheader("Feedback PRO")
if st.button("Generar feedback semanal PRO"):
    st.markdown(make_weekly_feedback_pro(target, ts_map, modality, hrmax))

st.subheader("Plan anual")
profile = UserProfile(
    age=int(age),
    height_cm=float(height_cm),
    weight_kg=float(weight_kg),
    days_per_week=int(days_per_week),
    goal=str(goal),
    modality=str(modality),
    start_date=start_date,
)
plan = build_plan(profile, baseline, hrmax)
st.dataframe(plan.head(21), use_container_width=True)

excel_bytes = plan_to_excel_bytes(plan)
st.download_button(
    label="Descargar plan anual en Excel",
    data=excel_bytes,
    file_name=f"plan_entrenamiento_anual_PRO_{modality}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
