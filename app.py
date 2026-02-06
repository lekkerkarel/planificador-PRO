import io
import zipfile
import gzip
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from dateutil import parser as dtparser

import gpxpy
from fitparse import FitFile
from lxml import etree

# ============================================================
# Planificador PRO (con funcionalidades del Universal)
# - Acepta: CSV (Strava), GPX/TCX/FIT (Garmin/Komoot/otros), ZIP completo
# - PRO: si hay FIT/TCX, usa datos punto a punto (HR/speed/distance)
# ============================================================

RUN_TYPES = {"run", "running", "carrera", "trail run", "treadmill", "virtual run"}
BIKE_TYPES = {
    "ride", "cycling", "bicicleta", "bike", "road", "road cycling", "mountain bike",
    "mtb", "gravel ride", "virtual ride", "e-bike ride", "ebike ride",
}

GOALS = {
    "correr": [
        ("5k", "Carrera 5K"),
        ("10k", "Carrera 10K"),
        ("media", "Media maratón 21K"),
        ("maraton", "Maratón 42K"),
        ("base", "Solo base aeróbica"),
        ("resistencia", "Mejorar resistencia y bajar pulsaciones"),
    ],
    "bicicleta": [
        ("gran_fondo", "Gran fondo (larga distancia, en ~12 meses)"),
        ("resistencia", "Mejorar resistencia y bajar pulsaciones"),
        ("puerto", "Subidas / puertos (mejorar umbral en subida)"),
        ("criterium", "Potencia y cambios de ritmo (tipo criterium)"),
        ("base", "Solo base aeróbica"),
    ],
}

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

def _to_naive_utc(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    try:
        return dt.dt.tz_convert(None)
    except Exception:
        return pd.to_datetime(dt, errors="coerce")

def _week_start_monday(ts: pd.Timestamp) -> pd.Timestamp:
    d = ts.normalize()
    return d - pd.Timedelta(days=d.weekday())

# ============================================================
# Parsers
# ============================================================

def _records_to_timeseries(records: List[dict]) -> pd.DataFrame:
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
    dt = ts["timestamp"].diff().dt.total_seconds()
    dt = dt.where((dt.notna()) & (dt >= 0) & (dt <= 60), 0)
    moving_s = float(dt.sum())

    distance_m = None
    if ts["distance_m"].notna().any():
        d0 = float(ts["distance_m"].dropna().iloc[0])
        d1 = float(ts["distance_m"].dropna().iloc[-1])
        distance_m = max(0.0, d1 - d0)
    elif ts["speed_mps"].notna().any():
        distance_m = float((ts["speed_mps"].fillna(0) * dt.fillna(0)).sum())

    avg_hr = float(ts["hr"].dropna().mean()) if ts["hr"].notna().any() else None
    avg_speed = float(ts["speed_mps"].dropna().mean()) if ts["speed_mps"].notna().any() else None
    return {"moving_s": moving_s, "distance_m": distance_m, "avg_hr": avg_hr, "avg_speed_mps": avg_speed}

def parse_fit_pro(file_bytes: bytes) -> Tuple[dict, pd.DataFrame]:
    fitfile = FitFile(BytesIO(file_bytes))
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

    recs = []
    for r in fitfile.get_messages("record"):
        fields = {f.name: f.value for f in r}
        ts = fields.get("timestamp")
        if ts is None:
            continue
        recs.append({"timestamp": ts, "hr": fields.get("heart_rate"), "speed_mps": fields.get("speed"), "distance_m": fields.get("distance")})
    ts_df = _records_to_timeseries(recs)
    summ2 = _summary_from_timeseries(ts_df)
    if start_dt is None and not ts_df.empty:
        start_dt = ts_df["timestamp"].iloc[0]

    summary = {
        "start_dt": start_dt,
        "sport": sport,
        "distance_km": (summ2["distance_m"] / 1000.0) if summ2["distance_m"] is not None else None,
        "moving_min": (summ2["moving_s"] / 60.0) if summ2["moving_s"] is not None else None,
        "avg_hr": summ2["avg_hr"],
        "avg_speed_mps": summ2["avg_speed_mps"],
        "source": "fit_pro",
    }
    return summary, ts_df

def parse_tcx_pro(file_bytes: bytes) -> Tuple[dict, pd.DataFrame]:
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
    df_raw = pd.read_csv(BytesIO(file_bytes))
    cols = {c.strip().lower(): c for c in df_raw.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    c_type = pick("activity type", "type")
    c_date = pick("activity date", "date", "start date", "start_time", "start time")
    c_dist = pick("distance")
    c_move = pick("moving time", "moving_time", "duration", "elapsed time")
    c_hr = pick("average heart rate", "avg heart rate", "avg_hr")

    if not (c_type and c_date and c_dist):
        return pd.DataFrame(columns=["start_dt","sport","distance_km","moving_min","avg_hr","avg_speed_mps","source"])

    df = df_raw.copy()
    df["start_dt"] = _to_naive_utc(df[c_date])
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
    z = zipfile.ZipFile(BytesIO(file_bytes))

    file_counts = {"fit":0,"fit.gz":0,"tcx":0,"tcx.gz":0,"gpx":0,"gpx.gz":0,"csv":0}
    for n in z.namelist():
        ln = n.lower()
        if ln.endswith(".fit.gz"): file_counts["fit.gz"] += 1
        elif ln.endswith(".fit"): file_counts["fit"] += 1
        elif ln.endswith(".tcx.gz"): file_counts["tcx.gz"] += 1
        elif ln.endswith(".tcx"): file_counts["tcx"] += 1
        elif ln.endswith(".gpx.gz"): file_counts["gpx.gz"] += 1
        elif ln.endswith(".gpx"): file_counts["gpx"] += 1
        elif ln.endswith(".csv"): file_counts["csv"] += 1

    activities: List[dict] = []
    ts_map: Dict[str, pd.DataFrame] = {}
    csv_frames = []

    for name in z.namelist():
        lname = name.lower()
        try:
            raw = z.read(name)
        except Exception:
            continue

        data = raw
        if lname.endswith(".gz"):
            try:
                data = gzip.decompress(raw)
            except Exception:
                data = raw

        if lname.endswith("activities.csv") or lname.endswith("activity.csv"):
            try:
                csv_frames.append(parse_strava_csv_basic(data))
            except Exception:
                pass

        if lname.endswith(".fit") or lname.endswith(".fit.gz"):
            try:
                summ, ts = parse_fit_pro(data)
                key = f"{name}"
                activities.append({**summ, "key": key, "file": name})
                if ts is not None and not ts.empty:
                    ts_map[key] = ts
            except Exception:
                continue

        elif lname.endswith(".tcx") or lname.endswith(".tcx.gz"):
            try:
                summ, ts = parse_tcx_pro(data)
                key = f"{name}"
                activities.append({**summ, "key": key, "file": name})
                if ts is not None and not ts.empty:
                    ts_map[key] = ts
            except Exception:
                continue

        elif lname.endswith(".gpx") or lname.endswith(".gpx.gz"):
            try:
                summ = parse_gpx_basic(data)
                key = f"{name}"
                activities.append({**summ, "key": key, "file": name})
            except Exception:
                continue

    act_df = pd.DataFrame(activities)

    if act_df.empty and csv_frames:
        act_df = pd.concat(csv_frames, ignore_index=True)
        act_df["key"] = act_df.index.astype(str)
        act_df["file"] = "activities.csv"
    elif (not act_df.empty) and csv_frames:
        csv_df = pd.concat(csv_frames, ignore_index=True)
        act_df["start_dt"] = _to_naive_utc(act_df.get("start_dt"))
        csv_df["start_dt"] = _to_naive_utc(csv_df["start_dt"])

        for idx, row in act_df.iterrows():
            if str(row.get("sport")) != "other":
                continue
            if pd.isna(row.get("start_dt")) or pd.isna(row.get("distance_km")):
                continue
            window = csv_df[
                csv_df["start_dt"].between(row["start_dt"] - pd.Timedelta(hours=6), row["start_dt"] + pd.Timedelta(hours=6))
            ].copy()
            if window.empty:
                continue
            window["dist_diff"] = (window["distance_km"] - float(row["distance_km"])).abs()
            best = window.sort_values("dist_diff").head(1)
            if not best.empty and float(best["dist_diff"].iloc[0]) <= 0.3:
                act_df.at[idx, "sport"] = best["sport"].iloc[0]

    if act_df.empty:
        act_df = pd.DataFrame(columns=["start_dt","sport","distance_km","moving_min","avg_hr","avg_speed_mps","source","key","file"])

    act_df["start_dt"] = _to_naive_utc(act_df.get("start_dt"))
    for c in ["distance_km","moving_min","avg_hr","avg_speed_mps"]:
        if c not in act_df.columns:
            act_df[c] = np.nan
        act_df[c] = pd.to_numeric(act_df[c], errors="coerce")

    act_df.attrs["file_counts"] = file_counts
    return act_df, ts_map

def parse_any_upload_pro(uploaded_file) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    b = uploaded_file.getvalue()
    name = (uploaded_file.name or "").lower()

    if name.endswith(".zip"):
        return parse_zip_pro(b)

    if name.endswith(".fit"):
        summ, ts = parse_fit_pro(b)
        df = pd.DataFrame([{**summ, "key": name, "file": name}])
        df["start_dt"] = _to_naive_utc(df["start_dt"])
        return df, ({name: ts} if ts is not None and not ts.empty else {})

    if name.endswith(".tcx"):
        summ, ts = parse_tcx_pro(b)
        df = pd.DataFrame([{**summ, "key": name, "file": name}])
        df["start_dt"] = _to_naive_utc(df["start_dt"])
        return df, ({name: ts} if ts is not None and not ts.empty else {})

    if name.endswith(".gpx"):
        summ = parse_gpx_basic(b)
        df = pd.DataFrame([{**summ, "key": name, "file": name}])
        df["start_dt"] = _to_naive_utc(df["start_dt"])
        return df, {}

    if name.endswith(".csv"):
        df = parse_strava_csv_basic(b)
        if df.empty:
            return pd.DataFrame(columns=["start_dt","sport","distance_km","moving_min","avg_hr","avg_speed_mps","source","key","file"]), {}
        df = df.copy()
        df["key"] = df.index.astype(str)
        df["file"] = "activities.csv"
        df["start_dt"] = _to_naive_utc(df["start_dt"])
        return df, {}

    return pd.DataFrame(columns=["start_dt","sport","distance_km","moving_min","avg_hr","avg_speed_mps","source","key","file"]), {}

# ============================================================
# Métricas / feedback
# ============================================================

def _zone_seconds(ts: pd.DataFrame, hrmax: int) -> Dict[str, float]:
    z = hr_zones(hrmax)
    out = {k: 0.0 for k in z.keys()}
    if ts is None or ts.empty or ts["hr"].notna().sum() < 10:
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
    if ts is None or ts.empty or ts["hr"].notna().sum() < 60:
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

def weekly_zone_distribution(target_df: pd.DataFrame, ts_map: Dict[str, pd.DataFrame], hrmax: int) -> pd.DataFrame:
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

def make_weekly_feedback_basic(target_df: pd.DataFrame, modality: str, hrmax: int) -> str:
    if target_df.empty or target_df["start_dt"].isna().all():
        return "No hay datos suficientes para generar feedback semanal."

    df = target_df.copy()
    df["start_dt"] = pd.to_datetime(df["start_dt"], errors="coerce")
    df = df[df["start_dt"].notna()].sort_values("start_dt")
    if df.empty:
        return "No hay datos suficientes para generar feedback semanal."

    df["week_start"] = df["start_dt"].apply(_week_start_monday)
    today = pd.Timestamp(date.today())
    this_week_start = _week_start_monday(today)
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
    long_km = float(w["distance_km"].fillna(0).max()) if sessions else 0.0
    minutes = float(w["moving_min"].fillna(0).sum()) if "moving_min" in w.columns else 0.0

    avg_hr = None
    if "avg_hr" in w.columns and w["avg_hr"].notna().any():
        avg_hr = float(w["avg_hr"].dropna().mean())

    good, warn, actions = [], [], []
    if sessions >= 4:
        good.append("Buena consistencia: 4 o más sesiones.")
    elif sessions >= 3:
        good.append("Consistencia correcta: 3 sesiones.")
    else:
        warn.append("Poca consistencia esta semana. Si puedes, intenta 3–4 sesiones.")

    if modality == "correr":
        if long_km >= 18:
            good.append("Tirada larga sólida para construir resistencia.")
        elif long_km >= 12:
            good.append("Tirada larga correcta para tu base actual.")
        else:
            actions.append("Incluye una tirada larga suave (Z2) semanal y súbela poco a poco.")
    else:
        if long_km >= 80:
            good.append("Salida larga sólida para base ciclista.")
        elif long_km >= 50:
            good.append("Salida larga correcta para tu base actual.")
        else:
            actions.append("Incluye una salida larga Z2 y practica hidratación/comida.")

    if avg_hr is not None:
        z = hr_zones(hrmax)
        if avg_hr >= z["Z3"][0]:
            warn.append("La FC media de la semana es alta para construir base. Puede que estés yendo demasiado fuerte.")
            actions.append("Haz rodajes en Z2 y deja la intensidad para 1 día (tempo/intervalos).")
        else:
            good.append("FC media compatible con trabajo aeróbico (base).")

    if not actions:
        actions = [
            "Mantén 1 sesión de calidad (tempo/intervalos) y el resto fácil.",
            "Asegura 1–2 sesiones de fuerza + core (20–35').",
            "Prioriza sueño e hidratación.",
        ]

    txt = f"""### Feedback semanal

**Periodo analizado:** {label}  
**Hechos:** {sessions} sesiones · {km:.1f} km · {minutes:.0f} min · sesión más larga {long_km:.1f} km"""
    if avg_hr is not None:
        txt += f" · FC media aprox {avg_hr:.0f} ppm\n"
    else:
        txt += "\n"

    if good:
        txt += "\n**Lo que va bien**\n"
        for g in good:
            txt += f"- {g}\n"

    if warn:
        txt += "\n**Ajustes o alertas**\n"
        for w0 in warn:
            txt += f"- {w0}\n"

    txt += "\n**Próximos pasos (1–3)**\n"
    for i, a in enumerate(actions[:3], start=1):
        txt += f"{i}. {a}\n"
    return txt

def make_weekly_feedback_pro(target_df: pd.DataFrame, ts_map: Dict[str, pd.DataFrame], modality: str, hrmax: int) -> str:
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
            if (z3 + z4) > 0.45:
                z_warn = "Estás pasando mucho tiempo en Z3–Z4. Si tu objetivo es base/resistencia, prueba a mover más tiempo a Z2."

    drift_line = ""
    drift = None
    if long_row is not None:
        ts_long = ts_map.get(long_row.get("key"))
        if ts_long is not None and not ts_long.empty:
            drift = _hr_drift(ts_long)
            if drift is not None:
                drift_line = f"**Deriva cardiaca (sesión larga):** {drift*100:.1f}% (ideal: baja; si es alta, baja ritmo o mejora recuperación)"

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
# Plan anual (1–7 días)
# ============================================================

@dataclass
class UserProfile:
    age: int
    height_cm: float
    weight_kg: float
    days_per_week: int
    goal: str
    modality: str
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
    days = int(days)
    days = max(1, min(7, days))
    if days == 1:
        return [round(max(total_km, long_km), 1)]

    long_km = min(long_km, total_km) if total_km and total_km > 0 else long_km
    remain = max(0.0, (total_km or 0.0) - (long_km or 0.0))

    n = days - 1
    if n == 1:
        weights = [1.0]
    elif n == 2:
        weights = [0.45, 0.55]
    elif n == 3:
        weights = [0.25, 0.30, 0.45]
    elif n == 4:
        weights = [0.18, 0.20, 0.25, 0.37]
    elif n == 5:
        weights = [0.12, 0.15, 0.18, 0.20, 0.35]
    else:
        weights = [0.10, 0.12, 0.14, 0.16, 0.18, 0.30]

    sess = [round(remain * w, 1) for w in weights] + [round(long_km or 0.0, 1)]
    diff = round((total_km or 0.0) - sum(sess), 1)
    if abs(diff) >= 0.1 and len(sess) >= 2:
        sess[0] = round(max(0.0, sess[0] + diff), 1)
    return sess

def build_plan(profile: UserProfile, baseline: Dict[str, float], hrmax: int) -> pd.DataFrame:
    start = next_monday(profile.start_date)
    rows = []
    for w in range(1, 53):
        phase = phase_for_week(w)
        w_km = weekly_volume_target(profile, w, baseline["w_km"])
        l_km = long_session_target(profile, w, baseline["long_km"])
        if l_km > 0.45 * w_km:
            w_km = round(l_km / 0.45, 1)

        templates = workout_templates(phase, profile.goal, hrmax, profile.modality)
        week_start = start + timedelta(weeks=w - 1)

        dmap = {}
        strength_days = []

        quality1 = "intervals" if phase in ("Construcción", "Específico") else "progressive"
        quality2 = "tempo" if phase != "Base" else "progressive"
        dpw = int(profile.days_per_week)

        if dpw == 7:
            train_days = [0, 1, 2, 3, 4, 5, 6]
        elif dpw == 6:
            train_days = [0, 1, 2, 3, 5, 6]
        elif dpw == 5:
            train_days = [0, 1, 3, 5, 6]
        elif dpw == 4:
            train_days = [1, 3, 5, 6]
        elif dpw == 3:
            train_days = [1, 3, 6]
        elif dpw == 2:
            train_days = [3, 6]
        else:
            train_days = [6]

        km_sessions = distribute_week_km(w_km, l_km, dpw)

        if dpw == 1:
            types = ["long"]
        elif dpw == 2:
            types = ["easy", "long"]
        elif dpw == 3:
            types = [quality1, "easy", "long"]
        elif dpw == 4:
            types = [quality1, "easy", quality2, "long"]
        elif dpw == 5:
            types = ["easy", quality1, quality2, "easy", "long"]
        elif dpw == 6:
            types = ["easy", quality1, "easy", quality2, "easy", "long"]
        else:
            types = ["easy", quality1, "easy", quality2, "recovery", "easy", "long"]

        for idx, dow in enumerate(train_days):
            dmap[dow] = (types[idx], km_sessions[idx] if idx < len(km_sessions) else 0.0)

        if dpw >= 5:
            strength_days = [2, 4]
        elif dpw == 4:
            strength_days = [0, 2]
        elif dpw == 3:
            strength_days = [0, 4]
        elif dpw == 2:
            strength_days = [0, 2]
        else:
            strength_days = [2]

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
                    {"Fecha": this_day, "Día": day_name, "Modalidad": modality_name, "Fase": phase, "Sesión": t["title"], "Volumen (km)": float(km),
                     "Detalles": t["details"] + ((" | " + extra) if extra else "")}
                )
            elif dow in strength_days:
                strength_key = "fuerza_35" if phase in ("Construcción", "Específico") else "fuerza_25"
                rows.append(
                    {"Fecha": this_day, "Día": day_name, "Modalidad": modality_name, "Fase": phase, "Sesión": "Fuerza + core", "Volumen (km)": 0.0,
                     "Detalles": "Fuerza: " + "; ".join(STRENGTH_LIBRARY[strength_key]) + " | Core: " + "; ".join(CORE_LIBRARY["core_20"])}
                )
            else:
                rows.append(
                    {"Fecha": this_day, "Día": day_name, "Modalidad": modality_name, "Fase": phase, "Sesión": "Descanso / movilidad", "Volumen (km)": 0.0,
                     "Detalles": "Movilidad 20': " + "; ".join(MOBILITY_LIBRARY["movilidad_20"])}
                )

    return pd.DataFrame(rows)

def plan_to_excel_bytes(plan: pd.DataFrame) -> bytes:
    plan = plan.copy()
    plan["Fecha"] = pd.to_datetime(plan["Fecha"])
    plan["Año"] = plan["Fecha"].dt.year
    plan["Mes"] = plan["Fecha"].dt.month

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for (_, _), chunk in plan.groupby(["Año", "Mes"], sort=True):
            month_name = chunk["Fecha"].dt.strftime("%B").iloc[0]
            sheet_name = f"{month_name[:28]}"
            chunk = chunk.sort_values("Fecha")[["Fecha", "Día", "Modalidad", "Fase", "Sesión", "Volumen (km)", "Detalles"]]
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
st.title("Planificador PRO (con lectura multi-plataforma)")
st.info("Sube tus actividades (CSV de Strava, GPX/TCX/FIT o ZIP completo). Si hay FIT/TCX, se usarán datos detallados para el feedback PRO.")

with st.expander("Cómo descargar el CSV en Strava"):
    st.markdown(
        """
1. Inicia sesión en Strava desde un ordenador y haz clic en tu foto de perfil. Selecciona **Ajustes**.  
2. En el menú lateral izquierdo, entra en **Mi cuenta**.  
3. Accede a **Descarga o elimina tu cuenta** y pulsa en **Solicita tu archivo**.  
4. Strava te enviará un correo con un archivo .zip. Descárgalo y descomprímelo.  
5. **Solo necesitas el archivo `activities.csv`** (o puedes subir el ZIP completo).
"""
    )

uploaded = st.file_uploader(
    "Sube un archivo: CSV (Strava), GPX/TCX/FIT (Garmin/Komoot/otros) o ZIP (export completo)",
    type=["csv", "gpx", "tcx", "fit", "zip"],
)

with st.sidebar:
    st.header("Modalidad")
    modality = st.selectbox("Selecciona modalidad", options=["correr", "bicicleta"])

    st.header("Perfil")
    age = st.number_input("Edad", min_value=12, max_value=90, value=36, step=1)
    height_cm = st.number_input("Altura (cm)", min_value=120, max_value=220, value=180, step=1)
    weight_kg = st.number_input("Peso (kg)", min_value=35.0, max_value=200.0, value=80.0, step=0.5)

    st.header("Objetivo")
    goal = st.selectbox("Selecciona objetivo principal", options=GOALS[modality], format_func=lambda x: x[1])[0]

    days_per_week = st.slider("Días de entrenamiento por semana", min_value=1, max_value=7, value=5)
    start_date = st.date_input("Fecha de inicio", value=date.today())

if uploaded is None:
    st.stop()

activities_df, ts_map = parse_any_upload_pro(uploaded)

file_counts = getattr(activities_df, "attrs", {}).get("file_counts", None)
if file_counts:
    with st.expander("Qué tipos de archivos he encontrado dentro del ZIP"):
        st.write(file_counts)

if activities_df.empty:
    st.error("No he podido leer actividades del archivo. Prueba con CSV de Strava, GPX/TCX/FIT o ZIP con actividades.")
    st.stop()

sport_needed = "run" if modality == "correr" else "ride"
target = activities_df[activities_df["sport"] == sport_needed].copy()

if target.empty and (activities_df["sport"] == "other").any():
    with st.expander("Clasificación de actividades sin deporte (solo si hace falta)"):
        st.write("Algunos formatos (como GPX) no indican si es correr o bici. Puedes asignarlo aquí.")
        assign_other = st.selectbox("Asignar actividades 'other' como:", options=["no cambiar", "correr", "bicicleta"], index=0)
        if assign_other != "no cambiar":
            activities_df.loc[activities_df["sport"] == "other", "sport"] = "run" if assign_other == "correr" else "ride"
            target = activities_df[activities_df["sport"] == sport_needed].copy()

if target.empty:
    st.warning("No he encontrado actividades de la modalidad seleccionada (o falta clasificación si has subido GPX).")

hrmax = estimate_hrmax(int(age))
baseline = weekly_baseline_simple(target) if not target.empty else {"w_km": 0.0, "long_km": 0.0, "sessions_w": 0.0}

st.subheader("Lectura rápida de tus datos")
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
    cols = [c for c in cols if c in show.columns]
    st.dataframe(show[cols].head(200), use_container_width=True)

st.subheader("Feedback")
has_pro_signal = len(ts_map) > 0
if st.button("Generar feedback semanal"):
    if has_pro_signal:
        st.markdown(make_weekly_feedback_pro(target, ts_map, modality, hrmax))
    else:
        st.markdown(make_weekly_feedback_basic(target, modality, hrmax))

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
