"""
3D Printing Farm Dashboard (Streamlit)
--------------------------------------
This file implements the MVP dashboard for managing a farm of 3D printers.

Features:
- Live progress tied to 5 g/s material consumption.
- Spool tracking with low-spool and out-of-spool failure detection.
- Attended workflow to resolve Completed/Failed/Stopped printers.
- Start job flow with options (Cannot Start / Force Start / Attended).
- CSV persistence for live status and append-only event logs.
- Simulated AI failure detection using camera images and a trained model.

Author: Jason Menard Vasallo
Date: 2025-10-03
For: Ocean Builders Technical Evaluation
"""

from __future__ import annotations

import csv
import os
import threading
import time
import uuid
import pandas as pd
import streamlit as st
import random
import sys
from datetime import datetime, timedelta
from math import ceil
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from inference.inference_failure_model import load_model, predict

IMAGE_DIR = "data/images"                           # Images of simulated images

# -------------------------------------------------------------------
# AI Failure Detection Model
# -------------------------------------------------------------------
# Loads the trained PyTorch model at startup.
# Used in the UI to classify simulated camera feed images as "good" or "bad".
# Failure is triggered automatically if probability(bad) > threshold.
AI_MODEL = load_model("outputs/best_model.pth")

# Constants, Files, and Configs
LIVE_FILE = "printers_live.csv"
EVENTS_FILE = "events_history.csv"

TOTAL_PRINTERS = 60
SPOOL_CAPACITY_G = 1000
PRINT_RATE_G_PER_S = 5
REFRESH_RATE_S = 1

LOW_MATERIAL_THRESHOLD_G = 150  # UI notifier only

BAD_PRINT_THRESHOLD = 0.35  # Probability cutoff for auto-failure trigger

# NOTE: All CSV reads/writes are wrapped in _FILE_LOCK to avoid race conditions
# when simulation loop and UI interactions happen concurrently.
_FILE_LOCK = threading.Lock()

FIELDNAMES = [
    # user fields (blank if Idle)
    "part_serial_no",
    "part_description",
    "part_name",
    "batch_no",
    "estimated_grams",
    "operator",
    # system fields
    "status",  # Idle, Printing, Completed, Failed, Stopped
    "printer",
    "start_time",
    "end_time",
    "estimated_min_time",        # total estimate in minutes (float)
    "estimated_end_time",        # ISO timestamp
    "estimated_remaining_sec",   # live remaining seconds (int)
    "remaining_material_g",
    "spool_id",
    "job_id",
    "progress_percent",          # 0-100
    "fail_reason",
]

# Helpers: Time & Formatting
def _now_iso() -> str:
    """Return current timestamp in ISO 8601 format (string)."""
    return datetime.now().isoformat()

def _parse_time(ts: str | float | int) -> datetime | None:
    """Format seconds into H:MM:SS or M:SS for display in UI captions."""
    if not ts or str(ts).strip() == "":
        return None
    try:
        return datetime.fromisoformat(str(ts))
    except Exception:
        return None

def _fmt_hms(seconds: int | float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def _safe_int(x, default=0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default

def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _get_random_image() -> str:
    files = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)
             if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not files:
        return ""
    return random.choice(files)


# I/O Layer
def init_csv() -> None:
    """
    Initialize CSV files for live printer state and events log.
    Creates with default Idle rows if they do not exist.
    """
    if not os.path.exists(LIVE_FILE):
        with _FILE_LOCK, open(LIVE_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            for i in range(1, TOTAL_PRINTERS + 1):
                writer.writerow({
                    "part_serial_no": "",
                    "part_description": "",
                    "part_name": "",
                    "batch_no": "",
                    "estimated_grams": "",
                    "operator": "",
                    "status": "Idle",
                    "printer": f"P{i:02d}",
                    "start_time": "",
                    "end_time": "",
                    "estimated_min_time": "",
                    "estimated_end_time": "",
                    "estimated_remaining_sec": "",
                    "remaining_material_g": SPOOL_CAPACITY_G,
                    "spool_id": f"S{i:03d}",
                    "job_id": "",
                    "progress_percent": 0,
                    "fail_reason": "",
                })

    if not os.path.exists(EVENTS_FILE):
        with _FILE_LOCK, open(EVENTS_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "event"])


def read_live() -> List[Dict]:
    """
    Read printer states from LIVE_FILE as a list of dicts.
    Returns [] if no file or empty file exists.
    """
    if not os.path.exists(LIVE_FILE) or os.path.getsize(LIVE_FILE) == 0:
        return []
    try:
        with _FILE_LOCK:
            df = pd.read_csv(LIVE_FILE)
    except pd.errors.EmptyDataError:
        return []

    df = df.fillna("")  # leave blanks for user/system empty fields
    
    # cast a few numeric fields safely
    for col in ["estimated_grams", "remaining_material_g", "progress_percent",
                "estimated_remaining_sec"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df.to_dict(orient="records")

def write_live(rows: List[Dict]) -> None:
    df = pd.DataFrame(rows, columns=FIELDNAMES)
    with _FILE_LOCK:
        df.to_csv(LIVE_FILE, index=False)

# All printer lifecycle changes and system actions MUST be logged
# for traceability (auditing and debugging).
def log_event(event: str) -> None:
    with _FILE_LOCK, open(EVENTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([_now_iso(), event])


# Core Logic
def _recompute_time_fields(row: Dict) -> None:
    """
    Update estimated_end_time and estimated_remaining_sec based on
    PRINT_RATE_G_PER_S, progress, and start_time.
    """
    if row["status"] != "Printing":
        return

    est_g = _safe_float(row.get("estimated_grams"), 0.0)
    if est_g <= 0:
        return

    # Remaining grams of the JOB (not the spool)
    progress = _safe_float(row.get("progress_percent"), 0.0)
    grams_remaining_job = max(0.0, est_g * (1.0 - progress / 100.0))

    # Seconds remaining given fixed 5 g/s
    sec_remaining = ceil(grams_remaining_job / PRINT_RATE_G_PER_S)

    start_dt = _parse_time(row.get("start_time", ""))
    if start_dt is None:
        start_dt = datetime.now()

    # For a resumed job, ETA is now + remaining
    eta_end = datetime.now() + timedelta(seconds=sec_remaining)

    row["estimated_remaining_sec"] = sec_remaining
    row["estimated_end_time"] = eta_end.isoformat()

    # Total estimated minutes (based on full job size)
    total_sec = ceil(est_g / PRINT_RATE_G_PER_S)
    row["estimated_min_time"] = round(total_sec / 60.0, 1)


def _tick_printing_row(row: Dict) -> None:      # Edge case: if estimated_grams <= 0, fail immediately (invalid job config).
    """
    Advance a printer job by one simulation tick (1s).
    - Updates progress% and spool usage.
    - Marks job as Failed if spool empty.
    - Marks job as Completed if progress >= 100%.
    - Otherwise updates ETA fields.
    """
    est_g = _safe_float(row.get("estimated_grams"), 0.0)
    if est_g <= 0:
        # Nothing to do; fail it as 'other' to avoid division by zero
        row["status"] = "Failed"
        row["end_time"] = _now_iso()
        row["fail_reason"] = "other"
        log_event(f"FAILURE(other): Job {row.get('job_id','')} on "
                  f"{row.get('printer','')}")
        return

    # Progress increment (percent) based on fixed grams/sec
    delta_pct = (PRINT_RATE_G_PER_S / est_g) * 100.0

    progress = _safe_float(row.get("progress_percent"), 0.0)
    progress = min(100.0, progress + delta_pct)
    row["progress_percent"] = round(progress, 3)  # keep it smooth

    # Spool decrement
    remain_spool = _safe_float(row.get("remaining_material_g"), 0.0)
    remain_spool -= PRINT_RATE_G_PER_S
    row["remaining_material_g"] = max(0.0, round(remain_spool, 3))

    # Check out-of-spool first
    if row["remaining_material_g"] <= 0:
        row["status"] = "Failed"
        row["end_time"] = _now_iso()
        row["fail_reason"] = "out_of_spool"
        log_event(f"FAILURE(out_of_spool): Job {row.get('job_id','')} on "
                  f"{row.get('printer','')}")
        return

    # Completion
    if row["progress_percent"] >= 100.0:
        row["status"] = "Completed"
        row["end_time"] = _now_iso()
        row["estimated_remaining_sec"] = 0
        log_event(f"COMPLETED: Job {row.get('job_id','')} on "
                  f"{row.get('printer','')}")
        return

    # Otherwise recompute ETA/remaining
    _recompute_time_fields(row)


def _start_job(
    rows: List[Dict],
    pid: str,
    serial: str,
    desc: str,
    name: str,
    batch: str,
    grams: float,
    operator: str,
) -> None:
    for r in rows:
        if r["printer"] == pid:
            now = datetime.now()
            total_sec = ceil(grams / PRINT_RATE_G_PER_S)
            r.update({
                "part_serial_no": serial,
                "part_description": desc,
                "part_name": name,
                "batch_no": batch,
                "estimated_grams": grams,
                "operator": operator,
                "status": "Printing",
                "start_time": now.isoformat(),
                "end_time": "",
                "estimated_min_time": round(total_sec / 60.0, 1),
                "estimated_end_time": (now + timedelta(seconds=total_sec)).isoformat(),
                "estimated_remaining_sec": total_sec,
                "job_id": f"JOB-{uuid.uuid4().hex[:6].upper()}",
                "progress_percent": 0,
                "fail_reason": "",
            })
            log_event(
                f"STARTED: {r['job_id']} on {pid} by {operator} "
                f"(G={grams}g, spool={r.get('remaining_material_g','')}g)"
            )
            break


def _reset_to_idle(row: Dict, clear_job: bool = True) -> None:
    """
    Reset a printer to Idle.
    If clear_job=True, clears job metadata fields as well.
    """
    if clear_job:
        for field in [
            "part_serial_no",
            "part_description",
            "part_name",
            "batch_no",
            "estimated_grams",
            "operator",
            "start_time",
            "end_time",
            "estimated_min_time",
            "estimated_end_time",
            "estimated_remaining_sec",
            "job_id",
            "progress_percent",
            "fail_reason",
        ]:
            row[field] = ""
        row["progress_percent"] = 0
    row["status"] = "Idle"


def _replace_spool(row: Dict) -> None:
    row["remaining_material_g"] = SPOOL_CAPACITY_G
    log_event(f"SPOOL_REPLACED: {row.get('printer','')} -> {SPOOL_CAPACITY_G}g")

# Background Simulation Loop
def simulation_loop() -> None:
    """
    Background thread loop:
    - Reads printer states.
    - Applies one tick of printing simulation to active jobs.
    - Writes updates back to LIVE_FILE every REFRESH_RATE_S.
    """
    while True:
        rows = read_live()
        if not rows:
            time.sleep(REFRESH_RATE_S)
            continue

        changed = False
        for r in rows:
            if r.get("status") == "Printing":
                _tick_printing_row(r)
                changed = True

        if changed:
            write_live(rows)

        time.sleep(REFRESH_RATE_S)

# Streamlit App
def _status_badge(status: str) -> str:
    status = (status or "").strip()
    if status == "Printing":
        return ":blue[**Printing**]"
    if status == "Completed":
        return ":green[**Completed**]"
    if status == "Failed":
        return ":red[**Failed**]"
    if status == "Stopped":
        return ":orange[**Stopped**]"
    return ":gray[*Idle*]"


def _low_spool_tag(row: Dict) -> str:
    # Low if below threshold OR insufficient for the remaining job
    remain_spool = _safe_float(row.get("remaining_material_g"), 0.0)
    est_g = _safe_float(row.get("estimated_grams"), 0.0)
    progress = _safe_float(row.get("progress_percent"), 0.0)
    grams_needed = max(0.0, est_g * (1.0 - progress / 100.0))

    if remain_spool <= 0:
        return ":red[**needs new spool**]"
    if remain_spool < grams_needed or remain_spool < LOW_MATERIAL_THRESHOLD_G:
        return ":red[low on spool]"
    return ""


def _render_printer_card(row: Dict, idx: int) -> None:
    """
    Render a single printer card in the UI with three columns:
      - Left: Printer ID, Status Badge, Spool remaining.
      - Middle: Progress bar, ETA, Remaining time.
      - Right: Camera feed + AI inference, Attended resolution flow.
    """
    col1, col2, col3 = st.columns([1, 2, 2], vertical_alignment="center")

    # Clear session state completely when Idle
    if row.get("status") == "Idle":
        st.session_state.pop(f"attend_open_{idx}", None)
        st.session_state.pop(f"last_img_{idx}", None)
        st.session_state.pop(f"last_cam_refresh_{idx}", None)


    # Left Column: Printer + Spool
    with col1:
        st.markdown(f"**{row['printer']}**  \n{_status_badge(row.get('status',''))}")
        mat = _safe_float(row.get("remaining_material_g"), 0.0)
        low_tag = _low_spool_tag(row)
        if low_tag:
            st.markdown(f"**Spool:** :red[{mat:.0f} g] {low_tag}")
        else:
            st.write(f"Spool: {mat:.0f} g")

    # Middle Column: Progress + ETA
    with col2:
        progress = _safe_float(row.get("progress_percent"), 0.0)
        st.progress(min(100, max(0, int(progress))), text=f"{progress:.1f}%")

        start_dt = _parse_time(row.get("start_time", ""))
        eta_dt = _parse_time(row.get("estimated_end_time", ""))
        remain_sec = _safe_int(row.get("estimated_remaining_sec"), 0)

        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S") if start_dt else "—"
        eta_str = eta_dt.strftime("%Y-%m-%d %H:%M:%S") if eta_dt else "—"
        remain_str = _fmt_hms(remain_sec)

        st.caption(
            f"Start: {start_str} • ETA End: {eta_str} • Remaining: {remain_str}"
        )

    # Right Column: Camera Feed + Inference
    with col3:
        if row.get("status") in {"Printing", "Failed"}:
            key_last = f"last_cam_refresh_{idx}"
            now = time.time()
            last_refresh = st.session_state.get(key_last, 0)

            # Only refresh new image every 5s while Printing
            if row.get("status") == "Printing" and now - last_refresh >= 5:
                img_path = _get_random_image()
                if img_path:
                    pred_class, prob_good, prob_bad = predict(img_path, AI_MODEL)
                    st.session_state[f"last_img_{idx}"] = (img_path, prob_good, prob_bad)
                    st.session_state[key_last] = now

                    # Fail immediately if bad AND failures are allowed
                    if prob_bad > BAD_PRINT_THRESHOLD and row["status"] != "Failed" and st.session_state["allow_bad_fail"]:
                        row["status"] = "Failed"
                        row["end_time"] = _now_iso()
                        row["fail_reason"] = "bad_print"
                        write_live(_update_and_get_rows(row))
                        log_event(
                            f"FAILURE(bad_print): Job {row.get('job_id','')} on {row.get('printer','')}"
                        )

            # Always display last known image until printer goes Idle
            if f"last_img_{idx}" in st.session_state:
                img_path, prob_good, prob_bad = st.session_state[f"last_img_{idx}"]
                st.image(img_path, width=200, caption="Camera Feed")
                st.markdown(f"**Good:** {prob_good*100:.1f}% • **Bad:** {prob_bad*100:.1f}%")

        else:
            st.write("No Camera Feed - No Print Job")

        # Clear all notifications/images if printer is Idle
        if row.get("status") == "Idle":
            st.session_state.pop(f"attend_open_{idx}", None)
            st.session_state.pop(f"last_img_{idx}", None)
            st.session_state.pop(f"last_cam_refresh_{idx}", None)
            return  # don't render resolve section for idle printers


        # Attended button (enabled only for Completed/Failed/Stopped)
        eligible = row.get("status") in {"Completed", "Failed", "Stopped"}
        key_attend = f"attend_{idx}"
        if not eligible:
            st.button("Attended", key=key_attend, disabled=True)
        else:
            if st.button("Attended", key=key_attend):
                st.session_state[f"attend_open_{idx}"] = True

        # Attended flow (simple inline controls)
        if st.session_state.get(f"attend_open_{idx}", False):
            reason = (row.get("fail_reason") or "").strip()
            status = row.get("status")

            if reason == "bad_print":
                with st.container(border=True):
                    st.error("Resolve this printer (Bad Print Detected):")
                    if status == "Failed":
                        if st.button("Go Idle", key=f"failed_idle_{idx}"):
                            _reset_to_idle(row, clear_job=True)
                            write_live(_update_and_get_rows(row))
                            log_event(f"ATTENDED(close): {row['printer']} Failed -> Idle")

                            # Force clear UI state
                            st.session_state.pop(f"attend_open_{idx}", None)
                            st.session_state.pop(f"last_img_{idx}", None)
                            st.session_state.pop(f"last_cam_refresh_{idx}", None)

                            st.rerun()


            else:
                with st.container(border=True):
                    st.info("Resolve this printer:")
                    if status == "Completed":
                        if st.button("Close & Go Idle", key=f"close_idle_{idx}"):
                            _reset_to_idle(row, clear_job=True)
                            write_live(_update_and_get_rows(row))
                            log_event(f"ATTENDED(close): {row['printer']} Completed -> Idle")
                            st.session_state[f"attend_open_{idx}"] = False
                            st.session_state.pop(f"last_img_{idx}", None)
                            st.rerun()

                    elif status == "Stopped":
                        if st.button("Close & Go Idle", key=f"stopped_idle_{idx}"):
                            _reset_to_idle(row, clear_job=True)
                            write_live(_update_and_get_rows(row))
                            log_event(f"ATTENDED(close): {row['printer']} Stopped -> Idle")
                            st.session_state[f"attend_open_{idx}"] = False
                            st.session_state.pop(f"last_img_{idx}", None)
                            st.rerun()

                    elif status == "Failed":
                        if reason == "out_of_spool":
                            st.info("Failure reason: out of spool")
                            col_a, col_b = st.columns(2)
                            with col_a:
                                if st.button("Replace Spool & Continue", key=f"cont_{idx}"):
                                    _replace_spool(row)
                                    row["status"] = "Printing"
                                    row["end_time"] = ""
                                    _recompute_time_fields(row)
                                    write_live(_update_and_get_rows(row))
                                    log_event(
                                        f"ATTENDED(replace_spool+continue): "
                                        f"{row['printer']} resume {row.get('job_id','')}"
                                    )
                                    st.session_state[f"attend_open_{idx}"] = False
                                    st.rerun()
                            with col_b:
                                if st.button("Replace Spool & Go Idle", key=f"idle_{idx}"):
                                    _replace_spool(row)
                                    _reset_to_idle(row, clear_job=True)
                                    write_live(_update_and_get_rows(row))
                                    log_event(
                                        f"ATTENDED(replace_spool+idle): {row['printer']}"
                                    )
                                    st.session_state[f"attend_open_{idx}"] = False
                                    st.session_state.pop(f"last_img_{idx}", None)
                                    st.rerun()
                        else:
                            st.info(f"Failure reason: {reason or 'manual'}")
                            if st.button("Go Idle", key=f"failed_idle_{idx}"):
                                _reset_to_idle(row, clear_job=True)
                                write_live(_update_and_get_rows(row))
                                log_event(f"ATTENDED(close): {row['printer']} Failed -> Idle")

                                # Force clear UI state 
                                st.session_state.pop(f"attend_open_{idx}", None)
                                st.session_state.pop(f"last_img_{idx}", None)
                                st.session_state.pop(f"last_cam_refresh_{idx}", None)

                                st.rerun()



def _update_and_get_rows(updated_row: Dict) -> List[Dict]:
    """
    Persist the updated row back into the full dataset and return the new list.
    """
    rows = read_live()
    for i, r in enumerate(rows):
        if r["printer"] == updated_row["printer"]:
            rows[i] = updated_row
            break
    return rows

# --------------------------- Main ---------------------------
def main() -> None:
    """
    Entry point for the Streamlit dashboard.

    Tabs:
      - Printers: Live status view of all printers.
      - Start Job: Form to start new print jobs.
      - Actions: Admin actions (fail, stop, reset).
      - Events: Event history log.
    """
    st.set_page_config(page_title="3D Printing Farm Dashboard", layout="wide")
    st.title("3D Printing Farm Dashboard")

    # Global option: allow AI-triggered bad print failures
    if "allow_bad_fail" not in st.session_state:
        st.session_state["allow_bad_fail"] = True

    st.sidebar.subheader("Settings")
    st.sidebar.checkbox(
        "Allow Failure Due to Bad Print",
        value=st.session_state["allow_bad_fail"],
        key="allow_bad_fail"
    )


    init_csv()
    time.sleep(0.15)  # fs flush

    # Start background loop only once
    if "sim_thread_started" not in st.session_state:
        t = threading.Thread(target=simulation_loop, daemon=True)
        t.start()
        st.session_state["sim_thread_started"] = True

    tab1, tab2, tab3, tab4 = st.tabs(["Printers", "Start Job", "Actions", "Events"])

    # -----------------------------
    # Printers Tab
    # Shows live status of all printers, including progress, spool, and camera.
    # -----------------------------
    with tab1:
        st.subheader("Printer Status")
        rows = read_live()
        if not rows:
            st.warning("No printer data available yet.")
        else:
            for idx, r in enumerate(rows):
                _recompute_time_fields(r)  # keep ETA fresh
                _render_printer_card(r, idx)

            # Persist any recomputed ETA/remaining from render pass
            write_live(rows)

    # -----------------------------
    # Start Job Tab
    # Form for creating new jobs on idle printers, with Force/Attended modes.
    # -----------------------------
    with tab2:
        st.subheader("Start New Job")
        rows = read_live()
        idle_rows = [r for r in rows if r.get("status") == "Idle"]

        if idle_rows:
            with st.form("start_job"):
                pid = st.selectbox("Printer", [r["printer"] for r in idle_rows])
                serial = st.text_input("Part Serial No")
                desc = st.text_input("Part Description")
                name = st.text_input("Part Name")
                batch = st.text_input("Batch No")
                grams = st.number_input("Estimated Grams", 1, 100000, 100)
                operator = st.text_input("Operator")

                # Check chosen printer's spool
                selected = next((r for r in rows if r["printer"] == pid), None)
                remaining_spool = _safe_float(
                    selected.get("remaining_material_g") if selected else 0, 0.0
                )

                cannot_start = grams > remaining_spool
                if cannot_start:
                    st.error(
                        "Cannot start: estimated grams exceed remaining spool."
                    )
                    col_fs, col_att = st.columns(2)
                    with col_fs:
                        force_start = st.checkbox(
                            "Force Start (may fail mid-print)", value=False
                        )
                    with col_att:
                        replace_first = st.checkbox(
                            "Attended (replace spool to 1000g first)", value=False
                        )
                else:
                    force_start = False
                    replace_first = False

                submitted = st.form_submit_button("Start Job")
                if submitted:
                    if replace_first:
                        _replace_spool(selected)
                    if cannot_start and not (force_start or replace_first):
                        st.warning("Choose Force Start or Attended to proceed.")
                    else:
                        _start_job(
                            rows, pid, serial, desc, name, batch, grams, operator
                        )
                        write_live(rows)
                        st.success(f"Job started on {pid}")
                        st.rerun()
        else:
            st.info("No idle printers available.")

    # -----------------------------
    # Actions Tab
    # Manual override actions: simulate failure, stop, reset printers.
    # -----------------------------
    with tab3:
        st.subheader("Printer Actions")
        rows = read_live()

        # Simulate failure (bad print)
        printing = [r["printer"] for r in rows if r.get("status") == "Printing"]
        if printing:
            fail_pid = st.selectbox("Simulate Failure", printing, key="fail_sel")
            if st.button("Fail Selected Printer", key="fail_btn"):
                for r in rows:
                    if r["printer"] == fail_pid:
                        r["status"] = "Failed"
                        r["end_time"] = _now_iso()
                        r["fail_reason"] = "manual"
                        log_event(
                            f"FAILURE(manual): Job {r.get('job_id','')} on {fail_pid}"
                        )
                write_live(rows)
                st.error(f"Simulated failure on {fail_pid}")
                st.rerun()

        # Stop a printing job (freeze state)
        if printing:
            stop_pid = st.selectbox("Stop Printing", printing, key="stop_sel")
            if st.button("Stop Selected Printer", key="stop_btn"):
                for r in rows:
                    if r["printer"] == stop_pid:
                        r["status"] = "Stopped"
                        r["end_time"] = _now_iso()
                        log_event(f"STOPPED: Job {r.get('job_id','')} on {stop_pid}")
                write_live(rows)
                st.warning(f"Stopped {stop_pid}")
                st.rerun()

        # Reset printer (Completed/Failed/Stopped -> Idle, resets spool to full)
        resetable = [
            r["printer"]
            for r in rows
            if r.get("status") in {"Completed", "Failed", "Stopped"}
        ]
        if resetable:
            reset_pid = st.selectbox("Full Reset (spool=1000g)", resetable, key="reset")
            if st.button("Reset Selected Printer", key="reset_btn"):
                for r in rows:
                    if r["printer"] == reset_pid:
                        _reset_to_idle(r, clear_job=True)
                        r["remaining_material_g"] = SPOOL_CAPACITY_G
                        log_event(f"RESET(full): {reset_pid} -> spool {SPOOL_CAPACITY_G}g")
                write_live(rows)
                st.success(f"Reset {reset_pid}")
                st.rerun()

    # -----------------------------
    # Events Tab
    # Displays recent system events (last 50).
    # -----------------------------
    with tab4:
        st.subheader("Events Log")
        if os.path.exists(EVENTS_FILE):
            with _FILE_LOCK:
                df = pd.read_csv(EVENTS_FILE)
            st.table(df.tail(50))
        else:
            st.info("No events yet.")

    # Auto-refresh
    time.sleep(REFRESH_RATE_S)
    st.rerun()

if __name__ == "__main__":
    main()
