import time
import base64
import threading
import subprocess
from pathlib import Path

import cv2
import numpy as np
import mss
import tkinter as tk
from tkinter.scrolledtext import ScrolledText

from openai import OpenAI


MODEL = "gpt-4.1-mini"
client = OpenAI()

ROOT_DIR = Path(__file__).resolve().parent
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

ML_DIR = ROOT_DIR / "ekg_ml"
ML_PYTHON = ML_DIR / ".venv" / "Scripts" / "python.exe"
ML_PREDICT_SCRIPT = ML_DIR / "predict_from_image.py"
ML_MODEL_PATH = ML_DIR / "models" / "baseline_rf.joblib"


def ensure_logs():
    LOG_DIR.mkdir(exist_ok=True)


def save_artifacts(frame_bgr, prompt_text, answer_text):
    ensure_logs()
    ts = time.strftime("%Y%m%d_%H%M%S")
    img_path = LOG_DIR / f"frame_{ts}.png"
    prompt_path = LOG_DIR / f"prompt_{ts}.txt"
    resp_path = LOG_DIR / f"response_{ts}.txt"

    cv2.imwrite(str(img_path), frame_bgr)
    prompt_path.write_text(prompt_text, encoding="utf-8")
    resp_path.write_text(answer_text, encoding="utf-8")
    return img_path, resp_path


def save_latest_frame(frame_bgr):
    """
    Always save a predictable filename for the latest snapshot,
    plus a timestamped copy for history.
    """
    ensure_logs()
    latest_path = LOG_DIR / "frame_latest.png"
    ts_path = LOG_DIR / f"frame_{time.strftime('%Y%m%d_%H%M%S')}.png"

    cv2.imwrite(str(latest_path), frame_bgr)
    cv2.imwrite(str(ts_path), frame_bgr)

    return latest_path, ts_path


def default_ekg_prompt():
    return (
        "You are a clinical assistant. Analyze the EKG rhythm strip visible on the screen image.\n"
        "Return a structured response:\n"
        "1) Most likely rhythm (one line)\n"
        "2) Supporting features (3-6 bullets: rate, regularity, P waves, PR, QRS width, ectopy, etc.)\n"
        "3) Key differentials (1-3) and how to distinguish each (bullets)\n"
        "4) Confidence: low/medium/high + brief reason\n"
        "5) Safety note: decision-support only; clinician must confirm.\n"
        "\n"
        "If the image does not clearly show an EKG rhythm strip, say what is missing and what to do next."
    )


def ask_ai_with_image(frame_bgr, user_question):
    ok, buf = cv2.imencode(".png", frame_bgr)
    if not ok:
        raise RuntimeError("Failed to encode frame to PNG.")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

    if not user_question.strip():
        prompt_text = default_ekg_prompt()
    else:
        prompt_text = (
            "You are a clinical assistant. Answer the user's question about what is visible in the screen image.\n"
            "Be clinically structured and include confidence.\n"
            "Decision-support only; clinician must confirm.\n\n"
            f"User question: {user_question.strip()}"
        )

    resp = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt_text},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"},
                ],
            }
        ],
    )
    return prompt_text, resp.output_text


def grab_screen_frame(sct, monitor):
    shot = sct.grab(monitor)
    frame = np.array(shot)
    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


def grab_clean_snapshot(sct, monitor):
    time.sleep(0.05)
    return grab_screen_frame(sct, monitor)


def run_local_ml_on_image(image_path: Path, topk: int = 3):
    """
    Calls ekg_ml/predict_from_image.py using the ekg_ml virtual env interpreter.

    Returns:
      (ok: bool, text: str)
    """
    if not ML_PYTHON.exists():
        return (
            False,
            f"Local ML not ready: missing ML venv python at: {ML_PYTHON}\n"
            "Open terminal in ekg_ml and run: .\\.venv\\Scripts\\Activate.ps1"
        )

    if not ML_PREDICT_SCRIPT.exists():
        return (False, f"Local ML not ready: missing script: {ML_PREDICT_SCRIPT}")

    if not ML_MODEL_PATH.exists():
        return (False, f"Local ML not ready: missing model: {ML_MODEL_PATH}\nRun: python .\\ekg_ml\\train_baseline.py")

    if not image_path.exists():
        return (False, f"Local ML error: missing image: {image_path}")

    cmd = [
        str(ML_PYTHON),
        str(ML_PREDICT_SCRIPT),
        "--image",
        str(image_path),
        "--model",
        str(ML_MODEL_PATH),
        "--topk",
        str(int(topk)),
    ]

    try:
        p = subprocess.run(
            cmd,
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as e:
        return (False, f"Local ML call failed: {e}")

    out = (p.stdout or "").strip()
    err = (p.stderr or "").strip()

    if p.returncode != 0:
        msg = "Local ML error (predict_from_image.py):\n"
        if err:
            msg += err
        else:
            msg += out if out else "Unknown error"
        return (False, msg)

    # Parse the "Top predictions:" block if present
    lines = out.splitlines()
    preds = []
    capture = False
    for line in lines:
        if line.strip().lower().startswith("top predictions"):
            capture = True
            continue
        if capture:
            s = line.strip()
            if not s:
                continue
            if s.startswith("Image:") or s.startswith("Model:"):
                continue
            if s[0].isdigit():
                preds.append(s)

    if preds:
        pretty = "Local ML (baseline RF):\n" + "\n".join([f"- {x}" for x in preds])
        return (True, pretty)

    # fallback: show raw output
    return (True, f"Local ML output:\n{out}")


class ChatUI:
    def __init__(self, root):
        self.root = root
        self.root.title("EKG AI Assistant (Griffin Health PLLC)")
        self.root.geometry("900x650")

        self.snapshot_label = tk.Label(root, text="No snapshot yet. Press A in the capture loop.", anchor="w")
        self.snapshot_label.pack(fill="x", padx=10, pady=(10, 5))

        self.chat = ScrolledText(root, wrap="word", height=26)
        self.chat.pack(fill="both", expand=True, padx=10, pady=5)
        self.chat.insert("end", "Ready. Press A to capture a clean snapshot, then ask a question here.\n\n")
        self.chat.configure(state="disabled")

        bottom = tk.Frame(root)
        bottom.pack(fill="x", padx=10, pady=(5, 10))

        self.entry = tk.Entry(bottom)
        self.entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.entry.bind("<Return>", lambda e: self.on_send())

        self.send_btn = tk.Button(bottom, text="Send", command=self.on_send)
        self.send_btn.pack(side="left")

        self.default_btn = tk.Button(bottom, text="Default EKG Read", command=self.on_default)
        self.default_btn.pack(side="left", padx=(8, 0))

        self.status = tk.Label(root, text="Status: idle", anchor="w")
        self.status.pack(fill="x", padx=10, pady=(0, 10))

        self.latest_frame = None
        self.latest_frame_path = None

    def set_status(self, text):
        self.status.config(text=f"Status: {text}")
        self.root.update_idletasks()

    def append_chat(self, who, text):
        self.chat.configure(state="normal")
        self.chat.insert("end", f"{who}: {text}\n\n")
        self.chat.see("end")
        self.chat.configure(state="disabled")

    def set_snapshot(self, frame_bgr):
        self.latest_frame = frame_bgr

        latest_path, ts_path = save_latest_frame(frame_bgr)
        self.latest_frame_path = latest_path

        ts = time.strftime("%H:%M:%S")
        self.snapshot_label.config(
            text=f"Snapshot captured at {ts}. Saved: {latest_path.name} (and {ts_path.name})."
        )

        # Run local ML right away after capture, in background thread
        def ml_worker():
            self.set_status("running local ML...")
            ok, ml_text = run_local_ml_on_image(latest_path, topk=3)
            if ok:
                self.append_chat("Local ML", ml_text)
            else:
                self.append_chat("Local ML", ml_text)
            self.set_status("idle")

        threading.Thread(target=ml_worker, daemon=True).start()

    def on_default(self):
        self.entry.delete(0, "end")
        self.entry.insert(0, "")
        self.on_send()

    def on_send(self):
        if self.latest_frame is None:
            self.append_chat("System", "No snapshot yet. Press A (in the capture window loop) to capture one.")
            return

        question = self.entry.get()
        self.entry.delete(0, "end")

        if question.strip():
            self.append_chat("You", question.strip())
        else:
            self.append_chat("You", "[Default EKG Read]")

        def worker():
            try:
                self.set_status("sending to AI...")
                prompt_used, answer = ask_ai_with_image(self.latest_frame, question)
                img_path, resp_path = save_artifacts(self.latest_frame, prompt_used, answer)
                self.append_chat("AI", answer)
                self.append_chat("System", f"Saved logs: {img_path.name}, {resp_path.name}")
                self.set_status("idle")
            except Exception as e:
                self.append_chat("System", f"AI ERROR: {e}")
                self.set_status("error")

        threading.Thread(target=worker, daemon=True).start()


def main():
    print("MODE = ON-DEMAND SCREEN SNAPSHOT + TKINTER CHAT + LOCAL-ML BRIDGE")
    print("Controls: click the small control window, then press A to capture, Q to quit.")
    print(f"Root: {ROOT_DIR}")
    print(f"Logs: {LOG_DIR}")
    print(f"ML python: {ML_PYTHON}")

    root = tk.Tk()
    ui = ChatUI(root)

    cv2.namedWindow("CAPTURE CONTROL (click here, then press A/Q)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CAPTURE CONTROL (click here, then press A/Q)", 520, 90)
    control_canvas = np.zeros((90, 520, 3), dtype=np.uint8)
    cv2.putText(
        control_canvas,
        "Click here, then press A=capture  Q=quit",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    with mss.mss() as sct:
        monitor = sct.monitors[1]

        def poll_keys():
            cv2.imshow("CAPTURE CONTROL (click here, then press A/Q)", control_canvas)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                cv2.destroyAllWindows()
                root.destroy()
                return

            if key in (ord("a"), ord("A")):
                ui.set_status("capturing snapshot...")
                frame = grab_clean_snapshot(sct, monitor)
                ui.set_snapshot(frame)
                ui.set_status("idle")

            root.after(10, poll_keys)

        root.after(10, poll_keys)
        root.mainloop()

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    print("Done.")


if __name__ == "__main__":
    main()
