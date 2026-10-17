import os
import sys

# Add E:\VideoAi2 to Python path so it can easily find tiktoken and other local packages
sys.path.append(r"E:\VideoAi2")

os.environ["HF_HOME"] = r"E:\hf_cache"   # model cache on E drive

import gc
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

FPS = 24
RATIOS = {
    "16:9 (landscape)": (16, 9),
    "9:16 (portrait)": (9, 16),
    "1:1 (square)": (1, 1),
    "4:3": (4, 3),
    "3:4": (3, 4),
}
SIZES = ["384", "448", "512", "640", "704"]

pipe = None
q = queue.Queue()


def round32(x):
    return max(32, int(round(x / 32)) * 32)


def calc_size(ratio, long_side):
    rw, rh = ratio
    if rw >= rh:
        w = long_side
        h = long_side * rh / rw
    else:
        h = long_side
        w = long_side * rw / rh
    return round32(w), round32(h)


def calc_frames(seconds):
    n = max(1, round((seconds * FPS - 1) / 8))
    return 8 * n + 1          


def translate(text):
    from deep_translator import GoogleTranslator, MyMemoryTranslator
    makers = (
        ("Google", lambda: GoogleTranslator(source="auto", target="en")),
        ("MyMemory", lambda: MyMemoryTranslator(source="autodetect", target="en-GB")),
    )
    for name, make in makers:
        for _ in range(2):
            try:
                res = make().translate(text)
                if res:
                    return res
            except Exception as e:
                q.put(("log", f"Translation ({name}) failed: {str(e)[:80]}"))
                time.sleep(2)
    q.put(("log", "Using prompt as is, quality might be affected"))
    return text


def load_pipe():
    global pipe
    if pipe is not None:
        return pipe

    import torch
    from transformers import T5EncoderModel
    from transformers import BitsAndBytesConfig as TfBnB
    from diffusers import LTXPipeline, LTXVideoTransformer3DModel
    from diffusers import BitsAndBytesConfig as DiffBnB

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not found. Install PyTorch with CUDA support:\n"
            "py -m pip install -U --target . torch "
            "--index-url https://download.pytorch.org/whl/cu126"
        )

    q.put(("status", "Loading compressed model (first time takes longer)..."))
    repo = "Lightricks/LTX-Video"

    text_encoder = T5EncoderModel.from_pretrained(
        repo,
        subfolder="text_encoder",
        quantization_config=TfBnB(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        ),
        torch_dtype=torch.bfloat16,
    )
    transformer = LTXVideoTransformer3DModel.from_pretrained(
        repo,
        subfolder="transformer",
        quantization_config=DiffBnB(load_in_8bit=True),
        torch_dtype=torch.bfloat16,
    )
    p = LTXPipeline.from_pretrained(
        repo,
        text_encoder=text_encoder,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )
    p.enable_model_cpu_offload()
    p.vae.enable_tiling()
    pipe = p
    return pipe


def worker(prompt_ua, ratio, long_side, seconds, steps, seed):
    try:
        import torch
        from diffusers.utils import export_to_video

        q.put(("status", "Processing prompt..."))
        prompt = translate(prompt_ua)
        q.put(("log", f"Model prompt: {prompt}"))

        p = load_pipe()
        w, h = calc_size(ratio, long_side)
        frames = calc_frames(seconds)
        q.put(("log", f"Resolution {w}x{h}, frames {frames} (~{frames / FPS:.1f} s)"))

        def on_step(pipeline, step, timestep, kwargs):
            q.put(("status", f"Generating: step {step + 1}/{steps}"))
            return kwargs

        gen = torch.Generator(device="cpu").manual_seed(seed)
        q.put(("status", "Generating..."))
        video = p(
            prompt=prompt,
            negative_prompt="worst quality, blurry, jittery, distorted",
            width=w,
            height=h,
            num_frames=frames,
            num_inference_steps=steps,
            generator=gen,
            callback_on_step_end=on_step,
        ).frames[0]

        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f"video_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
        export_to_video(video, out, fps=FPS)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        q.put(("done", out))
    except Exception as e:
        msg = str(e)
        if "out of memory" in msg.lower():
            msg = ("Out of memory. Decrease resolution, seconds or steps, "
                   "close your browser and try again.")
        q.put(("error", msg))


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Video Generator (Low RAM)")
        self.geometry("620x640")
        self.minsize(560, 600)
        pad = {"padx": 10, "pady": 4}

        ttk.Label(self, text="Prompt (English recommended):").pack(anchor="w", **pad)
        self.prompt = tk.Text(self, height=6, wrap="word", font=("Segoe UI", 11))
        self.prompt.pack(fill="x", **pad)
        self.prompt.insert("1.0", "A girl in a red coat walking through a snowy forest at dawn, "
                                  "soft golden light, cinematic")
        self.prompt.bind("<Control-KeyPress>", self.on_ctrl_key)

        btns = ttk.Frame(self)
        btns.pack(fill="x", padx=10, pady=2)
        ttk.Button(btns, text="Paste from Clipboard",
                   command=self.paste_prompt).pack(side="left")
        ttk.Button(btns, text="Clear",
                   command=self.clear_prompt).pack(side="left", padx=6)

        grid = ttk.Frame(self)
        grid.pack(fill="x", **pad)

        ttk.Label(grid, text="Aspect Ratio:").grid(row=0, column=0, sticky="w", pady=4)
        self.ratio = ttk.Combobox(grid, values=list(RATIOS), state="readonly", width=24)
        self.ratio.current(1)
        self.ratio.grid(row=0, column=1, sticky="w", padx=8)

        ttk.Label(grid, text="Seconds:").grid(row=1, column=0, sticky="w", pady=4)
        self.seconds = tk.Spinbox(grid, from_=1, to=10, increment=1, width=6)
        self.seconds.delete(0, "end")
        self.seconds.insert(0, "1")
        self.seconds.grid(row=1, column=1, sticky="w", padx=8)

        ttk.Label(grid, text="Resolution (Longest side):").grid(row=2, column=0, sticky="w", pady=4)
        self.size = ttk.Combobox(grid, values=SIZES, state="readonly", width=8)
        self.size.current(0)
        self.size.grid(row=2, column=1, sticky="w", padx=8)

        ttk.Label(grid, text="Steps (Quality):").grid(row=3, column=0, sticky="w", pady=4)
        self.steps = tk.Spinbox(grid, from_=10, to=50, increment=5, width=6)
        self.steps.delete(0, "end")
        self.steps.insert(0, "30")
        self.steps.grid(row=3, column=1, sticky="w", padx=8)

        ttk.Label(grid, text="Seed (empty = random):").grid(row=4, column=0, sticky="w", pady=4)
        self.seed = ttk.Entry(grid, width=10)
        self.seed.grid(row=4, column=1, sticky="w", padx=8)

        self.btn = ttk.Button(self, text="Generate", command=self.start)
        self.btn.pack(fill="x", padx=10, pady=8)

        self.bar = ttk.Progressbar(self, mode="indeterminate")
        self.bar.pack(fill="x", **pad)
        self.status = ttk.Label(self, text="Ready")
        self.status.pack(anchor="w", **pad)

        self.log = tk.Text(self, height=9, wrap="word", state="disabled",
                           font=("Consolas", 9))
        self.log.pack(fill="both", expand=True, **pad)

        self.open_btn = ttk.Button(self, text="Open Video Folder",
                                   command=self.open_folder)
        self.open_btn.pack(fill="x", padx=10, pady=(0, 10))

        self.after(200, self.poll)

    def paste_prompt(self):
        try:
            text = self.clipboard_get()
        except tk.TclError:
            messagebox.showinfo("Clipboard Empty", "Copy some text first.")
            return
        self.prompt.delete("1.0", "end")
        self.prompt.insert("1.0", text.strip())

    def clear_prompt(self):
        self.prompt.delete("1.0", "end")
        self.prompt.focus_set()

    def on_ctrl_key(self, event):
        if event.keycode == 86:
            try:
                text = self.clipboard_get()
            except tk.TclError:
                return "break"
            try:
                self.prompt.delete("sel.first", "sel.last")
            except tk.TclError:
                pass
            self.prompt.insert("insert", text)
            return "break"
        if event.keycode == 67:
            try:
                self.clipboard_clear()
                self.clipboard_append(self.prompt.get("sel.first", "sel.last"))
            except tk.TclError:
                pass
            return "break"
        if event.keycode == 65:
            self.prompt.tag_add("sel", "1.0", "end-1c")
            return "break"

    def write(self, text):
        self.log.config(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def start(self):
        text = self.prompt.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter a prompt.")
            return
        try:
            seconds = int(self.seconds.get())
            steps = int(self.steps.get())
            seed_txt = self.seed.get().strip()
            seed = int(seed_txt) if seed_txt else int(time.time()) % 100000
        except ValueError:
            messagebox.showwarning("Warning", "Seconds, steps and seed must be numbers.")
            return

        ratio = RATIOS[self.ratio.get()]
        long_side = int(self.size.get())
        self.btn.config(state="disabled")
        self.bar.start(12)
        self.write("--- Start ---")
        threading.Thread(
            target=worker,
            args=(text, ratio, long_side, seconds, steps, seed),
            daemon=True,
        ).start()

    def poll(self):
        try:
            while True:
                kind, data = q.get_nowait()
                if kind == "status":
                    self.status.config(text=data)
                elif kind == "log":
                    self.write(data)
                elif kind == "done":
                    self.finish()
                    self.status.config(text="Ready!")
                    self.write(f"Saved: {data}")
                    messagebox.showinfo("Done", f"Video saved to:\n{data}")
                    try:
                        os.startfile(data)
                    except Exception:
                        pass
                elif kind == "error":
                    self.finish()
                    self.status.config(text="Error")
                    self.write(f"ERROR: {data}")
                    messagebox.showerror("Error", data)
        except queue.Empty:
            pass
        self.after(200, self.poll)

    def finish(self):
        self.bar.stop()
        self.btn.config(state="normal")

    def open_folder(self):
        folder = os.path.dirname(os.path.abspath(__file__))
        os.startfile(folder)


if __name__ == "__main__":
    App().mainloop()