#!/usr/bin/env python3
"""Local Model Inspector and Inference Workbench for gpt-scratch.

Provides a research workbench for testing the from-scratch PyTorch transformer,
streaming autoregressive generations, decomposing BPE tokens, and inspecting
reproducible training and ablation artifacts.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# Add src to sys.path so gpt package is discoverable
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))

import torch
from gpt.eval import load_checkpoint, sample_next
from gpt.train import resolve_device

MODEL = None
TOKENIZER = None
DEVICE = None
CONFIG = None


def init_model(ckpt_path: str = "artifacts/checkpoints/tiny.pt"):
    global MODEL, TOKENIZER, DEVICE, CONFIG
    DEVICE = resolve_device("auto")
    full_path = ROOT_DIR / ckpt_path
    if not full_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {full_path}")
    MODEL, TOKENIZER, _ = load_checkpoint(str(full_path), device=DEVICE)
    CONFIG = MODEL.cfg
    param_count = sum(p.numel() for p in MODEL.parameters())
    print(f"[gpt-scratch] Model loaded: {param_count:,} parameters on device '{DEVICE}'")
    print(f"[gpt-scratch] Architecture: {CONFIG.n_layer} layers, {CONFIG.n_head} heads, d_model={CONFIG.n_embd}, vocab={TOKENIZER.vocab_size}")


class WorkbenchHandler(BaseHTTPRequestHandler):
    server_version = "GPTWorkbench/1.0"

    def log_message(self, format, *args):
        # Format logging safely without index errors
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")

    def do_GET(self):
        url_path = self.path.split("?")[0]

        if url_path in ("/", "/index.html"):
            self.serve_file(ROOT_DIR / "web" / "index.html", "text/html; charset=utf-8")
        elif url_path == "/styles.css":
            self.serve_file(ROOT_DIR / "web" / "styles.css", "text/css; charset=utf-8")
        elif url_path == "/app.js":
            self.serve_file(ROOT_DIR / "web" / "app.js", "application/javascript; charset=utf-8")
        elif url_path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        elif url_path.startswith("/results/"):
            rel_path = url_path[len("/results/"):]
            file_path = ROOT_DIR / "results" / rel_path
            mime, _ = mimetypes.guess_type(str(file_path))
            self.serve_file(file_path, mime or "application/octet-stream")
        elif url_path == "/api/info":
            self.serve_api_info()
        else:
            self.send_error(404, f"File not found: {url_path}")

    def do_HEAD(self):
        url_path = self.path.split("?")[0]
        if url_path in ("/", "/index.html", "/styles.css", "/app.js", "/api/info"):
            self.send_response(200)
            self.end_headers()
        else:
            self.send_error(404)

    def do_POST(self):
        url_path = self.path.split("?")[0]
        if url_path == "/api/generate":
            self.handle_generate()
        elif url_path == "/api/tokenize":
            self.handle_tokenize()
        else:
            self.send_error(404, "Endpoint not found")

    def serve_file(self, file_path: Path, content_type: str):
        if not file_path.is_file():
            self.send_error(404, f"File not found: {file_path.name}")
            return
        try:
            content = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error reading file: {e}")

    def serve_api_info(self):
        param_count = sum(p.numel() for p in MODEL.parameters())
        dev_desc = str(DEVICE)
        if torch.cuda.is_available() and "cuda" in dev_desc:
            dev_desc = f"cuda ({torch.cuda.get_device_name(0)})"

        data = {
            "model": {
                "parameters": param_count,
                "n_layer": CONFIG.n_layer,
                "n_head": CONFIG.n_head,
                "n_embd": CONFIG.n_embd,
                "block_size": CONFIG.block_size,
                "vocab_size": TOKENIZER.vocab_size,
                "tie_weights": CONFIG.tie_weights,
                "pos_encoding": CONFIG.pos_encoding,
                "device": dev_desc,
            },
            "training": {
                "dataset": "Tiny Shakespeare",
                "training_steps": 3000,
                "final_val_loss": 2.880,
                "val_perplexity": 17.81,
                "full_val_perplexity": 17.93,
                "checkpoint_size_mb": 3.52,
                "gpu_training_time_s": 83,
            }
        }
        res_bytes = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(res_bytes)))
        self.end_headers()
        self.wfile.write(res_bytes)

    def handle_tokenize(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            text = body.get("text", "")
            ids = TOKENIZER.encode(text)
            tokens = [
                {
                    "id": tid,
                    "piece": TOKENIZER.decode([tid]),
                    "bytes": list(TOKENIZER.vocab[tid]) if tid < len(TOKENIZER.vocab) else []
                }
                for tid in ids
            ]
            res_bytes = json.dumps({"tokens": tokens, "count": len(tokens)}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(res_bytes)))
            self.end_headers()
            self.wfile.write(res_bytes)
        except Exception as e:
            self.send_error(400, f"Tokenization error: {e}")

    @torch.no_grad()
    def handle_generate(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            prompt = body.get("prompt", "")
            max_tokens = min(max(int(body.get("max_tokens", 100)), 1), 500)
            temperature = float(body.get("temperature", 0.8))
            top_k = body.get("top_k", 50)
            if top_k is not None:
                top_k = int(top_k)
                if top_k <= 0:
                    top_k = None
            top_p = body.get("top_p", 0.95)
            if top_p is not None:
                top_p = float(top_p)
                if top_p <= 0.0 or top_p >= 1.0:
                    top_p = None

            seed_val = body.get("seed")
            generator = None
            if seed_val is not None and str(seed_val).strip() != "":
                try:
                    generator = torch.Generator(device=DEVICE).manual_seed(int(seed_val))
                except Exception:
                    generator = None

            prompt_ids = TOKENIZER.encode(prompt)
            if not prompt_ids:
                prompt_ids = [0]

            idx = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.close_connection = True

            start_t = time.perf_counter()
            MODEL.eval()

            for step in range(max_tokens):
                cond = idx[:, -CONFIG.block_size:]
                logits, _ = MODEL(cond)
                nxt = sample_next(
                    logits[:, -1, :],
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    generator=generator
                )
                nxt_id = nxt.item()
                idx = torch.cat([idx, nxt], dim=1)

                token_str = TOKENIZER.decode([nxt_id])
                chunk = json.dumps({"token": token_str, "id": nxt_id, "step": step + 1, "done": False})
                self.wfile.write(f"data: {chunk}\n\n".encode("utf-8"))
                self.wfile.flush()

            elapsed = time.perf_counter() - start_t
            done_msg = json.dumps({
                "done": True,
                "total_tokens": max_tokens,
                "elapsed_s": round(elapsed, 3),
                "tokens_per_sec": round(max_tokens / max(elapsed, 0.001), 1)
            })
            self.wfile.write(f"data: {done_msg}\n\n".encode("utf-8"))
            self.wfile.flush()

        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            try:
                err_msg = json.dumps({"error": str(e), "done": True})
                self.wfile.write(f"data: {err_msg}\n\n".encode("utf-8"))
                self.wfile.flush()
            except Exception:
                pass


def run_server(port: int = 8000, host: str = "127.0.0.1"):
    init_model()
    server_addr = (host, port)
    try:
        httpd = HTTPServer(server_addr, WorkbenchHandler)
    except OSError:
        port = 8080
        server_addr = (host, port)
        httpd = HTTPServer(server_addr, WorkbenchHandler)

    print(f"[gpt-scratch] Workbench active on http://{host}:{port}/ (device: {DEVICE})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[gpt-scratch] Server terminated.")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gpt-scratch local inspector workbench")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    args = parser.parse_args()
    run_server(port=args.port, host=args.host)
