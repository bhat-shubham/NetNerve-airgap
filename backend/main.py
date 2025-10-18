from typing import Annotated, List
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi import FastAPI, File, HTTPException, UploadFile, Body, Request
from fastapi.staticfiles import StaticFiles
from scapy.all import rdpcap
from scapy.layers.inet import IP, TCP, UDP, ICMP
from scapy.layers.l2 import ARP
from dotenv import load_dotenv
import asyncio, base64, uuid, os, datetime, re, multiprocessing, tempfile, logging, traceback, time
from llama_cpp import Llama
from queue import Queue

# setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

load_dotenv()

app = FastAPI(title="NetNerve Backend")

# gzip compression for large json responses
app.add_middleware(GZipMiddleware, minimum_size=500)

# serving frontend
if os.path.isdir("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

# cors setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS", "GET"],
    allow_headers=["Content-Type"],
)

@app.get("/")
async def root():
    return {"status": "success", "message": "NetNerve optimized backend is live!"}

@app.get("/health")
async def health_check():
    return {
        "status": "success",
        "model_loaded": bool(model_pool.qsize() > 0 if 'model_pool' in globals() else False),
    }

# pcap parsing
async def extract_packet_data(file_path):
    try:
        packets = await asyncio.to_thread(rdpcap, file_path)
    except Exception as e:
        logging.error(f"failed to parse pcap: {e}")
        raise HTTPException(status_code=400, detail="Error reading PCAP file")

    data = []
    for pkt in packets:
        pkt_info = {}
        if IP in pkt:
            pkt_info["src_ip"] = pkt[IP].src
            pkt_info["dst_ip"] = pkt[IP].dst
            pkt_info["packet_len"] = len(pkt)
            pkt_info["timestamp"] = pkt.time
            if TCP in pkt:
                pkt_info["protocol"] = "TCP"
                pkt_info["src_port"] = pkt[TCP].sport
                pkt_info["dst_port"] = pkt[TCP].dport
                pkt_info["flags"] = str(pkt[TCP].flags)
            elif UDP in pkt:
                pkt_info["protocol"] = "UDP"
                pkt_info["src_port"] = pkt[UDP].sport
                pkt_info["dst_port"] = pkt[UDP].dport
            elif ICMP in pkt:
                pkt_info["protocol"] = "ICMP"
        elif ARP in pkt:
            pkt_info["protocol"] = "ARP"
        if pkt_info:
            data.append(pkt_info)
    return data


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    start_time = time.time()
    MAX_FILE_SIZE_MB = int(os.environ.get("MAX_UPLOAD_MB", "10"))
    valid_extensions = [".pcap", ".cap"]

    if not (file.filename and file.filename.lower().endswith(tuple(valid_extensions))):
        raise HTTPException(status_code=400, detail="Invalid file extension.")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File too large. Max {MAX_FILE_SIZE_MB} MB allowed.")

    tmp_dir = tempfile.gettempdir()
    file_path = os.path.join(tmp_dir, f"{uuid.uuid4()}.pcap")
    with open(file_path, "wb") as f:
        f.write(content)

    try:
        packets = await asyncio.to_thread(rdpcap, file_path)
        protocols = set()
        for packet in packets:
            layer = packet
            while layer:
                protocols.add(layer.__class__.__name__)
                layer = layer.payload
        packet_data = await extract_packet_data(file_path)
        total_data_size = sum(pkt.get("packet_len", 0) for pkt in packet_data)
        processing_time = int((time.time() - start_time) * 1000)
        return {
            "status": "success",
            "protocols": list(protocols),
            "packet_data": packet_data,
            "total_data_size": total_data_size,
            "processing_time_ms": processing_time,
        }
    except Exception as e:
        logging.error(f"pcap read error: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"Corrupted pcap file: {e}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


# pooled model setup
MODEL_PATH = "models/Llama-3.2-3B.Q4_K_M.gguf"
raw_system_prompt = os.environ.get("SYSTEM_PROMPT", "")
system_prompt = raw_system_prompt.strip().strip('"').strip("\n")
MODEL_POOL_SIZE = int(os.environ.get("MODEL_POOL_SIZE", "1"))
MODEL_N_CTX = int(os.environ.get("MODEL_N_CTX", "2048"))
MODEL_MAX_TOKENS = int(os.environ.get("MODEL_MAX_TOKENS", "2000"))

model_pool: Queue = Queue()
model_pool_lock = asyncio.Lock()
model_semaphore = None


@app.on_event("startup")
def load_llm_at_startup():
    global model_pool, model_semaphore

    if not os.path.exists(MODEL_PATH):
        logging.warning(f"model not found at {MODEL_PATH}")
        return

    cpu_count = max(1, multiprocessing.cpu_count())
    suggested_pool = 1 if cpu_count < 8 else min(2, cpu_count // 8)
    pool_size = min(MODEL_POOL_SIZE, suggested_pool) if MODEL_POOL_SIZE else suggested_pool
    model_semaphore = asyncio.Semaphore(pool_size)

    logging.info(f"creating model pool with {pool_size} instances (cpu_count={cpu_count})")

    for i in range(pool_size):
        try:
            threads_per_model = max(1, (cpu_count // pool_size) - 1)
            m = Llama(
                model_path=MODEL_PATH,
                n_ctx=MODEL_N_CTX,
                n_threads=min(threads_per_model, 10),
                n_batch=128,
                f16_kv=True,
                use_mlock=True,
                embedding=False,
            )
            model_pool.put(m)
            logging.info(f"loaded model instance {i+1}/{pool_size} (threads={threads_per_model})")
        except Exception:
            logging.exception(f"failed to load model instance {i+1}")

    if model_pool.empty():
        logging.error("no model instances loaded. llm augmentation disabled.")


# prompt builders
def build_summary_prompt(protocols, packet_data, total_data_size):
    proto_counts, src_counts, dst_counts = {}, {}, {}
    for pkt in packet_data:
        proto = pkt.get("protocol", "Unknown")
        proto_counts[proto] = proto_counts.get(proto, 0) + 1
        src = pkt.get("src_ip")
        dst = pkt.get("dst_ip")
        if src:
            src_counts[src] = src_counts.get(src, 0) + 1
        if dst:
            dst_counts[dst] = dst_counts.get(dst, 0) + 1

    max_packets_to_show = 200
    if len(packet_data) > max_packets_to_show:
        packet_data = packet_data[:max_packets_to_show]

    summary_data = {
        'total_packets': len(packet_data),
        'sample_size': min(len(packet_data), max_packets_to_show),
        'total_bytes': total_data_size,
        'protocols': protocols[:20],
        'protocol_stats': proto_counts,
        'top_sources': [{'ip': k, 'count': v} for k, v in sorted(src_counts.items(), key=lambda x: x[1], reverse=True)[:10]],
        'top_destinations': [{'ip': k, 'count': v} for k, v in sorted(dst_counts.items(), key=lambda x: x[1], reverse=True)[:10]],
    }

    parts = [
        "Network Capture Statistics:",
        f"- Total Packets Analyzed: {summary_data['total_packets']}",
        f"- Data Volume: {summary_data['total_bytes']} bytes",
        f"- Protocol Distribution: {', '.join(f'{k}: {v}' for k, v in summary_data['protocol_stats'].items())}",
        "\nTop Communication Patterns:",
        "- Source IPs: " + ', '.join(f"{ip['ip']} ({ip['count']} packets)" for ip in summary_data['top_sources'][:5]),
        "- Destination IPs: " + ', '.join(f"{ip['ip']} ({ip['count']} packets)" for ip in summary_data['top_destinations'][:5]),
        "\nNote: " + ("Analysis limited to 200 packets for processing efficiency." if len(packet_data) > max_packets_to_show else "Full capture analyzed."),
    ]
    return "\n".join(parts)


# fallback deterministic summary
def generate_deterministic_summary(packet_data, protocols):
    unique_src_ips = len(set(pkt.get("src_ip") for pkt in packet_data if "src_ip" in pkt))
    unique_dst_ips = len(set(pkt.get("dst_ip") for pkt in packet_data if "dst_ip" in pkt))
    return f"### Analysis Summary\nAnalyzed {len(packet_data)} packets across {len(protocols)} protocols. Communication between {unique_src_ips} sources and {unique_dst_ips} destinations."


# model main inference (pooled)
async def run_llm_inference(prompt: str) -> str:
    if model_pool.empty():
        raise HTTPException(status_code=503, detail="LLM not available")

    await model_semaphore.acquire()
    model = None

    try:
        async with model_pool_lock:
            model = model_pool.get_nowait()

        def call_model():
            try:
                out = model.create_completion(
                    prompt=prompt,
                    max_tokens=MODEL_MAX_TOKENS,
                    temperature=0.2,
                    top_p=0.1,
                    repeat_penalty=1.15,
                    stop=["</s>", "User:", "\n\n\n"],
                    stream=False,
                )
                if isinstance(out, dict):
                    return out.get("choices", [{}])[0].get("text", "")
                return str(out)
            except Exception as e:
                return f"[ERROR during inference] {e}"

        text = await asyncio.to_thread(call_model)
        if not text or text.startswith("[ERROR"):
            raise HTTPException(status_code=500, detail=f"LLM error: {text}")
        return text.strip()

    except Exception as e:
        logging.exception("inference failed")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if model:
            async with model_pool_lock:
                model_pool.put(model)
        model_semaphore.release()


# summary endpoint
@app.post("/generate-summary/")
async def generate_summary(
    protocols: list[str] = Body(...),
    packet_data: list[dict] = Body(...),
    total_data_size: int = Body(...),
):
    start_time = time.time()
    deterministic = generate_deterministic_summary(packet_data, protocols)

    if os.environ.get("USE_LLM_AUGMENTATION", "1") != "1":
        return {
            "status": "success",
            "analysisType": "deterministic",
            "summary": deterministic,
            "processing_time_ms": int((time.time() - start_time) * 1000),
        }

    user_prompt = build_summary_prompt(protocols, packet_data, total_data_size)
    system_header = (system_prompt + "\n\n") if system_prompt else ""

    wrapper = """
System: You are a SOC analyst generating a detailed traffic analysis report. Your output must follow this exact format:

### Analysis Summary
[2-3 sentences summarizing total packets, data volume, and key findings]

### Protocol Insights
[List each major protocol with packet counts and behavioral assessment]

### Traffic Behavior
[2-3 paragraphs on traffic patterns, anomalies, and session characteristics]

### Potential Threat Indicators
| Indicator | Description | Severity |
|-----------|-------------|----------|
[List anomalies with descriptions and severity (High/Medium/Low)]

### Conclusion
[1-2 sentences on security posture and recommendations]
[Note if analysis was truncated]

Here's the traffic data to analyze:
"""

    full_prompt = f"{system_header}\n{wrapper}\n{user_prompt}\n\nAnalyst: Generating comprehensive traffic analysis report:\n"

    try:
        llm_summary = await run_llm_inference(full_prompt)
        return {
            "status": "success",
            "analysisType": "ai",
            "summary": llm_summary,
            "processing_time_ms": int((time.time() - start_time) * 1000),
        }
    except HTTPException as e:
        logging.error(f"LLM inference failed: {e.detail}")
        return {
            "status": "error",
            "analysisType": "deterministic",
            "summary": deterministic,
            "message": f"AI analysis failed: {e.detail}. Using deterministic fallback.",
            "processing_time_ms": int((time.time() - start_time) * 1000),
        }
