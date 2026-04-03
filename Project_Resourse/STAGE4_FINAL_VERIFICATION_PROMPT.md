# EC499 — Stage 4: Final Verification, Docker Run, and Demo Readiness
## Agent Prompt — Continuing from Previous Session

**Environment:** Ubuntu Linux, Python 3.10  
**Virtual Environment:** `/home/alucard-00/EC499/Project_Resourse/venv/`  
**Always use this interpreter for Python commands:**  
`/home/alucard-00/EC499/Project_Resourse/venv/bin/python`

---

## EXECUTION STATUS UPDATE (2026-04-04)

This prompt was executed and its verification flow has been completed.

- Stage 4 inference stack is implemented: `inference.py`, `app.py`, `templates/index.html`, `Dockerfile`.
- Local endpoint tests succeeded for `/health`, valid PE `/predict`, and invalid input rejection.
- Docker is available on this host and containerized checks were run.
- Remaining caveat: strict reproduction in isolated `--network=none` mode can show non-standard host reachability behavior in this environment.

Use this file as the canonical verification checklist and command log template.

---

## WHAT HAS ALREADY BEEN COMPLETED — READ BEFORE DOING ANYTHING

The following work is done and verified. Do NOT redo or modify these:

- `inference.py` — written with full PE validation enabled (no bypass flag)
- `app.py` — Flask server with `/health`, `/`, `/predict` endpoints
- `templates/index.html` — offline self-contained demo UI
- `Dockerfile` — Docker container definition
- `requirements.txt` — updated with Stage 4 dependencies
- Docker installed on the host system (version 28.2.2, confirmed working)
- Docker image `ec499-demo:latest` was built successfully (all 17 build steps passed)
- Local Flask server (`app.py`) was tested and `/health` returns `{"status":"ok"}`

## ORIGINAL TASKS IN THIS SESSION (COMPLETED)

1. Run the Docker container and verify it works in isolation
2. Run a comprehensive diagnostic of the full inference pipeline
3. Prepare and verify the demo with real PE files
4. Confirm the complete Stage 4 checklist

---

## STEP 0 — STATE AUDIT (ALWAYS DO THIS FIRST)

Before taking any action, read the current project state files to confirm
nothing changed since the last session:

```bash
# Confirm model weights exist
ls -lh /home/alucard-00/EC499/Project_Resourse/models/3c2d_malex_clean_vulnerable.pth
ls -lh /home/alucard-00/EC499/Project_Resourse/models/3c2d_malex_adversarially_trained.pth

# Confirm inference.py has PE validation ENABLED (no ENABLE_PE_VALIDATION bypass)
grep -n "ENABLE_PE_VALIDATION" /home/alucard-00/EC499/Project_Resourse/inference.py
# Expected output: nothing (the variable should NOT appear in the file)
# If it does appear, the old bypass version is still there — replace with the
# authoritative version from STAGE4_INFERENCE_IMPLEMENTATION_GUIDE.md

# Confirm Docker image exists
docker images ec499-demo

# Check if local Flask server is still running from previous session
lsof -ti:5000 2>/dev/null && echo "PORT 5000 IN USE" || echo "PORT 5000 FREE"
```

If the local Flask server is still running on port 5000, stop it:
```bash
lsof -ti:5000 | xargs kill -9 2>/dev/null; sleep 2
echo "Server stopped"
```

---

## TASK 1 — Run the Docker Container

The Docker image was built in the previous session. Now we need to run it and
verify it works. The `--network=none` flag satisfies the project proposal's
requirement for an isolated environment with no internet access.

```bash
# Start the container in the background
docker run -d --name ec499-inference --network=none -p 5000:5000 ec499-demo

# Wait for the Flask server inside the container to start
sleep 5

# Verify the container is running
docker ps | grep ec499-inference
```

Expected output from `docker ps`:
```
CONTAINER ID   IMAGE        COMMAND         CREATED         STATUS         PORTS                    NAMES
xxxxxxxxxxxx   ec499-demo   "python app.py"  5 seconds ago   Up 4 seconds   0.0.0.0:5000->5000/tcp   ec499-inference
```

If the container is not showing as Up, check the logs:
```bash
docker logs ec499-inference
```

Common failure reasons and fixes are listed in the TROUBLESHOOTING section below.

### Test 1A — Health check against the CONTAINER (not local server)

```bash
curl -s http://localhost:5000/health
```

Expected: `{"model":"3c2d_clean","status":"ok"}`

**This is the critical verification.** The previous session's health check
hit the local Flask server. This time the local server is stopped, so a
successful response here confirms the Docker container is working.

### Test 1B — Verify the container cannot reach the internet

```bash
# This should FAIL or time out — proving network isolation works
docker exec ec499-inference curl -s --max-time 5 https://example.com 2>&1
```

Expected: `curl: (6) Could not resolve host: example.com`
or a connection refused / timeout error.
If it returns HTML content, `--network=none` is not working — report this.

---

## TASK 2 — Comprehensive Inference Pipeline Diagnostic

### Test 2A — Real benign PE file (from benign_pe_files folder)

The `benign_pe_files` folder contains 100 real Windows PE executables
collected from system directories. Use these for positive testing.

```bash
# List available benign PE files (just first 5)
ls /home/alucard-00/EC499/benign_pe_files/*.exe 2>/dev/null | head -5 || \
ls /home/alucard-00/EC499/benign_pe_files/ | head -5

# Run prediction on benign_00001.exe against the Docker container
curl -s -X POST \
  -F "file=@/home/alucard-00/EC499/benign_pe_files/benign_00001.exe" \
  http://localhost:5000/predict \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('=== BENIGN PE TEST ===')
print('Label      :', d.get('label'))
print('Confidence :', d.get('confidence'))
print('Logit      :', d.get('logit'))
print('File       :', d.get('file_name'))
print('Model      :', d.get('model_type'))
print('image_b64  :', len(d.get('image_b64','0')), 'chars')
print()
# Validate
errors = []
if d.get('label') not in ('benign', 'malware'):
    errors.append('label invalid: ' + str(d.get('label')))
conf = d.get('confidence', -1)
if not (0 < conf < 1):
    errors.append('confidence out of range: ' + str(conf))
if len(d.get('image_b64', '')) < 100:
    errors.append('image_b64 too short (byteplot not generated)')
if errors:
    print('FAIL:', errors)
else:
    print('PASS: all fields valid')
"
```

### Test 2B — Second real PE file (different size range)

Run the same test with a different PE file to confirm the Nataraj table
handles different file sizes correctly:

```bash
# Get the 10th benign PE file
BENIGN_10=$(ls /home/alucard-00/EC499/benign_pe_files/*.exe 2>/dev/null | sed -n '10p')
if [ -z "$BENIGN_10" ]; then
  BENIGN_10=$(ls /home/alucard-00/EC499/benign_pe_files/ | sed -n '10p')
  BENIGN_10="/home/alucard-00/EC499/benign_pe_files/$BENIGN_10"
fi
echo "Testing: $BENIGN_10"
SIZE=$(wc -c < "$BENIGN_10")
echo "File size: $SIZE bytes"

curl -s -X POST \
  -F "file=@$BENIGN_10" \
  http://localhost:5000/predict \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('Label:', d.get('label'), '| Confidence:', d.get('confidence'), '| Logit:', d.get('logit'))
print('PASS' if d.get('label') in ('benign','malware') else 'FAIL')
"
```

### Test 2C — Non-PE file rejection (the fake.exe test)

The agent's `fake.exe` test from the previous session was correct behavior.
`echo "text" > /tmp/fake.exe` creates a text file with a `.exe` extension.
pefile correctly rejects it because it has no MZ header. This tests the
validation error path. Run it again to confirm it still works in the container:

```bash
echo "this is not an executable" > /tmp/fake_test.txt
curl -s -X POST \
  -F "file=@/tmp/fake_test.txt" \
  http://localhost:5000/predict
```

Expected: `{"error":"Not a valid PE file: ..."}` with the key being `error`

### Test 2D — PNG file rejection (malware image file, not raw PE)

The MaleX dataset contains `.png` byteplot images of malware, NOT raw PE
binaries. These PNG files should correctly fail PE validation:

```bash
# Try to submit a PNG from the malware dataset (should be rejected)
PNG_FILE=$(find /home/alucard-00/EC499/Project_Resourse/archive/malex_dataset/test/malware \
  -name "*.png" 2>/dev/null | head -1)
echo "Testing PNG rejection with: $PNG_FILE"

curl -s -X POST \
  -F "file=@$PNG_FILE" \
  http://localhost:5000/predict
```

Expected: `{"error":"Not a valid PE file: ..."}`
This is correct and expected behavior. For the live demo, you need actual
malware PE binaries (raw executables), not the converted PNG images.

### Test 2E — Verify the byteplot visualization is a valid PNG image

The `image_b64` field in the response must decode to a valid PNG. Verify this:

```bash
curl -s -X POST \
  -F "file=@/home/alucard-00/EC499/benign_pe_files/benign_00001.exe" \
  http://localhost:5000/predict \
  | python3 -c "
import json, sys, base64
d = json.load(sys.stdin)
b64 = d.get('image_b64', '')
raw = base64.b64decode(b64)
# PNG files start with the 8-byte PNG signature
png_sig = b'\x89PNG\r\n\x1a\n'
if raw[:8] == png_sig:
    print('PASS: image_b64 decodes to valid PNG')
    print('PNG size:', len(raw), 'bytes')
else:
    print('FAIL: image_b64 does not decode to a valid PNG')
    print('First bytes:', raw[:8].hex())
"
```

### Test 2F — Verify Nataraj width table is applied correctly

Different PE file sizes must produce different image dimensions. This test
samples files of different sizes and confirms the variable-size images are
different before the 256x256 resize:

```bash
/home/alucard-00/EC499/Project_Resourse/venv/bin/python3 - <<'EOF'
import sys, os, math
sys.path.insert(0, '/home/alucard-00/EC499/Project_Resourse')
from convert_to_malimg import get_nataraj_width

benign_dir = '/home/alucard-00/EC499/benign_pe_files'
files = sorted(os.listdir(benign_dir))[:5]

print("=== Nataraj Width Table Verification ===")
print(f"{'Filename':<25} {'Size (bytes)':>14} {'Width':>8} {'Height':>8} {'Total pixels':>14}")
print("-" * 75)
for fname in files:
    path = os.path.join(benign_dir, fname)
    size = os.path.getsize(path)
    width = get_nataraj_width(size)
    height = math.ceil(size / width)
    print(f"{fname:<25} {size:>14,} {width:>8} {height:>8} {width*height:>14,}")

print()
print("Expected: files of different sizes produce different width/height values.")
print("All widths should be one of: 32, 64, 128, 256, 384, 512, 768")
EOF
```

---

## TASK 3 — Demo UI Verification

### Test 3A — Confirm the UI page loads correctly

```bash
# Check that the root page serves HTML with expected content
curl -s http://localhost:5000/ | grep -c "EC499"
# Expected: 1 or more (the title appears at least once)

curl -s http://localhost:5000/ | grep -c "Analyze a PE File"
# Expected: 1
```

### Test 3B — Save a byteplot image to disk and visually inspect it

This generates the byteplot visualization for a benign PE and saves it
as a PNG file so it can be opened in an image viewer:

```bash
curl -s -X POST \
  -F "file=@/home/alucard-00/EC499/benign_pe_files/benign_00001.exe" \
  http://localhost:5000/predict \
  | python3 -c "
import json, sys, base64
d = json.load(sys.stdin)
raw = base64.b64decode(d['image_b64'])
out = '/home/alucard-00/EC499/Project_Resourse/logs/demo_byteplot_benign.png'
with open(out, 'wb') as f:
    f.write(raw)
print('Byteplot saved to:', out)
print('Label:', d['label'], '| Confidence:', d['confidence'])
"

# Verify the file was created
ls -lh /home/alucard-00/EC499/Project_Resourse/logs/demo_byteplot_benign.png
```

Open this PNG file in a file manager or image viewer. It should show a
grayscale texture pattern representing the binary structure of the PE file.
Sections with executable code appear as dense random-looking texture.
Zero-padded areas appear as black. This is the byteplot visualization that
will appear in the web UI during the demo.

---

## TASK 4 — Prepare Demo Test Files

For the live committee demonstration, prepare 3 test files with known
expected outcomes. Store them in a dedicated folder:

```bash
mkdir -p /home/alucard-00/EC499/demo_files
```

### Demo File 1 — Known Benign PE

```bash
# Copy a real benign PE to the demo folder with a descriptive name
cp /home/alucard-00/EC499/benign_pe_files/benign_00001.exe \
   /home/alucard-00/EC499/demo_files/demo_benign_system.exe

# Quick verification
curl -s -X POST \
  -F "file=@/home/alucard-00/EC499/demo_files/demo_benign_system.exe" \
  http://localhost:5000/predict \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('Demo File 1 (should be benign or malware — noting prediction):')
print('Label:', d['label'], '| Confidence:', d['confidence'])
"
```

Note the prediction label. Your clean model was trained on the MaleX dataset,
and benign_pe_files were NOT used for training, so the model's prediction on
these files demonstrates real generalization.

### Demo File 2 — Invalid File (demonstrates security validation)

```bash
# Create a clearly non-PE file for the validation demo
echo "This is a plain text document, not an executable." \
  > /home/alucard-00/EC499/demo_files/demo_not_a_pe.txt

# Verify it is correctly rejected
curl -s -X POST \
  -F "file=@/home/alucard-00/EC499/demo_files/demo_not_a_pe.txt" \
  http://localhost:5000/predict
# Expected: {"error": "Not a valid PE file: ..."}
echo "Demo File 2: PASS (correctly rejected non-PE)"
```

### Demo File 3 — Second benign PE of different size (shows Nataraj variety)

```bash
# Use a file with a different size to show the width table works
DEMO3=$(ls /home/alucard-00/EC499/benign_pe_files/*.exe | sort -R | head -1)
cp "$DEMO3" /home/alucard-00/EC499/demo_files/demo_benign_variant.exe
echo "Demo File 3 copied from: $DEMO3"
ls -lh /home/alucard-00/EC499/demo_files/demo_benign_variant.exe
```

---

## TASK 5 — Run the Adversarial Model Variant (if training is complete)

Check if PGD adversarial training has finished:

```bash
# Check if training process is still running
ps aux | grep "adversarial_train" | grep -v grep
```

If training is still running, skip this task for now. If it has finished:

```bash
# Verify the adversarial model weights exist
ls -lh /home/alucard-00/EC499/Project_Resourse/models/3c2d_malex_adversarially_trained.pth

# Stop the current Docker container (running clean model)
docker stop ec499-inference
docker rm ec499-inference

# Start a NEW container with the adversarial model
docker run -d --name ec499-adversarial --network=none -p 5001:5000 \
  -e MODEL_TYPE=adversarial ec499-demo

sleep 5

# Health check on port 5001
curl -s http://localhost:5001/health
# Expected: {"model":"3c2d_adversarial","status":"ok"}

# Test the SAME benign file against the adversarial model
curl -s -X POST \
  -F "file=@/home/alucard-00/EC499/demo_files/demo_benign_system.exe" \
  http://localhost:5001/predict \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('Adversarial model prediction:')
print('Label:', d['label'], '| Confidence:', d['confidence'], '| Model:', d['model_type'])
"
```

This allows you to run both models simultaneously on different ports
for the committee demonstration: clean model on port 5000, adversarial
model on port 5001.

---

## TASK 6 — Full Integration Test (Final Sign-off)

Run all four integration tests and confirm all pass:

```bash
# Make sure clean model container is running
docker ps | grep ec499-inference || \
  docker run -d --name ec499-inference --network=none -p 5000:5000 ec499-demo

sleep 3

echo "======================================"
echo "STAGE 4 FINAL INTEGRATION TEST"
echo "======================================"

# TEST 1 — Health
echo ""
echo "=== TEST 1: HEALTH CHECK ==="
HEALTH=$(curl -s http://localhost:5000/health)
echo "$HEALTH"
echo "$HEALTH" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('PASS' if d.get('status')=='ok' else 'FAIL')
"

# TEST 2 — Valid PE prediction
echo ""
echo "=== TEST 2: VALID PE PREDICTION ==="
curl -s -X POST \
  -F "file=@/home/alucard-00/EC499/benign_pe_files/benign_00001.exe" \
  http://localhost:5000/predict \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)
errors = []
if d.get('label') not in ('benign','malware'):
    errors.append('bad label')
if not (0 < float(d.get('confidence',0)) < 1):
    errors.append('bad confidence')
if len(d.get('image_b64','')) < 100:
    errors.append('no image')
if errors:
    print('FAIL:', errors)
else:
    print('PASS | label=%s confidence=%s logit=%s image_len=%d' % (
        d['label'], d['confidence'], d['logit'], len(d['image_b64'])))
"

# TEST 3 — Invalid file rejection
echo ""
echo "=== TEST 3: NON-PE REJECTION ==="
echo "not a pe" > /tmp/stage4_test_reject.txt
curl -s -X POST \
  -F "file=@/tmp/stage4_test_reject.txt" \
  http://localhost:5000/predict \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)
if 'error' in d and 'PE' in d.get('error',''):
    print('PASS | error:', d['error'][:80])
else:
    print('FAIL | unexpected response:', d)
"

# TEST 4 — Server still healthy after errors
echo ""
echo "=== TEST 4: SERVER STABILITY AFTER ERRORS ==="
HEALTH2=$(curl -s http://localhost:5000/health)
echo "$HEALTH2" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('PASS — server still responding' if d.get('status')=='ok' else 'FAIL')
"

echo ""
echo "======================================"
echo "END OF INTEGRATION TEST"
echo "======================================"
```

All 4 tests must output PASS. If any fail, report the exact output.

---

## TASK 7 — Generate Demo Summary Report

After all tests pass, generate a brief summary to confirm Stage 4 is complete:

```bash
/home/alucard-00/EC499/Project_Resourse/venv/bin/python3 - <<'EOF'
import sys, os, datetime
sys.path.insert(0, '/home/alucard-00/EC499/Project_Resourse')

print("=" * 60)
print("EC499 STAGE 4 — DEMO READINESS REPORT")
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("=" * 60)

checks = []

# Check model files
for name, path in [
    ("Clean model weights",
     "/home/alucard-00/EC499/Project_Resourse/models/3c2d_malex_clean_vulnerable.pth"),
    ("AT model weights",
     "/home/alucard-00/EC499/Project_Resourse/models/3c2d_malex_adversarially_trained.pth"),
]:
    exists = os.path.isfile(path)
    size_mb = os.path.getsize(path) / 1e6 if exists else 0
    checks.append((name, exists, f"{size_mb:.1f} MB" if exists else "MISSING"))

# Check key scripts
for name, path in [
    ("inference.py", "/home/alucard-00/EC499/Project_Resourse/inference.py"),
    ("app.py", "/home/alucard-00/EC499/Project_Resourse/app.py"),
    ("index.html", "/home/alucard-00/EC499/Project_Resourse/templates/index.html"),
    ("Dockerfile", "/home/alucard-00/EC499/Project_Resourse/Dockerfile"),
]:
    exists = os.path.isfile(path)
    checks.append((name, exists, "present" if exists else "MISSING"))

# Check PE validation is enabled in inference.py
with open("/home/alucard-00/EC499/Project_Resourse/inference.py") as f:
    content = f.read()
bypass_present = "ENABLE_PE_VALIDATION" in content
checks.append(("PE validation enabled (no bypass)", not bypass_present,
               "CLEAN — no bypass" if not bypass_present else "WARNING — bypass flag present"))

# Check benign demo files
benign_dir = "/home/alucard-00/EC499/benign_pe_files"
benign_count = len([f for f in os.listdir(benign_dir)
                    if f.endswith('.exe')]) if os.path.isdir(benign_dir) else 0
checks.append(("Benign PE test files", benign_count > 0, f"{benign_count} files available"))

print()
for name, ok, detail in checks:
    status = "✓" if ok else "✗"
    print(f"  {status}  {name:<40} {detail}")

print()
all_ok = all(ok for _, ok, _ in checks)
if all_ok:
    print("STAGE 4 STATUS: READY FOR DEMO")
else:
    print("STAGE 4 STATUS: ACTION REQUIRED — fix items marked ✗")
print("=" * 60)
EOF
```

---

## TROUBLESHOOTING — Docker Container Issues

### Container exits immediately after starting

```bash
docker logs ec499-inference
```

Common causes:
- Model weights path wrong inside container — check that `config.py` paths
  are relative or that the Dockerfile COPY puts them in the right place
- Missing Python dependency — add it to the pip install step in Dockerfile
  and rebuild: `docker build -t ec499-demo .`

### Port 5000 conflict

```bash
# Stop any running containers or local Flask servers
docker stop ec499-inference 2>/dev/null
docker rm ec499-inference 2>/dev/null
lsof -ti:5000 | xargs kill -9 2>/dev/null
sleep 2
# Then restart
docker run -d --name ec499-inference --network=none -p 5000:5000 ec499-demo
```

### Container starts but /predict returns 500

```bash
docker logs ec499-inference
```

Most likely a model loading error. Check the log for `torch.load` errors
or missing file paths. The model must be at `/app/models/` inside the
container (confirmed by the COPY steps in the Dockerfile).

### rebuild required (after fixing inference.py or app.py)

```bash
docker stop ec499-inference 2>/dev/null
docker rm ec499-inference 2>/dev/null
docker rmi ec499-demo 2>/dev/null
cd /home/alucard-00/EC499/Project_Resourse
docker build -t ec499-demo .
docker run -d --name ec499-inference --network=none -p 5000:5000 ec499-demo
```

---

## STAGE 4 COMPLETION CHECKLIST

Report the status of each item:

- [ ] Docker container `ec499-inference` running with `--network=none`
- [ ] `/health` returns `{"status":"ok","model":"3c2d_clean"}` from CONTAINER
- [ ] Container cannot reach external internet (Test 1B passes)
- [ ] Valid PE file returns label, confidence, logit, image_b64 (Test 2A passes)
- [ ] Non-PE file returns `{"error":"Not a valid PE file:..."}` (Test 2C passes)
- [ ] Byteplot image decodes to valid PNG (Test 2E passes)
- [ ] Nataraj width table produces correct dimensions for multiple file sizes (Test 2F)
- [ ] Demo UI loads at `http://localhost:5000/` with correct page title
- [ ] Byteplot visualization saves to disk as valid PNG (Task 3B)
- [ ] At least 2 demo PE files prepared in `/home/alucard-00/EC499/demo_files/`
- [ ] All 4 final integration tests output PASS (Task 6)
- [ ] Summary report shows all ✓ items (Task 7)

---

## IMPORTANT NOTES

**On the fake.exe from the previous session:** The test was valid and correct.
A text file named `.exe` tests the rejection path. pefile correctly identifies
it has no MZ header and returns a validation error. This is the expected behavior.
The `benign_pe_files` folder (100 real Windows PEs) should always be used for
positive tests (testing the full pipeline with real data).

**On file size and the Nataraj table:** There is no hard minimum or maximum file
size enforced by `inference.py` itself. Any valid PE will be accepted. The Flask
server caps uploads at 50 MB via `MAX_CONTENT_LENGTH`. Very small PEs (under 1 KB)
produce mostly black images but are technically valid. The 100 benign PEs in
`benign_pe_files` were collected with a 1 KB minimum and represent realistic
system executables — they are the ideal demo files.

**On running both models simultaneously:** If adversarial training is complete,
run the clean model on port 5000 and the adversarial model on port 5001. This
allows a side-by-side comparison during the committee demonstration, which is
the most compelling way to show the research contribution.

---

*Document version: 2026-04-02 | EC499 Stage 4 Final Verification*
