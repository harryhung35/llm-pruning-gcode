"""
Patch inference_no_rule_csv_with_timing.py to also save the raw model output
(before post-processing), so you can debug why the post-processor filters
everything out and leaves only G21\\nG90.

After running this patch, every inference run will save:
  - Filtered .gcode in   INFERENCE_OUTPUT_DIR/
  - Raw model output in  INFERENCE_OUTPUT_DIR/raw_outputs/  (NEW)

Run from llm-compressor directory:
    python patch_save_raw.py
"""

from pathlib import Path

p = Path("inference_no_rule_csv_with_timing.py")
if not p.exists():
    print("! inference_no_rule_csv_with_timing.py not found in current dir")
    print(f"  Current dir: {Path.cwd()}")
    exit(1)

code = p.read_text(encoding="utf-8")
Path("inference_no_rule_csv_with_timing.py.bak_raw").write_text(code, encoding="utf-8")
print("✓ Backup saved: inference_no_rule_csv_with_timing.py.bak_raw")

# Patch 1: Create raw_outputs directory at start of main()
old_mkdir = """    INFERENCE_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    INFERENCE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INFERENCE_PATH_JSON_DIR.mkdir(parents=True, exist_ok=True)"""

new_mkdir = """    INFERENCE_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    INFERENCE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INFERENCE_PATH_JSON_DIR.mkdir(parents=True, exist_ok=True)
    RAW_OUTPUT_DIR = INFERENCE_OUTPUT_DIR / "raw_outputs"
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)"""

if old_mkdir in code:
    code = code.replace(old_mkdir, new_mkdir)
    print("✓ Patched: added RAW_OUTPUT_DIR creation")
else:
    print("! Could not find mkdir block")

# Patch 2: Save raw_text right after generation, BEFORE normalize
# Find the line after generate_raw_text_with_timing returns
old_after_gen = """            first_token_latency_sec = fmt_sec(first_token_latency)
            inference_time_sec = fmt_sec(inference_time)

            gcode_text, _model_params = normalize_no_rule_gcode_keep_model_params("""

new_after_gen = """            first_token_latency_sec = fmt_sec(first_token_latency)
            inference_time_sec = fmt_sec(inference_time)

            # Save raw model output for debugging
            (RAW_OUTPUT_DIR / f"{dxf_file.stem}.txt").write_text(raw_text, encoding="utf-8")

            gcode_text, _model_params = normalize_no_rule_gcode_keep_model_params("""

if old_after_gen in code:
    code = code.replace(old_after_gen, new_after_gen)
    print("✓ Patched: save raw_text before normalize")
else:
    print("! Could not find raw_text save location")

p.write_text(code, encoding="utf-8")

# Verify
print("\n=== Verification ===")
print(f"  RAW_OUTPUT_DIR mentions: {code.count('RAW_OUTPUT_DIR')} (should be 3)")
print(f"  raw_outputs path:        {code.count('raw_outputs')} (should be 1)")
print("\nDone. Re-run inference_no_rule_csv_with_timing.py to populate raw_outputs/")
