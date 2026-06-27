# LLM Pruning + QLoRA Healing for Laser CAM G-code Generation

將 Granite-Code-3B 模型透過 PruneMe 剪枝與 QLoRA 微調修復，
應用於雷射 CAM G-code 自動生成。

---

## Pipeline
DXF 檔案

↓ (dxf2laserfill: src/dxf2laserfill/)

Path JSON (掃描路徑的文字描述)

↓ (build_user_prompt)

Prompt (System + 雷射參數 + Path JSON)

↓ (LLM 生成)

Raw G-code

↓ (normalize_no_rule_gcode_keep_model_params)

G-code 輸出

↓ 

IoU (vs DXF 即時計算的 ground truth)

## 執行流程

```bash
# Step 1: 剪枝
python pruneme_granite.py

# Step 2: QLoRA Healing 
python qlora_heal_ep1.py

# Step 3: 推論 
python inference_no_rule_csv_new_with_timing.py

# Step 4: 評估
python run_evaluation.py \
    --dxf-dir inference_input \
    --gcode-dir ep1_full_inference_output \
    --param-csv parameters_test.csv \
    --timing-csv ep1_full_timing.csv \
    --output-dir ep1_full_eval \
    --skip-overlay
```

---

## 檔案說明

| 檔案 | 說明 |
|---|---|
| `pruneme_granite.py` | PruneMe 剪枝主腳本 |
| `qlora_heal_ep1.py` | QLoRA 微調  |
| `vllm_inference.py` | vLLM 批次推論 |
| `inference_no_rule_csv_new_with_timing.py` | 逐筆推論 (原版) |
| `run_evaluation.py` | 評估主腳本 |
| `evaluator.py` | IoU 計算核心 |
| `overlay_dxf_gcode.py` | 產生 overlay 圖 |
| `patch_save_raw.py` | 在推論腳本加入 raw output 儲存功能 |
| `src/dxf2laserfill/` | DXF 轉換套件 |

---
