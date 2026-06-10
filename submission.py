import json
import re

# 請將您的文字內容貼在 text_data 變數中，或者從檔案讀取
text_data = """
Submission and Description

Public Score

Select

submission_v54_3_MANUAL_0.58.csv
Complete · fSean · 14m ago
0.8614

submission_v55_AUTO.csv
Complete · fSean · 29m ago
0.8676

submission_v54_3_SOFT.csv
Complete · fSean · 36m ago
0.8394

v6_expand1.15.csv
Complete · MUHAMMAD RAZA HASSAN · 2h ago
0.9675

const_1.0.csv
Complete · MUHAMMAD RAZA HASSAN · 2h ago
0.9117

submission_v54_MANUAL_0.62.csv
Complete · fSean · 4h ago
0.8645

submission_v54_SOFT.csv
Complete · fSean · 4h ago
0.8458

submission_v54_2_SOFT.csv
Complete · fSean · 4h ago
0.8647

submission_v54_2_AUTO.csv
Complete · fSean · 5h ago
0.9529

submission_v54_AUTO.csv
Complete · fSean · 8h ago
0.8461

raw_h061_shift0380_4plus_to3.csv
Complete · anaceci18 · 9h ago
0.8396

raw_h060_shift0400_4plus_to3.csv
Complete · anaceci18 · 9h ago
0.8392

submission_v53_AUTO.csv
Complete · fSean · 12h ago
0.9190

v52_4plus_to_3.csv
Complete · anaceci18 · 17h ago
0.8215

v52_4_to_2_5_to_3.csv
Complete · anaceci18 · 17h ago
0.8217

v52_week45_3_to_2.csv
Complete · anaceci18 · 18h ago
0.8235

v52_isolated_3_to_2.csv
Complete · anaceci18 · 18h ago
0.8217

v52_4_to_3.csv
Complete · anaceci18 · 18h ago
0.8215

submission_52th_1_adj.csv
Complete · fSean · 20h ago
0.9769

submission_52th_2.csv
Complete · fSean · 20h ago
0.8219

submission_52th.csv
Complete · fSean · 21h ago
0.8217

submission_v52.csv
Complete · fSean · 1d ago
0.8250

submission_45th_30k_pure_rounded.csv
Complete · fSean · 1d ago · test 0.82
0.8246

submission_45th_30k_ShiftStretch.csv
Complete · fSean · 1d ago
0.8461

submission_50th.csv
Complete · fSean · 1d ago
0.9529

submission_49th_meta.csv
Complete · fSean · 2d ago
0.8953

submission_49th_C_stack.csv
Complete · fSean · 2d ago
0.9075

submission_49th_A_arg.csv
Complete · fSean · 2d ago
0.9576

submission_49th.csv
Complete · fSean · 2d ago
0.9011

lgbm_round_qrecent3_w05.csv
Complete · anaceci18 · 3d ago
0.8246

submission_v6.csv
Complete · MUHAMMAD RAZA HASSAN · 3d ago
0.9347

lgbm_tail_t23_240_t34_340_t45_440.csv
Complete · anaceci18 · 3d ago
0.8305

submission_45th_30k_optimized.csv
Complete · fSean · 3d ago
0.8458

uplift_combined_v37_tail.csv
Complete · anaceci18 · 3d ago · rounded upwards
0.8252

blend_lgbm90_v379_round.csv
Complete · anaceci18 · 3d ago
0.8246

tailblend_v37ge3p0_w25.csv
Complete · anaceci18 · 3d ago
0.8246

submission_v39_snap28_qrecent3_w12.csv
Complete · anaceci18 · 3d ago
0.9142

submission_45th_30k_ABraw_rounded.csv
Complete · fSean · 3d ago
0.8246

submission_47th.csv
Complete · fSean · 3d ago
0.8711

submission_48th_snap.csv
Complete · fSean · 3d ago
0.8672

submission_48th.csv
Complete · fSean · 4d ago
0.8646

v37_snap28_qrecent3_w12_resnap.csv
Complete · anaceci18 · 4d ago · POSSIBLE CANDIDATE
0.8282

v37_snap28_blend_qseason_all_10.csv
Complete · anaceci18 · 4d ago
0.8329

v37_snap28_blend_qseason_recent3_10.csv
Complete · anaceci18 · 4d ago
0.8293

v37_snap_tol290.csv
Complete · anaceci18 · 4d ago
0.8343

v37_asym_down25_up30.csv
Complete · anaceci18 · 4d ago
0.8363

v37_snap_zero30_tol25.csv
Complete · anaceci18 · 4d ago
0.8369

v37_snap_horizon_inc.csv
Complete · anaceci18 · 4d ago
0.8374

v37_snap_tol28new.csv
Complete · anaceci18 · 4d ago
0.8342

submission_45th_30k_ABraw.csv
Complete · fSean · 4d ago
0.8428

submission_45th_30k_naked.csv
Complete · fSean · 4d ago
0.8530

submission_45th_30k_25perzero.csv
Complete · fSean · 4d ago
0.8587

submission_45th.csv
Complete · fSean · 4d ago
0.8487

submission_45thopt.csv
Complete · fSean · 4d ago
0.8533

submission_45thstacked.csv
Complete · fSean · 4d ago
0.8944

v37_later_weeks_plus.csv
Complete · anaceci18 · 5d ago
0.8572

v37_snap_tol25.csv
Complete · anaceci18 · 5d ago
0.8378

v37_soft_round_20.csv
Complete · anaceci18 · 5d ago
0.8442

selective_top15_const1.csv
Complete · anaceci18 · 5d ago
1.1073

submission_lgbm_tabular.csv
Complete · anaceci18 · 5d ago
0.9428

submission_fold5.csv
Complete · anaceci18 · 5d ago
0.9538

sub_v6_x1.2.csv
Complete · MUHAMMAD RAZA HASSAN · 5d ago
1.0318

submission_44.csv
Complete · MUHAMMAD RAZA HASSAN · 5d ago
0.9803

win_blend_m1.35.csv
Complete · MUHAMMAD RAZA HASSAN · 5d ago
0.9474

submission_v28.csv
Complete · fSean · 5d ago · v28: per-region z-score weather normalization (drift-invariant), stronger reg (depth=6 l2=10), CV 0.5299, scale 0.85
0.8926

submission_v27.csv
Complete · fSean · 5d ago · v27: all 3 leakage bugs fixed (ghost feats + embargo gap + OOF expanding window), CatBoost GPU, CV 0.5507, scale 0.85
0.8790

submission_v26.csv
Complete · fSean · 5d ago · v26: stride=1 (1.72M samples) + 1000-iter prod CatBoost GPU depth=8, CV 0.5282, scale 0.85
0.8708

submission_v25.csv
Complete · fSean · 5d ago · v25: strict temporal CV + 800-iter production CatBoost GPU depth=8, CV 0.5274, scale 0.85
0.8724

submission_v24.csv
Complete · fSean · 5d ago · v24: strict temporal block CV (leak-free stats), CatBoost GPU depth=8, CV 0.5274, scale 0.9
0.8799

submission_v23.csv
Complete · fSean · 5d ago · v23
0.8826

submission_v22_seasonal.csv
Complete · fSean · 6d ago · v22: pure seasonal prior only (per-region per-month training-era means), no model correction
0.9586

submission_v21.csv
Complete · fSean · 6d ago · v21: CatBoost GPU residual model (pred = seasonal_prior + weather_correction), oracle floor 0.7785, CV 0.5219, no scale
0.8716

submission_v20.csv
Complete · fSean · 6d ago · v20: CatBoost GPU (depth=8, lr=0.03, ordered boost, 5K iters), CV 0.4984, scale 0.9
0.8791

submission_v19.csv
Complete · fSean · 6d ago · v19: two-stage XGBoost GPU (stage0: weather->y0, stage1: weather+inferred_y0->5wk), CV 0.4878, scale 0.85
0.8770

submission_v18.csv
Complete · fSean · 6d ago · v18: two-stage GPU MLP (stage0: weather->current_drought, stage1: weather+inferred_y0->5wk), 1024-dim 5-layer, GPU, scale 0.85
0.8801

submission_v17_knn.csv
Complete · fSean · 6d ago · v17: within-region kNN (k=10) on local z-score+anom28 similarity — drift-invariant retrieval approach
0.8980

submission_v16.csv
Complete · fSean · 6d ago · v16: EDA-driven local z-score norm + zero_prob + deficit_roll_cum4w (179 feats), stride=2, scale=0.85
0.8731

submission_v15.csv
Complete · fSean · 6d ago · v15: v2 features + seasonal-score-per-target-month (158), stride=1 dense windows (1.72M samples), scale 0.85
0.8718

submission_v14.csv
Complete · fSean · 6d ago · v14_a
0.8767

submission_44th.csv
Complete · fSean · 6d ago · v44
1.1028

sub_v6_x0.6.csv
Complete · MUHAMMAD RAZA HASSAN · 6d ago
0.9151

sub_ZEROS.csv
Complete · MUHAMMAD RAZA HASSAN · 6d ago
1.2088

sub_gate0.6.csv
Complete · MUHAMMAD RAZA HASSAN · 6d ago
0.9380

sub_gate0.3.csv
Complete · MUHAMMAD RAZA HASSAN · 6d ago
0.9350

sub_gate0.4.csv
Complete · MUHAMMAD RAZA HASSAN · 6d ago
0.9357

submission_v13.csv
Complete · fSean · 7d ago · v13: physical water-balance/SPEI drought features (precip-PET), v2-config GBM, scale 0.85
0.8693

submission_v12.csv
Complete · fSean · 7d ago · v12: density-ratio importance-weighted GBM (covariate-shift correction), scale 0.9
0.8725

submission_v11.csv
Complete · fSean · 7d ago · v11: MLP neural net (extrapolates vs GBM saturation), 3-seed ensemble, no scale
0.8759

submission_v10.csv
Complete · fSean · 7d ago · v10: 5-member LightGBM ensemble (seed+leaf diversity), v2 feats, scale 0.85
0.8664

submission_v9.csv
Complete · fSean · 7d ago · v9: v2 predictions x0.85 (test MAE-optimal is below mean due to right-skew)
0.8640

submission_v8.csv
Complete · fSean · 7d ago · v8: v2 predictions x1.14 (calibrate to test mean 1.21)
0.8857

submission_v7_zeros.csv
Complete · fSean · 7d ago · v7 DIAGNOSTIC: all-zeros probe to learn exact test mean score
1.2088

submission_v6.csv
Complete · fSean · 7d ago · v6: domain adaptation — drop drift-corrupted abs-temp/anom28, keep percentile/SPI replacements (119 feats)
0.8963

submission_v5.csv
Complete · fSean · 7d ago · v5: +drift-robust percentile/SPI drought features (175), v2-params, prod rounds x3 (CV 0.4917)
0.8855

submission_v4.csv
Complete · fSean · 7d ago · v4: drift-robust feature restriction (109/165, dropped absolute-temp & training-normal anomalies) to force transferable signal
0.8938

submission_v3.csv
Complete · fSean · 7d ago · v3: +drift-invariant dryness feats (dewpoint/wetbulb depression, within-window relanom, 165), fixed prod rounds, stronger reg (leaves31 mcs500)
0.8666

submission_v2.csv
Complete · fSean · 7d ago · v2: +monthly-normal anomaly & dry-streak feats (153), honest per-region temporal CV w/ leak-free climatology (CV 0.511, clim-only 0.857)
0.8647

submission_43.csv
Complete · MUHAMMAD RAZA HASSAN · 7d ago
0.8964

submission_v1.csv
Complete · fSean · 7d ago · v1 baseline: LightGBMx5 91d-window aggregates + region climatology, MAE obj (val macro MAE 0.5345)
0.8917

submission_42nd.csv
Complete · fSean · 7d ago · v42
0.9782

submission_42.csv
Complete · MUHAMMAD RAZA HASSAN · 8d ago
0.8694

submission_41st.csv
Complete · fSean · 8d ago · v41
0.9463

submission_point5.csv
Complete · fSean · 8d ago · v40_point5
1.0356

submission_20percent.csv
Complete · fSean · 8d ago · v40_20%
1.0322

submission_39th.csv
Complete · fSean · 8d ago · v39_fromv28
0.8777

submission_38th.csv
Complete · fSean · 8d ago · v38_TFT
0.9504

submissiontest.csv
Complete · anaceci18 · 9d ago
0.9888

submission_37th.csv
Complete · fSean · 9d ago · v37_optfromv28
0.8497

submission_36th.csv
Complete · fSean · 11d ago · v36
0.9428

submission_35th.csv
Complete · fSean · 11d ago · v35
0.9216

submission_32nd.csv
Complete · fSean · 13d ago · v32model
0.9052

submission_31st.csv
Complete · fSean · 14d ago · v31_back to PyTorch
0.8897

submission_31th.csv
Complete · MUHAMMAD RAZA HASSAN · 17d ago
1.0774

submission_v30.csv
Complete · fSean · 17d ago · v30_softinferfromv29
0.9273

submission_29th.csv
Complete · fSean · 17d ago · v29
1.0195

submission_28th.csv
Complete · fSean · 17d ago · v28
0.8690

submission_28th.csv
Complete · MUHAMMAD RAZA HASSAN · 17d ago
0.9049

submission_27th_tozero.csv
Complete · fSean · 18d ago · v27_tozero
0.8920

submission_27th.csv
Complete · fSean · 18d ago · v27model
0.8840

submission_27.csv
Complete · MUHAMMAD RAZA HASSAN · 18d ago
0.9222

submission_26th.csv
Complete · fSean · 18d ago · v26
0.8780

submission_25th.csv
Complete · fSean · 19d ago · v25
0.9254

submission_24th.csv
Complete · fSean · 19d ago · v24model
0.9874

submission_24th.csv
Complete · MUHAMMAD RAZA HASSAN · 19d ago
0.9264

submission_23rd.csv
Complete · fSean · 20d ago · v23model
0.8832

submission_22nd_tozero.csv
Complete · fSean · 20d ago · v22model_tozero
0.9192

submission_22nd.csv
Complete · fSean · 20d ago · v22model_simplified
0.9013

submission_20th.csv
Complete · MUHAMMAD RAZA HASSAN · 20d ago
1.0089

submission.csv
Complete · anaceci18 · 20d ago
1.0600

submission_19th.csv
Complete · fSean · 20d ago · v19model
1.0796

submission_18th.csv
Complete · fSean · 21d ago · v18model
0.9949

submission_17th.csv
Complete · fSean · 21d ago · v17model
0.9574

submission_16th.csv
Complete · fSean · 21d ago · v16model
1.1036

submission_15th.csv
Complete · fSean · 22d ago · v15model
1.0878

submission_14th.csv
Complete · fSean · 22d ago · v14model
1.0313

submission_12th.csv
Complete · fSean · 22d ago · v12model
1.1245

submission_10th.csv
Complete · fSean · 23d ago · v10model
1.0064

submission_8th.csv
Complete · fSean · 23d ago · v8model
1.0561

submission_7th.csv
Complete · fSean · 23d ago · v7model
1.1665

submission_6th.csv
Complete · fSean · 24d ago · v6model
1.0551

submission_5th.csv
Complete · fSean · 24d ago · v5model
1.0908
"""

# 若是從 txt 檔案讀取，請取消註解以下兩行並修改檔名
# with open('fSean_submissions.txt', 'r', encoding='utf-8') as f:
#     text_data = f.read()

def parse_submissions(text):
    submissions = []
    # 使用正規表示式匹配每一筆紀錄的模式
    # 模式說明：
    # 1. (.*\.csv) -> 檔名
    # 2. Complete \· (.*?) \· (.*?)(?: \· (.*?))? -> 作者、時間、(可選) 描述
    # 3. ([\d\.]+) -> 分數
    pattern = re.compile(
        r'(?P<fileName>[^\n]+\.csv)\n'
        r'Complete · (?P<author>[^·]+) · (?P<time>[^·\n]+)(?: · (?P<description>[^\n]+))?\n'
        r'(?P<score>[\d\.]+)',
        re.MULTILINE
    )

    for match in pattern.finditer(text):
        author = match.group('author').strip()
        
        # 僅篩選 fSean 的資料
        if author == 'fSean':
            desc = match.group('description')
            submissions.append({
                "fileName": match.group('fileName').strip(),
                "description": desc.strip() if desc else "",
                "score": float(match.group('score').strip())
            })
            
    return submissions

# 執行解析
fsean_data = parse_submissions(text_data)

# 轉換成 JSON 格式字串並印出
json_output = json.dumps(fsean_data, indent=4, ensure_ascii=False)
print(json_output)

# 選擇性：寫入 JSON 檔案
with open('fSean_submissions.json', 'w', encoding='utf-8') as f:
    json.dump(fsean_data, f, indent=4, ensure_ascii=False)
    print("\n已成功儲存至 fSean_submissions.json")