import numpy as np
import pandas as pd

# ===== 输入路径 =====
pred_path = r"C:\Users\11952\Desktop\softhand_results\online test\xian\leap.npy"
leap_path = r"C:\Users\11952\Desktop\softhand_results\online test\xian\preds.npy"

# ===== 输出路径 =====
save_path = r"C:\Users\11952\Desktop\xian_preds_true.csv"

# ===== 加载数据 =====
pred_data = np.load(pred_path)
leap_data = np.load(leap_path)

# ===== 保证数据一一对应，展平处理 =====
pred_flat = pred_data.flatten()
leap_flat = leap_data.flatten()

# ===== 拼接为 DataFrame（第一列是leap，第二列是pred） =====
df = pd.DataFrame({
    "leap": leap_flat,
    "pred": pred_flat
})

# ===== 保存为 CSV =====
df.to_csv(save_path, index=False, encoding="utf-8-sig")
print(f"CSV 文件已保存到: {save_path}")
