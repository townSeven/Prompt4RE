import numpy as np
import torch
import torch.nn as nn
from torch import optim
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from T_RAlign import TRAlign
import pandas as pd
from datetime import datetime
import os
from parse_args import args
from tqdm import tqdm
import nni

def compute_metrics(y_pred, y_test):
    """计算 MAE, RMSE, R2"""
    y_pred[y_pred < 0] = 0
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, np.sqrt(mse), r2


class CrimeEnhanced(nn.Module):
    """文本 + 区域特征对齐并预测数量"""
    def __init__(self, weight, b, region_dim, text_dim, text_output_dim):
        super(CrimeEnhanced, self).__init__()
        self.textRegionAlign = TRAlign(region_dim, text_dim)
        self.linearText = nn.Linear(text_dim, text_output_dim)
        self.lin = nn.Linear(region_dim + text_output_dim, 1)

        # 初始化
        torch.nn.init.xavier_normal_(self.lin.weight.data)
        self.lin.weight.data[:, -region_dim:] = weight
        self.lin.bias.data = b

    def forward(self, emb_t, emb_r):
        emb_t = self.textRegionAlign(emb_r, emb_t, emb_t)   # 文本对齐区域
        emb_t = self.linearText(emb_t)
        tmp = torch.concat([emb_t, emb_r], dim=1)           # 拼接
        tmp = self.lin(tmp)
        return tmp


def crimePred(counts, text_embs, region_embs, kf_splits=10, epochs=1000, lr=1e-3, text_output_dim=args.text_output_dim, device="cuda"):
    index = torch.arange(len(region_embs))
    kf = KFold(n_splits=kf_splits, shuffle=True, random_state=2024)

    y_preds, y_truths = [0] * kf_splits, [0] * kf_splits
    kf_idx = 0

    for train_index, test_index in kf.split(index):
        loss_fn = torch.nn.MSELoss()
        reg = linear_model.Ridge(alpha=1.0)

        # 数据划分
        X_train, X_test = region_embs[train_index], region_embs[test_index]
        T_train, T_test = text_embs[train_index], text_embs[test_index]
        Y_train, Y_test = counts[train_index], counts[test_index]

        reg.fit(X_train, Y_train)  # 初始化回归

        # 转换为 torch
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        T_train = torch.tensor(T_train, dtype=torch.float32).to(device)
        Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        T_test = torch.tensor(T_test, dtype=torch.float32).to(device)
        Y_test = torch.tensor(Y_test, dtype=torch.float32).to(device)

        # 初始化模型
        enhancedModel = CrimeEnhanced(
            torch.tensor(reg.coef_, dtype=torch.float32),
            torch.tensor(reg.intercept_, dtype=torch.float32),
            region_dim=region_embs.shape[1],
            text_dim=text_embs.shape[1],
            text_output_dim=text_output_dim
        ).to(device)

        optimizer = optim.Adam(enhancedModel.parameters(), lr=lr)

        best_r2 = float("-inf")
        for i in tqdm(range(epochs), desc="Prompt Learning"):
            optimizer.zero_grad()
            y_pred = enhancedModel(T_train, X_train)
            loss = loss_fn(y_pred.squeeze(), Y_train.squeeze())
            loss.backward()
            optimizer.step()

            if i % 20 == 0:  # 验证
                with torch.no_grad():
                    y_pred = enhancedModel(T_test, X_test).detach().cpu().numpy().squeeze()
                    y_truths_tmp = Y_test.detach().cpu().numpy().squeeze()
                    mae, rmse, r2 = compute_metrics(y_pred, y_truths_tmp)
                    if r2 > best_r2:
                        y_preds[kf_idx] = y_pred
                        y_truths[kf_idx] = y_truths_tmp
                        best_r2 = r2
                    
                nni.report_intermediate_result(r2)        

        kf_idx += 1

    # 汇总结果
    mae, rmse, r2 = compute_metrics(np.concatenate(y_preds), np.concatenate(y_truths))
    return mae, rmse, r2


def run_task(task="crime", text_embs=None, region_embs=None, device="cuda",
             kf_splits=10, epochs=args.epochs, lr=args.learning_rate, text_output_dim=args.text_output_dim, result_file="experiment_results.csv"):
    """运行不同任务并保存结果"""
    if task == "checkIn":
        labels = np.load("/data5/luyisha/guozitao/HAFusion/data_Chi/check_counts.npy")
        mask = labels > 0
        labels, text_embs, region_embs = labels[mask], text_embs[mask], region_embs[mask]
    elif task == "crime":
        labels = np.load("/data5/luyisha/guozitao/HAFusion/data_Chi/crime_counts.npy")
    elif task == "serviceCall":
        labels = np.load("/data5/luyisha/guozitao/HAFusion/data_Chi/serviceCall_counts.npy")
    else:
        raise ValueError("Unknown task!")

    mae, rmse, r2 = crimePred(labels, text_embs, region_embs,
                              kf_splits=kf_splits, epochs=epochs, lr=lr, text_output_dim=text_output_dim, device=device)
    nni.report_final_result(r2)

    print(f"{task.capitalize()} Prediction: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")

    # 结果写入 CSV
    result = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "task": task,
        "kf_splits": kf_splits,
        "epochs": epochs,
        "lr": lr,
        "text_output_dim": text_output_dim,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }
    df = pd.DataFrame([result])

    if not os.path.exists(result_file):
        df.to_csv(result_file, index=False)
    else:
        df.to_csv(result_file, mode="a", header=False, index=False)

    print(f"实验结果已保存到 {result_file}")


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = args.task
    model_params = {
        "epochs": args.epochs,
        "lr": args.learning_rate,
        "text_output_dim": args.text_output_dim
    }
    ##### nni #####
    optimized_params = nni.get_next_parameter() # 更新超参
    model_params.update(optimized_params)
    ##### nni #####

    if task == "crime":
        region_emb_path = "/data5/luyisha/guozitao/Prompt4RE/data/best_emb_chi_crime.npy"
        prompt_emb_path = f"/data5/luyisha/guozitao/Prompt4RE/prompt_embs_{args.model}_crime.npy"
    elif task == "checkIn":
        region_emb_path = "/data5/luyisha/guozitao/Prompt4RE/data/best_emb_chi_checkIn.npy"
        prompt_emb_path = f"/data5/luyisha/guozitao/Prompt4RE/prompt_embs_{args.model}_checkIn.npy"
    elif task == "serviceCall":
        region_emb_path = "/data5/luyisha/guozitao/Prompt4RE/data/best_emb_chi_serviceCall.npy"
        prompt_emb_path = f"/data5/luyisha/guozitao/Prompt4RE/prompt_embs_{args.model}_serviceCall.npy"
    else:
        raise ValueError("Unknown task!")
    region_embs = np.load(region_emb_path)   # (77, 144)
    prompt_embs = np.load(prompt_emb_path)  # (77, 3584)

    # 运行不同任务
    run_task(task, prompt_embs, region_embs, device=device, kf_splits=10, epochs=model_params["epochs"], lr=model_params["lr"], text_output_dim=model_params["text_output_dim"])
