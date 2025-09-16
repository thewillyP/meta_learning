import pandas as pd
from clearml import Task
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import mannwhitneyu
import gc


def exp_decay(t, a, b, c):
    return a * np.exp(-b * t) + c


def download_and_process_tasks_batch(tag, experiment_name, batch_size=20):
    print(f"DOWNLOADING TASKS FOR {experiment_name}: {tag}")

    tasks = Task.get_tasks(project_name="oho", tags=[tag])
    print(f"Found {len(tasks)} tasks")

    all_dfs = []

    for batch_start in range(0, len(tasks), batch_size):
        batch_end = min(batch_start + batch_size, len(tasks))
        batch_tasks = tasks[batch_start:batch_end]
        print(f"Processing batch {batch_start + 1}-{batch_end}/{len(tasks)}")

        data_rows = []

        for task in batch_tasks:
            base_task_data = {"task_id": task.id, "task_name": task.name}

            # Get config
            params = task.get_parameters() or {}
            for key, value in params.items():
                if any(x in key for x in ["learning_rate", "optimizer/_type", "seed/global_seed"]):
                    base_task_data[key] = value

            # Get ALL metrics from reported_scalars consistently
            reported_scalars = task.get_reported_scalars()

            for metric_name in ["validation/loss", "test/loss", "final_test/loss"]:
                if metric_name in reported_scalars:
                    variants = reported_scalars[metric_name]
                    for variant, data in variants.items():
                        if data["x"] and data["y"]:
                            for iteration, value in zip(data["x"], data["y"]):
                                if isinstance(value, (int, float)):
                                    row = base_task_data.copy()
                                    row["iteration"] = iteration
                                    row[f"metric/{metric_name}/{variant}"] = value
                                    data_rows.append(row)

            # Only add base task data if no time series data was added
            if not any(metric in reported_scalars for metric in ["validation/loss", "test/loss", "final_test/loss"]):
                data_rows.append(base_task_data)

        if data_rows:
            batch_df = pd.DataFrame(data_rows)
            all_dfs.append(batch_df)

        gc.collect()

    if all_dfs:
        df = pd.concat(all_dfs, ignore_index=True)
    else:
        df = pd.DataFrame()

    print(f"DataFrame created: {df.shape}")
    if not df.empty:
        final_test_cols = [col for col in df.columns if "final_test" in col and "loss" in col]
        print(f"Final test loss columns found: {final_test_cols}")

        # Check for zero values
        if final_test_cols:
            final_test_col = final_test_cols[0]
            zero_count = (df[final_test_col] == 0).sum()
            if zero_count > 0:
                print(f"WARNING: {zero_count} zero values detected in final test loss!")
                zero_tasks = df[df[final_test_col] == 0]["task_id"].unique()
                print(f"Tasks with zero final test loss: {zero_tasks[:5]}...")

    return df


def analyze_single_optimizer(df, experiment_name):
    print(f"ANALYZING {experiment_name}")

    final_test_cols = [col for col in df.columns if "final_test" in col and "loss" in col]
    if not final_test_cols:
        print("ERROR: No final test loss column found")
        return {}
    final_test_col = final_test_cols[0]

    validation_data = df.dropna(
        subset=["metric/validation/loss/validation_loss", "config/learners/0/optimizer/learning_rate"]
    )

    # Get all learning rates and check each for zero final test losses
    learning_rates = validation_data["config/learners/0/optimizer/learning_rate"].unique()
    valid_lrs = []

    for lr in learning_rates:
        # Get final test data for this learning rate
        final_test_data = df.dropna(subset=[final_test_col, "config/learners/0/optimizer/learning_rate"])
        lr_final_test = final_test_data[final_test_data["config/learners/0/optimizer/learning_rate"] == lr]
        unique_final_test = lr_final_test.drop_duplicates(["task_id"])
        final_losses = unique_final_test[final_test_col].values

        # Check for zeros
        zero_count = (final_losses == 0).sum()
        if zero_count == 0:
            valid_lrs.append(lr)
        else:
            print(f"EXCLUDING learning rate {lr}: {zero_count} tasks have zero final test loss")

    if not valid_lrs:
        print("ERROR: No valid learning rates found (all have zero final test losses)")
        return {}

    print(f"Valid learning rates: {len(valid_lrs)}/{len(learning_rates)}")

    # Filter validation data to only valid learning rates
    valid_validation_data = validation_data[
        validation_data["config/learners/0/optimizer/learning_rate"].isin(valid_lrs)
    ]

    # Get best learning rate from valid ones
    last_iter_validation = (
        valid_validation_data.groupby(["task_id", "config/learners/0/optimizer/learning_rate"])
        .agg({"iteration": "max", "metric/validation/loss/validation_loss": "last"})
        .reset_index()
    )

    last_iter_validation = valid_validation_data.merge(
        last_iter_validation[["task_id", "config/learners/0/optimizer/learning_rate", "iteration"]],
        on=["task_id", "config/learners/0/optimizer/learning_rate", "iteration"],
    )

    lr_validation_summary = last_iter_validation.groupby("config/learners/0/optimizer/learning_rate").agg(
        {"metric/validation/loss/validation_loss": "mean"}
    )

    best_lr = lr_validation_summary["metric/validation/loss/validation_loss"].idxmin()
    print(f"Best learning rate: {best_lr}")

    # Get final test losses for best LR
    final_test_data = df.dropna(subset=[final_test_col, "config/learners/0/optimizer/learning_rate"])
    best_lr_final_test = final_test_data[final_test_data["config/learners/0/optimizer/learning_rate"] == best_lr]
    unique_final_test = best_lr_final_test.drop_duplicates(["task_id"])
    final_losses = unique_final_test[final_test_col].values

    print(f"Tasks with best LR: {len(unique_final_test)}")
    print(f"Final test loss: mean={np.mean(final_losses):.6f}, std={np.std(final_losses):.6f}, n={len(final_losses)}")

    # Get effective learning rates
    test_loss_curves = df[
        (df["config/learners/0/optimizer/learning_rate"] == best_lr) & df["metric/test/loss/test_loss"].notna()
    ]

    effective_learning_rates = []
    if not test_loss_curves.empty:
        best_lr_seeds = test_loss_curves["config/seed/global_seed"].unique()

        for seed in best_lr_seeds:
            seed_data = test_loss_curves[test_loss_curves["config/seed/global_seed"] == seed]
            if len(seed_data) > 5:
                try:
                    popt, _ = curve_fit(
                        exp_decay,
                        seed_data["iteration"].values,
                        seed_data["metric/test/loss/test_loss"].values,
                        bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
                        maxfev=2000,
                    )
                    effective_learning_rates.append(popt[1])
                except:
                    continue

    if effective_learning_rates:
        print(
            f"Effective LR: mean={np.mean(effective_learning_rates):.6f}, std={np.std(effective_learning_rates):.6f}, n={len(effective_learning_rates)}"
        )

    return {
        "single_optimizer": {
            "best_lr0": best_lr,
            "final_losses": final_losses,
            "effective_learning_rates": effective_learning_rates,
        }
    }


def analyze_dual_optimizer(df, experiment_name):
    print(f"ANALYZING {experiment_name}")

    final_test_cols = [col for col in df.columns if "final_test" in col and "loss" in col]
    if not final_test_cols:
        print("ERROR: No final test loss column found")
        return {}
    final_test_col = final_test_cols[0]

    optimizer_types = df["config/learners/0/optimizer/_type"].dropna().unique()
    print(f"Optimizer types: {optimizer_types}")

    results = {}

    for opt_type in optimizer_types:
        print(f"\nAnalyzing {opt_type}")
        opt_df = df[df["config/learners/0/optimizer/_type"] == opt_type]

        validation_data = opt_df.dropna(
            subset=[
                "metric/validation/loss/validation_loss",
                "config/learners/0/optimizer/learning_rate",
                "config/learners/1/optimizer/learning_rate",
            ]
        )

        if validation_data.empty:
            continue

        # Get all LR pairs and check each for zero final test losses
        lr_pairs = validation_data[
            ["config/learners/0/optimizer/learning_rate", "config/learners/1/optimizer/learning_rate"]
        ].drop_duplicates()
        valid_lr_pairs = []

        for _, row in lr_pairs.iterrows():
            lr0, lr1 = (
                row["config/learners/0/optimizer/learning_rate"],
                row["config/learners/1/optimizer/learning_rate"],
            )

            # Get final test data for this LR pair
            final_test_data = opt_df.dropna(subset=[final_test_col])
            pair_final_test = final_test_data[
                (final_test_data["config/learners/0/optimizer/learning_rate"] == lr0)
                & (final_test_data["config/learners/1/optimizer/learning_rate"] == lr1)
            ]
            unique_final_test = pair_final_test.drop_duplicates(["task_id"])
            final_losses = unique_final_test[final_test_col].values

            # Check for zeros
            zero_count = (final_losses == 0).sum()
            if zero_count == 0 and len(final_losses) > 0:
                valid_lr_pairs.append((lr0, lr1))
            elif zero_count > 0:
                print(f"EXCLUDING LR pair ({lr0}, {lr1}): {zero_count} tasks have zero final test loss")

        if not valid_lr_pairs:
            print(f"ERROR: No valid LR pairs found for {opt_type}")
            continue

        print(f"Valid LR pairs for {opt_type}: {len(valid_lr_pairs)}/{len(lr_pairs)}")

        # Filter validation data to only valid LR pairs
        valid_mask = validation_data.apply(
            lambda x: (x["config/learners/0/optimizer/learning_rate"], x["config/learners/1/optimizer/learning_rate"])
            in valid_lr_pairs,
            axis=1,
        )
        valid_validation_data = validation_data[valid_mask]

        # Find best LR pair from valid ones
        last_iter_validation = (
            valid_validation_data.groupby(
                ["task_id", "config/learners/0/optimizer/learning_rate", "config/learners/1/optimizer/learning_rate"]
            )
            .agg({"iteration": "max", "metric/validation/loss/validation_loss": "last"})
            .reset_index()
        )

        last_iter_validation = valid_validation_data.merge(
            last_iter_validation[
                [
                    "task_id",
                    "config/learners/0/optimizer/learning_rate",
                    "config/learners/1/optimizer/learning_rate",
                    "iteration",
                ]
            ],
            on=[
                "task_id",
                "config/learners/0/optimizer/learning_rate",
                "config/learners/1/optimizer/learning_rate",
                "iteration",
            ],
        )

        lr_validation_summary = last_iter_validation.groupby(
            ["config/learners/0/optimizer/learning_rate", "config/learners/1/optimizer/learning_rate"]
        ).agg({"metric/validation/loss/validation_loss": "mean"})

        best_idx = lr_validation_summary["metric/validation/loss/validation_loss"].idxmin()
        best_lr0, best_lr1 = best_idx
        print(f"Best LR pair: learner0={best_lr0}, learner1={best_lr1}")

        # Get final test losses for best LR pair
        final_test_data = opt_df.dropna(subset=[final_test_col])
        best_lr_final_test = final_test_data[
            (final_test_data["config/learners/0/optimizer/learning_rate"] == best_lr0)
            & (final_test_data["config/learners/1/optimizer/learning_rate"] == best_lr1)
        ]

        unique_final_test = best_lr_final_test.drop_duplicates(["task_id"])
        final_losses = unique_final_test[final_test_col].values

        print(f"Tasks with best LR pair: {len(unique_final_test)}")
        print(
            f"Final test loss: mean={np.mean(final_losses):.6f}, std={np.std(final_losses):.6f}, n={len(final_losses)}"
        )

        # Get effective learning rates
        test_loss_curves = opt_df[
            (opt_df["config/learners/0/optimizer/learning_rate"] == best_lr0)
            & (opt_df["config/learners/1/optimizer/learning_rate"] == best_lr1)
            & opt_df["metric/test/loss/test_loss"].notna()
        ]

        effective_learning_rates = []
        if not test_loss_curves.empty:
            best_lr_seeds = test_loss_curves["config/seed/global_seed"].unique()
            for seed in best_lr_seeds:
                seed_data = test_loss_curves[test_loss_curves["config/seed/global_seed"] == seed]
                if len(seed_data) > 5:
                    try:
                        popt, _ = curve_fit(
                            exp_decay,
                            seed_data["iteration"].values,
                            seed_data["metric/test/loss/test_loss"].values,
                            bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
                            maxfev=2000,
                        )
                        effective_learning_rates.append(popt[1])
                    except:
                        continue

        if effective_learning_rates:
            print(
                f"Effective LR: mean={np.mean(effective_learning_rates):.6f}, std={np.std(effective_learning_rates):.6f}, n={len(effective_learning_rates)}"
            )

        results[opt_type] = {
            "best_lr0": best_lr0,
            "best_lr1": best_lr1,
            "final_losses": final_losses,
            "effective_learning_rates": effective_learning_rates,
        }

    return results


def plot_and_test(all_results):
    # Flatten conditions
    all_conditions = []
    for exp_name, result_dict in all_results.items():
        for opt_type, data in result_dict.items():
            all_conditions.append((f"{exp_name}-{opt_type}", data))

    # Create plots - separate violin and box plots
    n_conditions = len(all_conditions)
    if n_conditions > 0:
        # Violin plots
        fig1, axes1 = plt.subplots(2, n_conditions, figsize=(4 * n_conditions, 8))
        if n_conditions == 1:
            axes1 = axes1.reshape(2, 1)

        for i, (name, data) in enumerate(all_conditions):
            # Final test loss violin
            final_losses = data["final_losses"]
            parts = axes1[0, i].violinplot([final_losses], positions=[0], showmeans=True, showmedians=True)

            mean_val = np.mean(final_losses)
            std_val = np.std(final_losses)
            axes1[0, i].text(
                0.1,
                mean_val,
                f"μ={mean_val:.4f}\nσ={std_val:.4f}",
                transform=axes1[0, i].get_xaxis_transform(),
                verticalalignment="center",
            )

            axes1[0, i].set_title(f"{name}: Final Test Loss")
            axes1[0, i].set_ylabel("Final Test Loss")
            axes1[0, i].set_xticks([0])
            axes1[0, i].set_xticklabels([name.split("-")[-1]])

            # Effective learning rate violin
            if data["effective_learning_rates"]:
                eff_lrs = data["effective_learning_rates"]
                parts = axes1[1, i].violinplot([eff_lrs], positions=[0], showmeans=True, showmedians=True)

                mean_val = np.mean(eff_lrs)
                std_val = np.std(eff_lrs)
                axes1[1, i].text(
                    0.1,
                    mean_val,
                    f"μ={mean_val:.4f}\nσ={std_val:.4f}",
                    transform=axes1[1, i].get_xaxis_transform(),
                    verticalalignment="center",
                )

                axes1[1, i].set_title(f"{name}: Effective LR")
                axes1[1, i].set_ylabel("Effective Learning Rate")
                axes1[1, i].set_xticks([0])
                axes1[1, i].set_xticklabels([name.split("-")[-1]])
            else:
                axes1[1, i].text(0.5, 0.5, "No Data", ha="center", va="center", transform=axes1[1, i].transAxes)
                axes1[1, i].set_title(f"{name}: Effective LR (No Data)")

        plt.tight_layout()
        plt.show()

        # Box plots
        fig2, axes2 = plt.subplots(2, n_conditions, figsize=(4 * n_conditions, 8))
        if n_conditions == 1:
            axes2 = axes2.reshape(2, 1)

        for i, (name, data) in enumerate(all_conditions):
            # Final test loss box
            final_losses = data["final_losses"]
            bp = axes2[0, i].boxplot(
                [final_losses], positions=[0], patch_artist=True, boxprops=dict(facecolor="lightblue", alpha=0.7)
            )

            mean_val = np.mean(final_losses)
            std_val = np.std(final_losses)
            axes2[0, i].text(
                0.1,
                mean_val,
                f"μ={mean_val:.4f}\nσ={std_val:.4f}",
                transform=axes2[0, i].get_xaxis_transform(),
                verticalalignment="center",
            )

            axes2[0, i].set_title(f"{name}: Final Test Loss")
            axes2[0, i].set_ylabel("Final Test Loss")
            axes2[0, i].set_xticks([0])
            axes2[0, i].set_xticklabels([name.split("-")[-1]])

            # Effective learning rate box
            if data["effective_learning_rates"]:
                eff_lrs = data["effective_learning_rates"]
                bp = axes2[1, i].boxplot(
                    [eff_lrs], positions=[0], patch_artist=True, boxprops=dict(facecolor="lightgreen", alpha=0.7)
                )

                mean_val = np.mean(eff_lrs)
                std_val = np.std(eff_lrs)
                axes2[1, i].text(
                    0.1,
                    mean_val,
                    f"μ={mean_val:.4f}\nσ={std_val:.4f}",
                    transform=axes2[1, i].get_xaxis_transform(),
                    verticalalignment="center",
                )

                axes2[1, i].set_title(f"{name}: Effective LR")
                axes2[1, i].set_ylabel("Effective Learning Rate")
                axes2[1, i].set_xticks([0])
                axes2[1, i].set_xticklabels([name.split("-")[-1]])
            else:
                axes2[1, i].text(0.5, 0.5, "No Data", ha="center", va="center", transform=axes2[1, i].transAxes)
                axes2[1, i].set_title(f"{name}: Effective LR (No Data)")

        plt.tight_layout()
        plt.show()

    # Mann-Whitney U tests
    print(f"\nMANN-WHITNEY U TESTS:")
    for i in range(len(all_conditions)):
        for j in range(i + 1, len(all_conditions)):
            name1, data1 = all_conditions[i]
            name2, data2 = all_conditions[j]

            print(f"\n{name1} vs {name2}:")

            # Final test loss (lower is better)
            if len(data1["final_losses"]) > 0 and len(data2["final_losses"]) > 0:
                statistic, p_value = mannwhitneyu(data1["final_losses"], data2["final_losses"], alternative="two-sided")
                print(f"  Final Test Loss: p={p_value:.6f}", end="")
                if p_value < 0.05:
                    better = name1 if np.mean(data1["final_losses"]) < np.mean(data2["final_losses"]) else name2
                    print(f" - SIGNIFICANT: {better} has lower final test loss (BETTER)")
                else:
                    print(f" - No significant difference")

            # Effective learning rate (higher is better)
            if len(data1["effective_learning_rates"]) > 0 and len(data2["effective_learning_rates"]) > 0:
                statistic, p_value = mannwhitneyu(
                    data1["effective_learning_rates"], data2["effective_learning_rates"], alternative="two-sided"
                )
                print(f"  Effective LR: p={p_value:.6f}", end="")
                if p_value < 0.05:
                    better = (
                        name1
                        if np.mean(data1["effective_learning_rates"]) > np.mean(data2["effective_learning_rates"])
                        else name2
                    )
                    print(f" - SIGNIFICANT: {better} has higher effective LR (BETTER)")
                else:
                    print(f" - No significant difference")


# Download data
df_exp1 = download_and_process_tasks_batch("opt: 6141b7bad8ad4ff583bca905426ba946", "EXPERIMENT 1")
df_exp2 = download_and_process_tasks_batch("opt: 68692b6cf1cf4e3cb3760eb43f64abe7", "EXPERIMENT 2")
df_exp3 = download_and_process_tasks_batch("opt: 43c6a9a3aae94198add30c71612cd544", "EXPERIMENT 3")

# Run analysis
results_exp1 = analyze_single_optimizer(df_exp1, "EXPERIMENT 1")
results_exp2 = analyze_dual_optimizer(df_exp2, "EXPERIMENT 2")
results_exp3 = analyze_dual_optimizer(df_exp3, "EXPERIMENT 3")

all_results = {"Experiment1": results_exp1, "Experiment2": results_exp2, "Experiment3": results_exp3}

plot_and_test(all_results)

print("\nAnalysis complete!")
print("Note: Higher effective learning rate = faster convergence (better)")
print("Learning rate combinations with zero final test losses were excluded from analysis")
