import os
import wandb
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

from collections import defaultdict, OrderedDict

agent = ["PPO"] #, "PPO_LSTM"]
games = [ "Stratego", "TicTacToe", "Diamant", "ExplodingKittens", "LoveLetter", "DotsAndBoxes", "SushiGo"] # "TAG/TicTacToe", "TAG/Stratego",
stat_type = "sp" #  "sp" or "opp"
opponents = ["random", "osla", "mcts"]
n_players = 2
reward_type = "SCORE"
ENTITY = "martinballa"  # '<entity>'
project = "PyTAG-SP"  # '<project>'
api = wandb.Api()

window = 100
results = {}
result_lst = []


# collect all the data first and then plot them
runs = api.runs(f"{ENTITY}/{project}")

# filter wrong seed and unfinished runs
runs = [run for run in runs if run.config["seed"] != 1 and run.state == "finished" and
        run.config["n_players"] == n_players and run.config["learning_rate"] == 0.001 and
        run.config["reward_type"] == reward_type]

# agents = defaultdict(list) #{list}
wins = defaultdict(list)
ties = defaultdict(list)
losses = defaultdict(list)
score_diffs = defaultdict(list)

filter_1m = True

# todo get final stats -> maybe summary?
# average final stats

# todo check learning rates
for run in runs:
    # seed = run.config["seed"]
    game = run.config["env_id"]
    for opp in opponents:
        wins[f"{game}_{opp}"].append(run.summary[f"eval/{opp}/episodic_wins"])
        ties[f"{game}_{opp}"].append(run.summary[f"eval/{opp}/episodic_ties"])
        losses[f"{game}_{opp}"].append(run.summary[f"eval/{opp}/episodic_losses"])
        score_diffs[f"{game}_{opp}"].append(run.summary[f"eval/{opp}/episodic_score_diff"])

        # print(runs[0].summary[f"eval/{opp}/mean_return"])

full_stats = defaultdict(list)
if stat_type == "opp":

    metrics = ["episodic_wins", "episodic_ties", "episodic_losses", "episodic_score_diff"]
    for run in runs:
        game = run.config["env_id"]

        keys = ["charts/global_step"]
        for metric in metrics:
            for opp in opponents:
                full_stats[f"{game}_{opp}_{metric}"] = run.history(keys=["global_step", f"eval/{opp}/{metric}"], samples=5000)

    # todo combined plots - all agents on one plot?
    window = 10
    titles = ["win rate", "tie rate", "loss rate", "score difference"]
    metrics = ["episodic_wins", "episodic_ties", "episodic_losses", "episodic_score_diff"]
    for metric, title in zip(metrics, titles):
        for game in games:
            fig, ax = plt.subplots(figsize=(6, 4))
            if metric != "episodic_score_diff":
                ax.set_ylim([-0.1, 1.1])
            for opp in opponents:
                filename = os.path.expanduser(f"~/data/pyTAG-SP/plots/{game}/{game}_{metric}_{n_players}p.png")
                if filter_1m:
                    filename = os.path.expanduser(f"~/data/pyTAG-SP/plots/{game}/{game}_{metric}_{n_players}p_1m.png")
                if len(full_stats[f"{game}_{opp}_{metric}"]) == 0:
                    print(f"no data for {game}_{opp}_{metric}")
                    continue
                df = full_stats[f"{game}_{opp}_{metric}"].sort_values("global_step").rolling(window=window, min_periods=window).mean()
                if filter_1m:
                    df = df[df["global_step"] < int(1e6)]

                ax.plot(df["global_step"], df[f"eval/{opp}/{metric}"], label=f"{opp}")

            ncol=2,
            ax.legend(ncol=ncol, frameon=False)  # bbox_to_anchor=(0.90, 0.1)) #, loc='lower right', frameon=False)
            plt.xlabel("steps", labelpad=0)
            plt.ylabel(title)
            plt.title(f"{n_players} players {game} {title}")
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            plt.savefig(filename, bbox_inches="tight", dpi='figure', pad_inches=0)
            plt.clf()
elif stat_type == "sp":
    # todo implement self-play stats
    metrics = ["episodic_player_scores", "episodic_wins", "episodic_losses", "episodic_ties" , "episodic_score_diff", "episodic_return", "episodic_length", "episodic_outcomes", "total_ep_length"]
    for run in runs:
        game = run.config["env_id"]

        keys = ["charts/global_step"]
        for metric in metrics:
            full_stats[f"{game}_{metric}"] = run.history(keys=["global_step", f"charts/{metric}"], samples=5000)

    # todo combined plots - all agents on one plot?
    window = 100
    titles = metrics #["win rate", "tie rate", "loss rate", "score difference"]
    # metrics = ["episodic_wins", "episodic_ties", "episodic_losses", "episodic_score_diff"]
    for metric, title in zip(metrics, titles):
        for game in games:
            fig, ax = plt.subplots(figsize=(6, 4))
            if not ["score" not in metric and "length" in metric]:
                ax.set_ylim([-0.1, 1.1])
            filename = os.path.expanduser(f"~/data/pyTAG-SP/plots/{game}/sp_{game}_{metric}_{n_players}p.png")
            if filter_1m:
                filename = os.path.expanduser(f"~/data/pyTAG-SP/plots/{game}/sp_{game}_{metric}_{n_players}p_1m.png")
            if len(full_stats[f"{game}_{metric}"]) == 0:
                print(f"no data for {game}_{metric}")
                continue

            df = full_stats[f"{game}_{metric}"]
            if filter_1m:
                df = df[df["global_step"] < int(1e6)]
            df = df.sort_values("global_step").rolling(window=window, min_periods=window)
            mean = df.mean()
            sem = df.sem()
            # sem = full_stats[f"{game}_{metric}"].sort_values("global_step").rolling(window=window, min_periods=window).sem()
            # if filter_1m:
            #     df = df[df["global_step"] < int(1e6)]
            #     sem = sem[sem["global_step"] < int(1e6)]
            # mean = full_stats[f"{game}_{metric}"].sort_values("global_step").rolling(window=window, min_periods=window).mean()
            # total_steps = full_stats[f"{game}_{metric}"].rolling(window=window, min_periods=window).max()

            total_steps = np.linspace(0, mean["global_step"].max(), len(mean))
            kwargs = dict(alpha=0.2, linewidths=0) #, color=col, zorder=10 - counter)
            mean = mean["charts/" + metric]
            sem = sem["charts/" + metric]
            ax.fill_between(total_steps, mean - sem, mean + sem, **kwargs, zorder=10)
            ax.plot(total_steps, mean, label=f"SP", zorder=100)

            # ncol=2,
            # ax.legend(ncol=ncol, frameon=False)  # bbox_to_anchor=(0.90, 0.1)) #, loc='lower right', frameon=False)
            plt.xlabel("steps", labelpad=0)
            plt.ylabel(title)
            plt.title(f"{n_players} players {game} {title} self-play")
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            plt.savefig(filename, bbox_inches="tight", dpi='figure', pad_inches=0)
            plt.clf()
        plt.close("all")
print("all done")

# from previous plots
# groups = OrderedDict(sorted(groups.items(), key=lambda item: item[0]))
#
# fig, ax = plt.subplots(figsize=(6, 4))
# counter = 0
# colors = plt.cm.rainbow(np.linspace(0, 1, len(groups)))
# label = groups.keys()
# # https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
# line_types = ["-", "--"]  # (0, (5, 10))]
# if len(labels) != len(groups):
#     labels = groups
# for g, label in zip(groups, labels):
#     df = pd.concat(groups[g], ignore_index=True).sort_values("global_step")
#
#     line_type = line_types[0]
#     col = colors[counter]
#
#     min_period = len(groups[g])  # this is how many datapoints we have for the same total_step
#     window_ = min_period * window  # window * len(dfs)  # adjust window to the extra data points
#     # this leaves too few datapoints
#     # mean = df["test/rewards"].rolling(window=window_, min_periods=window_).mean().dropna()[0::window_]
#     # std = df["test/rewards"].rolling(window=window_, min_periods=window_).sem().dropna()[0::window_]
#     mean = df[METRIC_NAME].dropna().rolling(window=window_, min_periods=min_period).mean()
#     std = df[METRIC_NAME].dropna().rolling(window=window_, min_periods=min_period).sem()
#     total_steps = df[METRIC_NAME].dropna().rolling(window=window_, min_periods=min_period).max()
#
#     total_steps = np.linspace(0, np.max(total_steps), len(mean))
#     kwargs = dict(alpha=0.2, linewidths=0, color=col, zorder=10 - counter)
#     ax.fill_between(total_steps, mean - std, mean + std, **kwargs)
#
#     ax.plot(total_steps, mean, linestyle=line_type, linewidth=1, label=label, color=col, zorder=100 - counter)
#     counter += 1
#
# # ncol=2,
# ax.legend(ncol=ncols, frameon=False)  # bbox_to_anchor=(0.90, 0.1)) #, loc='lower right', frameon=False)
# plt.title(title)
# plt.xlabel("2.5e6 steps", labelpad=0)
# plt.ylabel(metric)
# if metric != "length":
#     ax.set_ylim(Y_RANGES)
# ax.spines["top"].set_visible(False)
# # ax.spines["bottom"].set_visible(False)
# ax.spines["right"].set_visible(False)
#
# if not os.path.exists(os.path.dirname(filename)):
#     os.makedirs(os.path.dirname(filename))
# plt.savefig(filename, bbox_inches="tight", dpi='figure', pad_inches=0)
#
# # save plots
# ax.spines["left"].set_visible(False)
# if filename is not None:
#     plt.margins(0, 0)
#     plt.savefig(filename, bbox_inches="tight", dpi='figure', pad_inches=0)
#     print("figure saved")
#
# plt.show()
# print("")
# [print(f"{key}: {len(groups[key])}") for key in groups.keys()]
#
# for run in runs:
#     if run.state != "finished":
#         continue
#     if run.config["opponent"] == "mcts":
#         continue
#     if "old" in run.tags:
#         continue
#     values = []
#     ignore = False
#
#     for filter, val in zip(filters, filter_values):
#         if filter in run.config:
#             if val != "" and run.config[filter] != val:
#                 ignore = True
#                 break
#             values.append(run.config[filter])
#         else:
#             values.append(str(-1)) # means not applicable
#     if ignore:
#         continue
#
#     # todo update to: scan_history to get all the data
#     # data = run.scan_history(keys=["global_step", "charts/episodic_return"])
#     data = run.history(keys=["global_step", "charts/SPS", "charts/episodic_return", "charts/episodic_wins", "charts/episodic_length"], samples=5000)
#     data["name"] = str(values)
#     summaries.append(data)
#
#     if str(values) in groups:
#         groups[str(values)].append(data)
#     else:
#         groups[str(values)] = [data]
#         labels.append((values[0], values[-1][4:]))
#
#
# # todo specify order and rename them
# groups = OrderedDict(sorted(groups.items(), key=lambda item: item[0]))
#
# # fig, ax = plt.subplots(figsize=(6, 4))
# counter = 0
#
# if len(labels) != len(groups):
#     labels = groups
# summary_df = pd.DataFrame()
# for g, label in zip(groups, labels):
#     df = pd.concat(groups[g], ignore_index=True).sort_values("global_step")
#
#     min_period = len(groups[g])# this is how many datapoints we have for the same total_step
#     window_ = min_period * window #window * len(dfs)  # adjust window to the extra data points
#     # this leaves too few datapoints
#     # mean = df["test/rewards"].rolling(window=window_, min_periods=window_).mean().dropna()[0::window_]
#     # std = df["test/rewards"].rolling(window=window_, min_periods=window_).sem().dropna()[0::window_]
#     # mean = df[METRIC_NAME].dropna().mean()
#     # std = df[METRIC_NAME].dropna().sem()
#     with pd.option_context('display.float_format', '{:0.2f}'.format):
#         last_wins = f"{df['charts/episodic_wins'][-window:].mean():.2f}({df['charts/episodic_wins'][-window:].sem():.2f})"
#         last_score = f"{df['charts/episodic_return'][-window:].mean():.2f}({df['charts/episodic_return'][-window:].sem():.2f})"
#         last_length = f"{df['charts/episodic_length'][-window:].mean():.2f}({df['charts/episodic_length'][-window:].sem():.2f})"
#         last_speed = f"{df['charts/SPS'][-window:].mean():.2f}({df['charts/SPS'][-window:].sem():.2f})"
#         mean_speed = f"{df['charts/SPS'].mean():.2f}({df['charts/SPS'].sem():.2f})"
#         first_speed = f"{df['charts/SPS'][:window].mean():.2f}({df['charts/SPS'][:window].sem():.2f})"
#         # total_steps = df[METRIC_NAME].max()
#         # df[METRIC_NAME][-window:].mean() # can access last 100 steps
#         # print(f"{g} {metric}: {mean}({std})")
#         # todo convert this into an easily readable table - set precision
#         print(f"{g} last wins: {last_wins} last score: {last_score} last length: {last_length} last speed: {last_speed} mean speed: {mean_speed} first speed: {first_speed}")
#         # print(f"{g[-1]}}")
#         result_lst.append([g[1], g[g.find("/")+1:g.find("]")], last_wins, last_score, last_length, last_speed, mean_speed, first_speed])
#
# results_df = pd.DataFrame(result_lst, columns=["n_players", "game", "last_wins", "last_score", "last_length", "last_speed", "mean_speed", "first_speed"])
#
# print(result_lst)
# print("################# latex ##################")
# print(results_df.to_latex())
# print("################# latex (dropped first/last) ##################")
# print(results_df.drop(["last_speed", "first_speed"], axis=1).to_latex())
# # [print(f"{key} {results[key]}: {len(groups[key])}") for key in groups.keys()]
