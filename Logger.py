from typing import Dict, List, Tuple, Any
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
import json
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
import os
import pandas as pd


class Logger:
    def __init__(self, logging_path: str, num_episodes: int, num_iterations: int):

        self.num_episodes = num_episodes
        self.num_iterations = num_iterations
        ## get current time to set up logging directory
        date = datetime.now()
        temp = list([date.year, date.month, date.day, date.hour, date.minute])
        temp = [str(x) for x in temp]
        self.time = "_".join(temp)
        self.GENERAL_PATH = (
            f"{logging_path}/results/{str(date.year)}_{str(date.month)}_{str(date.day)}"
        )
        self.RESULT_PATH = f"{self.GENERAL_PATH}/results_{self.time}"
        self.PERFORMANCE_PATH = f"{self.GENERAL_PATH}/performance_results"

        ## create directories
        os.makedirs(self.RESULT_PATH, exist_ok=True)
        os.makedirs(f"{self.RESULT_PATH}/plots/summary", exist_ok=True)
        os.makedirs(f"{self.RESULT_PATH}/plots/pmv_categories", exist_ok=True)
        os.makedirs(f"{self.RESULT_PATH}/experiments_csv", exist_ok=True)
        os.makedirs(f"{self.RESULT_PATH}/model_weights", exist_ok=True)

    def _plot(
        self,
        epsilons: List[float],
        losses: List[float],
        tair: List[float],
        actions: List[float],
        pmv: List[float],
        qheat: List[float],
        rewards: List[float],
        occ: List[float],
        plot_filename: str,
        title: str,
    ):

        epsilons = np.array(epsilons)
        losses = np.array(losses)
        tair = np.array(tair)
        actions = np.array(actions)
        pmv = np.array(pmv)
        qheat = np.array(qheat)
        rewards = np.array(rewards)
        occ = np.array(occ)

        # Plotting the summary of simulation
        fig = make_subplots(
            rows=8,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            specs=[
                [{"secondary_y": False}],
                [{"secondary_y": False}],
                [{"secondary_y": False}],
                [{"secondary_y": True}],
                [{"secondary_y": True}],
                [{"secondary_y": False}],
                [{"secondary_y": True}],
                [{"secondary_y": True}],
            ],
        )

        iterations = len(tair)
        t = np.linspace(0.0, iterations - 1, iterations)
        # Add traces
        fig.add_trace(
            go.Scatter(
                name="Tair(state)",
                x=t,
                y=tair.flatten(),
                mode="lines",
                line=dict(width=1, color="cyan"),
            ),
            row=1,
            col=1,
        )
        # fig.add_trace(go.Scatter(name='Tair_avg', x=t, y=pd.Series(tair.flatten()).rolling(window=60).mean(), mode='lines',
        #              line=dict(width=2, color='blue')), row=1, col=1)

        fig.add_trace(
            go.Scatter(
                name="Tset(action)",
                x=t,
                y=actions.flatten(),
                mode="lines",
                line=dict(width=1, color="fuchsia"),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                name="Tset_avg",
                x=t,
                y=pd.Series(actions.flatten()).rolling(window=24).mean(),
                mode="lines",
                line=dict(width=2, color="purple"),
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                name="Pmv",
                x=t,
                y=pmv.flatten(),
                mode="lines",
                line=dict(width=1, color="gold"),
            ),
            row=3,
            col=1,
        )
        # fig.add_trace(go.Scatter(name='Pmv_avg', x=t, y=pd.Series(pmv.flatten()).rolling(window=60).mean(), mode='lines',
        #              line=dict(width=2, color='darkorange')), row=3, col=1)

        fig.add_trace(
            go.Scatter(
                name="Heating",
                x=t,
                y=qheat.flatten(),
                mode="lines",
                line=dict(width=1, color="red"),
            ),
            row=4,
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                name="Heating_cumulative",
                x=t,
                y=np.cumsum(qheat.flatten()),
                mode="lines",
                line=dict(width=2, color="darkred"),
            ),
            row=4,
            col=1,
            secondary_y=True,
        )

        fig.add_trace(
            go.Scatter(
                name="Reward",
                x=t,
                y=rewards.flatten(),
                mode="lines",
                line=dict(width=1, color="lime"),
            ),
            row=5,
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                name="Reward_cum",
                x=t,
                y=np.cumsum(rewards.flatten()),
                mode="lines",
                line=dict(width=2, color="darkgreen"),
            ),
            row=5,
            col=1,
            secondary_y=True,
        )

        fig.add_trace(
            go.Scatter(
                name="Occupancy",
                x=t,
                y=occ.flatten(),
                mode="lines",
                line=dict(width=1, color="black"),
            ),
            row=6,
            col=1,
        )
        ## training part

        fig.add_trace(
            go.Scatter(
                name="Epsilons",
                x=t,
                y=epsilons.flatten(),
                mode="lines",
                line=dict(width=1, color="blue"),
            ),
            row=7,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                name="Training Log Loss",
                x=t,
                y=losses.flatten(),
                mode="lines",
                line=dict(width=1, color="darkblue"),
            ),
            row=8,
            col=1,
        )

        # Set x-axis title
        fig.update_xaxes(title_text="Timestep (-)", row=6, col=1)
        # Set y-axes titles
        fig.update_yaxes(title_text="<b>Tair</b> (°C)", range=[10, 24], row=1, col=1)
        fig.update_yaxes(title_text="<b>Tset</b> (°C)", range=[14, 22], row=2, col=1)
        fig.update_yaxes(title_text="<b>PMV</b> (-)", row=3, col=1)
        fig.update_yaxes(
            title_text="<b>Heat Power</b> (kJ/hr)", row=4, col=1, secondary_y=False
        )
        fig.update_yaxes(
            title_text="<b>Heat Energy</b> (kJ)", row=4, col=1, secondary_y=True
        )
        fig.update_yaxes(
            title_text="<b>Reward</b> (-)",
            row=5,
            col=1,
            range=[-5, 5],
            secondary_y=False,
        )
        fig.update_yaxes(
            title_text="<b>Tot Reward</b> (-)", row=5, col=1, secondary_y=True
        )
        fig.update_yaxes(title_text="<b>Occ</b> (-)", row=6, col=1)
        fig.update_yaxes(title_text="<b>Epsilon</b> (-)", row=7, col=1)
        fig.update_yaxes(title_text="<b>Log Loss</b> (-)", row=8, col=1, type="log")

        fig.update_xaxes(nticks=50)
        fig.update_layout(
            template="plotly_white",
            font=dict(family="Courier New, monospace", size=10),
            legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1),
        )

        fig.update_layout(title_text=title)

        pyo.plot(fig, filename=plot_filename)

    def pmv_percentages(self, pmv: np.ndarray):

        temp = np.array(pmv)
        intervals = []

        length = 8
        lower = -2
        step = 0.5

        ranges = np.zeros(length)

        for i in range(length):

            if i == 0:
                ranges[i] = (temp < lower).sum()
                interval = f"[-inf,{lower}]"
                intervals.append(interval)

            elif i == 7:
                upper = (i - 1) * step + lower
                ranges[i] = (upper <= temp).sum()
                interval = f"[{upper},inf]"
                intervals.append(interval)

            else:
                lower_1 = lower + (i - 1) * step
                upper_1 = lower + (i) * step
                ranges[i] = ((lower_1 <= temp) & (temp < upper_1)).sum()
                interval = f"[{lower_1},{upper_1}]"
                intervals.append(interval)

        ranges = ranges / ranges.sum()

        # assign data
        data = pd.DataFrame({"intervals": intervals, "ranges": ranges})

        return data

    def plot_pmv_percentages(
        self, pmv: np.ndarray, savepath: str, title: str, plot_title: str
    ):

        data = self.pmv_percentages(pmv)
        # compute percentage of each format
        percentage = []
        for i in range(data.shape[0]):
            pct = data.ranges[i] * 100
            percentage.append(round(pct, 2))
        data["Percentage"] = percentage

        f, a = plt.subplots(1, 1, figsize=(15, 7))
        colors_list = [
            "darkred",
            "coral",
            "coral",
            "seagreen",
            "lime",
            "seagreen",
            "coral",
            "darkred",
        ]

        graph = plt.bar(x=data.intervals, height=data.ranges, color=colors_list)

        plt.xlabel("PMV value interval")
        plt.ylabel("Percentage of hours in interval")
        plt.title(plot_title)

        i = 0
        for p in graph:
            width = p.get_width()
            height = p.get_height()
            x, y = p.get_xy()
            plt.text(
                x + width / 2,
                y + height * 1.01,
                str(data.Percentage[i]) + "%",
                ha="center",
                weight="bold",
            )
            i += 1

        plt.savefig(f"{savepath}/{title}.png", dpi=400)

    def plot_and_logging(
        self,
        episode_num,
        tair,
        actions,
        pmv,
        qheat,
        rewards,
        occ,
        losses,
        epsilons,
        agent,
        is_summary=False,
    ):

        suffix = episode_num + 1
        if is_summary:
            suffix = "summary"

        plot_filename = f"{self.RESULT_PATH}/plots/summary/results_{suffix}.html"
        plot_title = (
            f"Episode Number {suffix} with α = {agent.env.alpha} and β = {agent.env.beta}"
            if not (is_summary)
            else f"Summary with α = {agent.env.alpha} and β = {agent.env.beta}"
        )

        self._plot(
            epsilons,
            losses,
            tair,
            actions,
            pmv,
            qheat,
            rewards,
            occ,
            plot_filename=plot_filename,
            title=plot_title,
        )

        ## padding losses and epsilons so that they fit into dataframe

        len_difference = len(tair) - len(losses)
        pad_losses = [0 for i in range(len_difference)]
        pad_epsilon = [epsilons[0] for i in range(len_difference)]
        losses = pad_losses + losses
        epsilons = pad_epsilon + epsilons
        data = pd.DataFrame(
            {
                "loss": losses,
                "epsilon": epsilons,
                "tair": tair,
                "action": actions,
                "pmv": pmv,
                "qheat": qheat,
                "reward": rewards,
                "occ": occ,
            }
        )

        data["reward"] = data.reward.apply(lambda x: float(x[0]))
        data.to_csv(
            f"{self.RESULT_PATH}/experiments_csv/experiments_results_{suffix}.csv"
        )

        ## PLOTTING PMV INTERVALS
        plot_title = f"Number of hours the algorithm spent in different PMV intervals"
        plot_title = plot_title + " (Total)" if is_summary else ""

        ## only keep pmv when occupancy > 0
        self.plot_pmv_percentages(
            pmv=np.array(data[data.occ > 0]["pmv"]),
            savepath=f"{self.RESULT_PATH}/plots/pmv_categories",
            title=f"PMV_Categories_{suffix}",
            plot_title=plot_title,
        )

        ## saving parameters of environment

        f = open(f"{self.RESULT_PATH}/env_params_{self.time}.json", "w")
        log_dict = agent.log_dict()

        ## concatenate the two dicts
        log_dict = {**log_dict, **agent.env.log_dict()}

        log_dict["num_episodes"] = self.num_episodes
        log_dict["num_iterations"] = self.num_iterations

        log_dict["final_reward"] = np.array(rewards).cumsum()[-1]
        log_dict["final_cumulative_heating"] = np.array(qheat).cumsum()[-1]

        pmv_df = self.pmv_percentages(np.array(pmv)).to_dict()

        log_dict["pmvs"] = {
            pmv_df["intervals"][i]: pmv_df["ranges"][i]
            for i in pmv_df["intervals"].keys()
        }

        f.write(json.dumps(log_dict, indent=True))
        f.close()

        agent.save(
            directory=f"{self.RESULT_PATH}/model_weights", filename=f"torch_ep_{suffix}"
        )

        return data

    def log_performance_pipeline(self, results: Dict[str, Any]):

        os.makedirs(self.PERFORMANCE_PATH, exist_ok=True)

        filename = f"performance_results_{self.time}"

        with open(f"{self.PERFORMANCE_PATH}\{filename}", "w") as f:
            f.write(json.dumps(results, indent=True))

