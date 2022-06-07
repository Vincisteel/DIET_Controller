from abc import ABCMeta, abstractmethod
from re import S
from typing import Dict, List, Tuple, Any
import numpy as np
from datetime import datetime
import json
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
import os
import pandas as pd
from agent.Agent import Agent


class Logger(metaclass=ABCMeta):
    @property
    @abstractmethod
    def RESULT_PATH(self):
        ...

    @property
    @abstractmethod
    def PERFORMANCE_PATH(self):
        ...

    @abstractmethod
    def plot_and_logging(
        self, summary_df: pd.DataFrame, agent: Agent, episode_num: int, is_summary: bool
    ) -> None:
        pass

    @staticmethod
    def pmv_percentages(pmv: np.ndarray):

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

    @staticmethod
    def plot_pmv_percentages(
        pmv: np.ndarray, savepath: str, title: str, plot_title: str
    ):

        data = Logger.pmv_percentages(pmv)
        # compute percentage of each format
        percentage = []
        for i in range(data.shape[0]):
            pct = data.ranges[i] * 100
            percentage.append(round(pct, 2))

        data["Percentage"] = percentage

        _ = plt.subplots(1, 1, figsize=(15, 7))
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


class SimpleLogger(Logger):
    def __init__(
        self, logging_path: str, agent_name: str, num_episodes: int, num_iterations: int
    ):

        self.num_episodes = num_episodes
        self.num_iterations = num_iterations
        ## get current time to set up logging directory
        date = datetime.now()
        temp = list([date.year, date.month, date.day, date.hour, date.minute])
        temp = [str(x) for x in temp]
        self.time = "_".join(temp)
        self.GENERAL_PATH = f""

        self.TIME_PATH = f"{str(date.year)}_{str(date.month)}_{str(date.day)}"
        self._RESULT_PATH = (
            f"{logging_path}/results/{agent_name}/{self.TIME_PATH}/results_{self.time}"
        )
        self._PERFORMANCE_PATH = (
            f"{logging_path}/results/{agent_name}/performance_results"
        )

        ## create directories
        os.makedirs(self.RESULT_PATH, exist_ok=True)
        os.makedirs(f"{self.RESULT_PATH}/plots/summary", exist_ok=True)
        os.makedirs(f"{self.RESULT_PATH}/plots/pmv_categories", exist_ok=True)
        os.makedirs(f"{self.RESULT_PATH}/experiments_csv", exist_ok=True)
        os.makedirs(f"{self.RESULT_PATH}/model_weights", exist_ok=True)

    @property
    def RESULT_PATH(self):
        return self._RESULT_PATH

    @property
    def PERFORMANCE_PATH(self):
        return self._PERFORMANCE_PATH

    def _plot(
        self,
        summary_df: pd.DataFrame,
        opts: Dict[str, Dict[str, Any]],
        plot_filename: str,
        title: str,
    ):

        # moving average indicates whether we want to compute for a given column
        # moving_averages = {key: val for key, val in sorted(secondary_ys.items(), key = lambda ele: ele[0])}

        num_rows = len(summary_df.columns)

        # Plotting the summary of simulation
        fig = make_subplots(
            rows=num_rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            specs=[
                [{"secondary_y": (res["secondary_y"] is not None)}]
                for res in opts.values()
            ],
        )

        ## please add more if the plot gets bigger
        colors = [
            ("cyan", "dark cyan"),
            ("fuchsia", "purple"),
            ("gold", "darkgoldenrod"),
            ("red", "darkred"),
            ("lime", "darkgreen"),
            ("turquoise", "darkturquoise"),
            ("slateblue", "darkslateblue"),
            ("royalblue", "darkblue"),
        ]

        iterations = len(summary_df)
        t = np.linspace(0.0, iterations - 1, iterations)
        # Set x-axis title
        fig.update_xaxes(title_text="Timestep (-)", row=num_rows, col=1)

        for i, column_name in enumerate(summary_df.columns):
            # Add traces
            fig.add_trace(
                go.Scatter(
                    name=column_name,
                    x=t,
                    y=np.array(summary_df[column_name]),
                    mode="lines",
                    line=dict(width=1, color=colors[i][0]),
                ),
                row=i + 1,
                col=1,
                secondary_y=False,
            )

            # set y axis title
            unit = opts[column_name]["unit"]
            range = opts[column_name]["range"]

            if range is not None:
                fig.update_yaxes(
                    title_text=f"<b>{column_name}</b> { unit}",
                    range=range,
                    row=i + 1,
                    col=1,
                )
            else:
                fig.update_yaxes(
                    title_text=f"<b>{column_name}</b> { unit}", row=i + 1, col=1
                )

            if opts[column_name]["secondary_y"] is not None:
                suffix = ""
                arr = summary_df[column_name]

                if opts[column_name]["secondary_y"] == "moving_average":
                    suffix = "avg"
                    arr = np.array(arr.rolling(24).mean())

                elif opts[column_name]["secondary_y"] == "cumulative":
                    suffix = "cumulative"
                    arr = np.cumsum(np.array(arr))

                fig.add_trace(
                    go.Scatter(
                        name=f"{column_name}_{suffix}",
                        x=t,
                        y=arr,
                        mode="lines",
                        line=dict(width=2, color=colors[i][1]),
                    ),
                    row=i + 1,
                    col=1,
                    secondary_y=True,
                )

                if range is not None and (
                    opts[column_name]["secondary_y"] == "moving_average"
                ):
                    fig.update_yaxes(
                        title_text=f"<b>{column_name}_{suffix}</b> { unit}",
                        range=range,
                        row=i + 1,
                        col=1,
                        secondary_y=True,
                    )
                else:
                    fig.update_yaxes(
                        title_text=f"<b>{column_name}_{suffix}</b> { unit}",
                        row=i + 1,
                        col=1,
                        secondary_y=True,
                    )

        fig.update_xaxes(nticks=50)
        fig.update_layout(
            template="plotly_white",
            font=dict(family="Courier New, monospace", size=10),
            legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1),
        )

        fig.update_layout(title_text=title)

        pyo.plot(fig, filename=plot_filename)

    def log(
        self, summary_df: pd.DataFrame, suffix: str, is_summary: bool, agent: Agent
    ):

        summary_df.to_csv(
            f"{self.RESULT_PATH}/experiments_csv/experiments_results_{suffix}.csv"
        )

        ## saving parameters of environment

        f = open(f"{self.RESULT_PATH}/env_params_{self.time}.json", "w")
        log_dict = agent.log_dict()

        ## concatenate the two dicts
        log_dict = {**log_dict, **agent.env.log_dict()}

        log_dict["num_episodes"] = self.num_episodes
        log_dict["num_iterations"] = self.num_iterations

        log_dict["final_reward"] = np.array(summary_df["Reward"]).cumsum()[-1]
        log_dict["final_cumulative_heating"] = np.array(summary_df["Heating"]).cumsum()[
            -1
        ]

        pmv_df = self.pmv_percentages(np.array(summary_df["PMV"])).to_dict()

        log_dict["pmvs"] = {
            pmv_df["intervals"][i]: pmv_df["ranges"][i]
            for i in pmv_df["intervals"].keys()
        }

        f.write(json.dumps(log_dict, indent=True))
        f.close()

        agent.save(
            directory=f"{self.RESULT_PATH}/model_weights", filename=f"torch_ep_{suffix}"
        )

        return summary_df

    def plot_and_logging(
        self,
        summary_df: pd.DataFrame,
        agent: Agent,
        episode_num: int,
        is_summary: bool,
        opts: Dict[str, Dict[str, Any]],
    ):

        suffix = str(episode_num + 1)
        if is_summary:
            suffix = "summary"

        plot_filename = f"{self.RESULT_PATH}/plots/summary/results_{suffix}.html"
        plot_title = (
            f"Episode Number {suffix} with α = {agent.env.alpha} and β = {agent.env.beta}"
            if not (is_summary)
            else f"Summary with α = {agent.env.alpha} and β = {agent.env.beta}"
        )

        self._plot(
            summary_df=summary_df,
            plot_filename=plot_filename,
            title=plot_title,
            opts=opts,
        )

        ## PLOTTING PMV INTERVALS
        plot_title = f"Number of hours the algorithm spent in different PMV intervals"
        plot_title = plot_title + " (Total)" if is_summary else ""

        ## only keep pmv when occupancy > 0
        Logger.plot_pmv_percentages(
            pmv=np.array(summary_df[summary_df["Occ"] > 0]["PMV"]),
            savepath=f"{self.RESULT_PATH}/plots/pmv_categories",
            title=f"PMV_Categories_{suffix}",
            plot_title=plot_title,
        )

        self.log(
            summary_df=summary_df, suffix=suffix, is_summary=is_summary, agent=agent
        )

        return

    def log_performance_pipeline(self, results: Dict[str, Any]):

        os.makedirs(self.PERFORMANCE_PATH, exist_ok=True)

        filename = f"performance_results_{self.time}"

        with open(f"{self.PERFORMANCE_PATH}\{filename}", "w") as f:
            f.write(json.dumps(results, indent=True))
