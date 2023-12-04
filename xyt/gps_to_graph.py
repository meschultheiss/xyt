import pandas as pd
import networkx as nx
import random
import numpy as np
from matplotlib import pyplot as plt
import gif
import geopandas as gpd
import multiprocessing as mp
import os
from pathos.multiprocessing import ProcessingPool as Pool


class GPStoGraph:
    def __init__(self) -> None:
        pd.set_option("display.max_columns", 999)

    @staticmethod
    def verify_columns(df: pd.DataFrame, expected_columns: list[str]) -> None:
        """
        Checks for missing columns in a DataFrame.

        Args:
            - df (pd.DataFrame): DataFrame to check columns.
            - expected_columns (list[str]): List of expected column names.

        Raises:
            - ValueError: If columns in `expected_columns` are missing in the DataFrame.
        """

        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"Columns missing in DataFrame: {', '.join(missing_columns)}"
            )

    @staticmethod
    def _preprocess_nodes(nodes) -> pd.DataFrame:
        """
        Preprocesses GPS data by removing consecutive duplicate location entries.

        Args:
            - nodes (pd.DataFrame): DataFrame containing GPS data.

        Returns:
            - pd.DataFrame: Preprocessed GPS data.
        """

        # Create a copy of the original DataFrame
        nodes_ = nodes.reset_index(drop=True).copy()

        # Shift the location_id to create a 'location_id_next' column
        nodes_["location_id_next"] = nodes_.groupby("user_id_day")["location_id"].shift(
            -1
        )

        # Identify consecutive duplicate locations and remove them
        index_to_drop = []
        i = 0
        for i_0 in nodes_.loc[
            nodes_.location_id == nodes_.groupby("user_id_day")["location_id"].shift(-1)
        ].index:
            if i_0 <= i:
                index_to_drop.append(i_0)
                continue
            else:
                i = i_0
                location_id = nodes_.loc[i, "location_id"]

                while location_id == nodes_.loc[i, "location_id_next"]:
                    i += 1
                    index_to_drop.append(i)

                # Update 'finished_at' and 'location_id_next' for the first occurrence
                nodes_.loc[i_0, "finished_at"] = nodes_.loc[i, "finished_at"]
                nodes_.loc[i_0, "location_id_next"] = nodes_.loc[i, "location_id_next"]

        # Drop the identified duplicate locations and reset index
        nodes_.drop(index_to_drop, inplace=True)
        nodes_.reset_index(drop=True, inplace=True)

        # Create an 'edges' column with tuples of location_id and location_id_next
        nodes_["edges"] = list(zip(nodes_.location_id, nodes_.location_id_next))

        # Create a GeoDataFrame and parse geometry to a tuple of coordinates in the right projection
        nodes_ = gpd.GeoDataFrame(
            nodes_, geometry=gpd.points_from_xy(nodes_.lon, nodes_.lat), crs="EPSG:4326"
        )
        nodes_["coordinates"] = list(zip(nodes_.geometry.x, nodes_.geometry.y))

        return nodes_

    @staticmethod
    def _get_motifs(nodes) -> pd.DataFrame:
        """
        Computes motifs based on GPS data and constructs corresponding graphs.

        Args:
            - nodes (pd.DataFrame): DataFrame containing GPS data.

        Returns:
            - pd.DataFrame: DataFrame with computed motifs and corresponding graphs.
        """

        mtfs = pd.DataFrame(index=nodes.user_id_day.unique())
        mtfs["graph"] = np.nan
        mtfs["graph_flat"] = np.nan

        mtfs["graph"] = mtfs["graph"].astype("object")
        mtfs["graph_flat"] = mtfs["graph_flat"].astype("object")

        for user_id_ in mtfs.index:
            nodes_ = nodes.loc[nodes.user_id_day == user_id_]

            try:
                attributes = nodes_.drop_duplicates(
                    subset=["user_id_day", "location_id", "started_at", "finished_at"]
                )[
                    [
                        "user_id_day",
                        "location_id",
                        "coordinates",
                        "started_at",
                        "finished_at",
                    ]
                ]
                attributes.set_index("location_id", inplace=True)

                attributes_time = attributes.groupby(by=["location_id"]).aggregate(
                    {"started_at": list, "finished_at": list}
                )
                attributes_time = (
                    attributes_time.groupby(level=0)
                    .apply(
                        lambda attributes_time: attributes_time.xs(
                            attributes_time.name
                        ).to_dict()
                    )
                    .to_dict()
                )

                attributes_location = pd.DataFrame(
                    attributes["coordinates"].drop_duplicates()
                ).T.to_dict()

                G = nx.DiGraph()
                G.add_edges_from(nodes_.edges[:-1].to_list())

                nx.set_node_attributes(G, attributes_time, name="time")
                nx.set_node_attributes(G, attributes_location)

                mtfs.at[user_id_, "graph"] = G
                mtfs.at[user_id_, "graph_flat"] = (
                    nx.to_numpy_array(G).flatten().tolist()
                )

            except:
                print("Exception raised for " + str(user_id_))
                continue

        return mtfs.reset_index(drop=False)

    def get_graphs(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Extracts motifs and graphs from GPS data.

        Args:
            - df (pd.DataFrame): DataFrame containing GPS data.
            - verbose (bool): Verbosity for multiprocessing (default is True).

        Returns:
            - pd.DataFrame: DataFrame with extracted motifs and graphs.
        """

        col_to_use = [
            "user_id",
            "user_id_day",
            "location_id",
            "started_at",
            "finished_at",
            "lon",
            "lat",
            "home_location_id",
        ]

        self.verify_columns(df, col_to_use)

        nodes = df[col_to_use].copy()

        nodes = nodes.sort_values(by=["user_id_day", "started_at"])
        nodes = nodes.drop_duplicates(
            subset=["user_id_day", "location_id", "started_at"]
        )
        nodes.reset_index(inplace=True, drop=True)

        mtfs = pd.DataFrame(index=nodes.user_id_day.unique())
        mtfs["mtf_loc"] = np.nan
        mtfs["mtf_loc"] = mtfs["mtf_loc"].astype("object")

        nodes_ = self._preprocess_nodes(nodes)

        cores = mp.cpu_count() - 1

        if verbose:
            print("Multiprocessing is launched with %s cores in parallel" % cores)
            print("...")

        # split the df in as many array as the machine has cores
        user_ids = np.array_split(nodes_.user_id_day.unique(), cores, axis=0)
        nodes_split = []
        for u in user_ids:
            nodes_split.append(nodes_.loc[nodes_.user_id_day.isin(u.tolist())])

        # create the multiprocessing pool
        pool = Pool(cores)

        # process the DataFrame by mapping function to each df across the pool
        df_out = np.vstack(pool.map(self._get_motifs, nodes_split))

        # return the df
        mtf = pd.DataFrame(
            df_out, columns=["user_id_day", "DiGraph_motif", "motif_flat"]
        )

        # close down the pool and join
        pool.close()
        pool.join()
        pool.clear()

        mtf.set_index("user_id_day", inplace=True)
        mtf["user_id"] = mtf.index.str[:5]

        if verbose:
            print("Job done !")

        return mtf

    @staticmethod
    @gif.frame
    def _plot_graph(mtf: pd.DataFrame, user_id_: str, i: int) -> plt.figure:
        """
        Generates a plot of the graph motif for a specific user.

        Args:
            - mtf (pd.DataFrame): DataFrame containing motif data.
            - user_id_ (str): User identifier.
            - i (int): Index of the user's data.

        Returns:
            - plt.figure: Figure with the plot of the graph motif.
        """

        G_all = nx.compose_all(
            mtf.loc[mtf.user_id == user_id_, "DiGraph_motif"].values.tolist()
        )

        user_id_day_ = mtf.loc[mtf.user_id == user_id_].index[i]

        G1 = mtf.loc[user_id_day_, "DiGraph_motif"]

        f = plt.figure(
            frameon=False, figsize=(10, 10), dpi=100
        )  # figsize=(10,10),dpi=50,

        # plot the backgroud graph
        pos = nx.get_node_attributes(G_all, "coordinates")
        nx.draw_networkx_edges(
            G_all,
            pos,
            edge_color="black",
            alpha=0.20,
            arrowstyle="->",
            arrowsize=15,
            width=1,
            connectionstyle="arc3,rad=+0.15",
        )  # wedge,shrink_factor=0.5
        nx.draw_networkx_nodes(
            G_all,
            pos,
            node_color="black",
            alpha=0.20,
            node_size=70,
            nodelist=list(G_all.nodes())[1:],
        )
        nx.draw_networkx_nodes(
            G_all,
            pos,
            node_color="#dbcc9c",
            alpha=0.90,
            node_size=100,
            nodelist=list(G_all.nodes())[:1],
        )

        # plot the new daily graph
        pos = nx.get_node_attributes(G1, "coordinates")
        nx.draw_networkx_edges(
            G1,
            pos,
            edge_color="black",
            alpha=0.90,
            arrowstyle="->",
            arrowsize=15,
            width=2,
            connectionstyle="arc3,rad=+0.15",
        )  # wedge,shrink_factor=0.5
        nx.draw_networkx_nodes(
            G1,
            pos,
            node_color="black",
            alpha=0.90,
            node_size=100,
            nodelist=list(G1.nodes())[1:],
        )

        return f

    def plot_graph(self, mtf: pd.DataFrame, path="") -> None:
        """
        Creates GIFs displaying the graph motif for multiple users.

        Args:
            - mtf (pd.DataFrame): DataFrame containing motif data.
            - path (str): Path to save the GIFs (default is '').
        """

        for user_id_ in random.choices(mtf.user_id.unique(), k=2):
            frames = []
            for i in range(0, len(mtf.loc[mtf.user_id == user_id_].index)):
                frames.append(self._plot_graph(mtf, user_id_, i))
            file_name = "%s.gif" % user_id_
            save_path = os.path.join(path, file_name)
            gif.save(frames, save_path, duration=500)

    def plot_motif(self, mtf: pd.DataFrame) -> None:
        """
        Plots motifs and their frequency distribution.

        Args:
            - mtf (pd.DataFrame): DataFrame containing motif data.
        """

        mtf_ = mtf.copy()
        motif_to_keep = 9
        mtf_["motif_id"] = 99

        for i_, mtf in enumerate(mtf_.motif_flat.value_counts().index[:motif_to_keep]):
            mtf_.loc[mtf_.motif_flat.apply(lambda x: x == mtf), "motif_id"] = i_ + 1

        mtf_list = mtf_.groupby("motif_id")["motif_flat"].agg(
            motif_flat=pd.Series.mode, count=pd.Series.count
        )
        mtf_list["count"] = mtf_list["count"] / mtf_list["count"].sum()

        f = plt.figure(
            frameon=False,
            figsize=(20, 2),
        )  # figsize=(10,10),dpi=50,

        for sublot, id_ in enumerate(mtf_list.index[:-1]):
            axis = f.add_subplot(
                1,
                9,
                sublot + 1,
                xticks=[],
                yticks=[],
                frame_on=False,
                title="MOTIF %s\n(%s%%)"
                % (id_, round(mtf_list.loc[id_, "count"] * 100, 1)),
            )  # title='motif %s (%d%%)'%(counter,expl_)
            motif = mtf_list.loc[id_, "motif_flat"]
            dim = np.sqrt(len(motif)).astype(int)
            motif_arr = np.asarray(motif).reshape((dim, dim))
            G = nx.DiGraph(motif_arr)
            pos = nx.circular_layout(G)
            nx.draw_networkx_edges(
                G,
                pos,
                edge_color="black",
                alpha=0.80,
                arrowstyle="->",
                arrowsize=15,
                width=1,
                connectionstyle="arc3,rad=+0.15",
            )  # wedge,shrink_factor=0.5
            nx.draw_networkx_nodes(
                G, pos, node_color="black", node_size=70, nodelist=list(range(1, dim))
            )
            nx.draw_networkx_nodes(
                G, pos, node_color="#dbcc9c", node_size=80, nodelist=[0]
            )

        plt.show()

    @staticmethod
    def _get_mtf_sequence(mtf_data, user_id, pad=31) -> np.array:
        """
        Retrieves a motif sequence for a specific user.

        Args:
            - mtf_data: DataFrame containing motif data.
            - user_id (str): User identifier.
            - pad (int): Length of the output sequence (default is 31).

        Returns:
            - np.array: Padded motif sequence for the specified user.
        """

        # Extract motif data for the specified user
        sequence = (
            mtf_data.loc[mtf_data["user_id"] == user_id, ["motif_id", "date"]]
            .set_index("date")
            .sort_index()
        )

        # Align the first observation on Monday
        pad_before = 0
        if len(sequence) > 0:
            pad_before = sequence.index.dayofweek[0]  # 0 is Monday, 1 is Tuesday, etc.

        # Resample the time series to have continuous days, fill in missing values with 0
        sequence = (
            sequence.resample("1D")
            .mean()
            .fillna(0)
            .astype(int)
            .reset_index(drop=True)
            .T
        )

        # Make all the series the same length by padding after
        pad_after_ = pad - sequence.shape[1] - pad_before
        pad_after = max(0, pad_after_)

        if sequence.shape[1] > pad:
            return np.pad(
                sequence.values.flatten(), pad_width=[pad_before, 0], mode="constant"
            )[:pad]
        else:
            return np.pad(
                sequence.values.flatten(),
                pad_width=[pad_before, pad_after],
                mode="constant",
            )[:pad]
        # Example usage:
        # motif_sequence = get_mtf_sequence(treatment_data, mtf_data, phase=1, user_id='your_user_id')

    def motif_sequence(self, mtf: pd.DataFrame, n_cols: int = 60) -> pd.DataFrame:
        """
        Generates motif sequences for users based on motif data.

        Args:
            - mtf (pd.DataFrame): DataFrame containing motif data.
            - n_cols (int): Number of columns for the motif sequence (default is 60).

        Returns:
            - pd.DataFrame: DataFrame with generated motif sequences for users.
        """

        mtf_ = mtf.copy()
        motif_to_keep = 9
        mtf_["motif_id"] = 99

        for i_, mtf in enumerate(mtf_.motif_flat.value_counts().index[:motif_to_keep]):
            mtf_.loc[mtf_.motif_flat.apply(lambda x: x == mtf), "motif_id"] = i_ + 1

        mtf_ = mtf_.reset_index()
        mtf_["date"] = mtf_["user_id_day"].str[-8:]
        mtf_["date"] = pd.to_datetime(mtf_["date"], format="%Y%m%d")

        mtf_seq = pd.DataFrame(index=mtf_.user_id.unique(), columns=range(n_cols))

        for usrs in mtf_.user_id.unique():
            try:
                mtf_seq.loc[usrs] = self._get_mtf_sequence(
                    mtf_, user_id=usrs, pad=n_cols
                ).tolist()
            except:
                print("Exception raised on user " + usrs + " / phase 1")
                continue

        return mtf_seq
