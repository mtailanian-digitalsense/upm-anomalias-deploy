import pandas as pd
from datetime import timedelta
import numpy as np

from src.utils.logger_config import logger
from src.utils.correlation_computation import compute_rolling_correlation_pd


class DataLoader:
    def __init__(self, offline, flow_meters_info_csv_path, offline_root_dir="../data/flow-meters"):
        self.offline = offline
        self.offline_root_dir = offline_root_dir

        self.config = pd.read_csv(flow_meters_info_csv_path, index_col=0)
        self.config[pd.isna(self.config)] = None

    def load_train_data(self, signal_id: str):
        control_id = self.config.loc[signal_id]['ID_CONTROL']
        start = self.config.loc[signal_id]['TRAIN_STARTS']
        end = self.config.loc[signal_id]['TRAIN_ENDS']

        return self.load_data(signal_id, control_id, start, end)

    def load_new_data(self, signal_id, current_time, window_in_hours):
        control_id = self.config.loc[signal_id]['ID_CONTROL']

        start = current_time - timedelta(hours=window_in_hours)
        end = current_time

        return self.load_data(signal_id, control_id, start, end)

    def load_data(self, signal_id, control_id, start, end):
        if self.offline:
            series = self.offline_load_data(signal_id, start, end)
        else:
            series = self.query_data(signal_id, control_id, start, end)

        # Create a correlation column
        if (series is not None) and 'control' in series.columns:
            series = compute_rolling_correlation_pd(series)

        return series

    def query_data(self, signal_id: str, control_id: str, date_start, date_end):
        """
        Makes the query to the database
        :param signal_id:
        :param control_id:
        :param date_start:
        :param date_end:
        :return: A dataframe with the index as the timestamp and the columns 'signal' and 'control' (if available)
        """

        # TODO: implement

        return pd.DataFrame()

    def offline_load_data(self, signal_id: str, date_start, date_end):
        """
        Loads the training data from the offline files, present in the folder self.offline_root_dir.
        The files should already contain the columns 'signal' and 'control' (if available). Therefore, we do not need
        the control_id to load the data.

        :param signal_id: id of the signal to load
        :param date_start: training start date
        :param date_end: training end date
        :return: A dataframe with the index as the timestamp and the columns 'signal' and 'control' (if available)
        """
        file_path = f"{self.offline_root_dir}/all_{signal_id}.csv"
        try:
            series = pd.read_csv(file_path, index_col=0)
        except Exception as e:
            logger.error(f"Error loading data. Missing or unable to read {file_path}: {e}")
            return None

        # Convert the index to datetime
        series.index = pd.to_datetime(series.index)
        series = series.sort_index()

        # Check if the data is enough to cover the requested time range
        if series.index[-1] < pd.to_datetime(date_end):
            logger.error(f"Error loading data. Not enough data in {file_path} to cover the requested time range")
            return None

        series = series[series.index > date_start]
        series = series[series.index <= date_end]

        return series


if __name__ == '__main__':
    import plotly.graph_objects as go

    dataloader = DataLoader(offline=True, flow_meters_info_csv_path='train_data.csv')
    train_df = dataloader.load_train_data('330A0013FIC.PIDA.PV')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_df.index, y=train_df['signal'], mode='lines', name='signal'))
    for _ in range(10):
        new_data = dataloader.load_new_data('330A0013FIC.PIDA.PV', window_in_hours=72)
        fig.add_trace(go.Scatter(x=new_data.index, y=new_data['signal'], mode='lines', name='signal'))

    fig.show()