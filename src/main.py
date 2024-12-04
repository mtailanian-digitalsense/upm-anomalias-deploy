import argparse
import logging
import threading
from apscheduler.schedulers.background import BackgroundScheduler
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

from dash.dependencies import Input, Output

from src.algorithms.correlation import CorrelationDetector
from src.algorithms.isolation_forest import IsolationForestDetector
from src.algorithms.knn import KNNDetector
from src.algorithms.kmeans_darts import KMeansDartsDetector
from src.algorithms.regressor_knn import KNNRegressorDetector
from src.algorithms.xgb_darts import XGBDartsDetector
# from src.algorithms.fft import FFTDetector
from src.algorithms.ensemble import EnsembleDetector

from src.dataloader import DataLoader
from src.dashboard import get_signal_options, app
from src.utils.logger_config import logger, dash_handler
from src.constants import RUN_PREDICTIONS_EVERY_H, SIMULATION_TIME_D

# Set APScheduler logging level to WARNING
logging.getLogger('apscheduler').setLevel(logging.ERROR)

# Global counter for the number of prediction updates
update_counter = {signal['value']: -1 for signal in get_signal_options()}


def initialize_and_train_detectors(signal_options, data_loader, detectors):
    for signal in signal_options:
        logger.info(f"Initializing (training) detector for {signal['value']}")

        # detector = CorrelationDetector(config_path="configs/correlation.yaml")
        # detector = IsolationForestDetector(config_path="configs/isolation_forest.yaml")
        # detector = KNNDetector(config_path="configs/knn.yaml")
        # detector = KMeansDartsDetector(config_path="configs/kmeans_darts.yaml")
        # detector = KNNRegressorDetector(config_path="configs/regressor_knn.yaml")
        # detector = XGBDartsDetector(config_path="configs/xgb_darts.yaml")
        # detector = FFTDetector(config_path="configs/fft.yaml")
        detector = EnsembleDetector(config_dir="configs")

        train_df = data_loader.load_train_data(signal['value'])
        detector.fit(train_df)
        detectors[signal['value']] = detector

        logger.info(f"Detector for {signal['value']} initialized and trained!")


def schedule_predictions(detectors, data_loader, offline):
    scheduler = BackgroundScheduler()
    job_interval_s = 0 if offline else RUN_PREDICTIONS_EVERY_H * 60 * 60

    def update_predictions():
        global update_counter
        for signal in detectors.keys():
            detector = detectors[signal]
            if detector is None:
                continue

            update_counter[signal] += 1
            t = get_current_time(data_loader.offline_last_timestamps[signal], update_counter[signal])
            new_data = data_loader.load_new_data(signal, t)
            results = detector.predict(new_data)
            print
            # logger.info(f"Predictions updated at {t} for signal {signal}")
            # if results and results['detection']:
            #     logger.warning(f"Anomaly detected at {results['time_index']} for signal {signal}")

    scheduler.add_job(update_predictions, 'interval', seconds=job_interval_s)
    scheduler.start()


def main(offline, flow_meters_info_csv_path):
    signal_options = get_signal_options()
    data_loader = DataLoader(offline, flow_meters_info_csv_path, offline_root_dir="../data/flow-meters")

    # Start a new thread to initialize and train detectors
    detectors = {signal['value']: None for signal in signal_options}
    detectors_thread = threading.Thread(target=initialize_and_train_detectors, args=(signal_options, data_loader, detectors))
    detectors_thread.start()

    @app.callback(
        Output('live-update-graph', 'figure'),
        Output('log-console', 'value'),
        Output('second-log-console', 'value'),
        Input('interval-component', 'n_intervals'),
        Input('signal-dropdown', 'value'),
    )
    def refresh_chart(n, selected_signal):
        detector = detectors[selected_signal]
        if not detector:
            return go.Figure(), dash_handler.get_log(), "Initializing detector..."

        figure = detector.plot(title=f"{selected_signal} (refresh {n})", show=False)

        # TODO: raise alarms when anomalies. Should create a new logger?
        anomalies_log = "\n".join(n * ["Anomaly detected!"])

        return figure, dash_handler.get_log(), anomalies_log

    schedule_predictions(detectors, data_loader, offline)
    app.run_server(debug=True, use_reloader=False)


def get_current_time(offline_start_timestamp, n):
    """
    Get the current time
    :param offline_start_timestamp: if None, we are in the case of online mode. Otherwise, we are in offline mode, and
        this time indicates the start of the simulation
    :param n: for the offline mode, the number of prediction updates that have been done so far
    :return: the current time
    """
    if not offline_start_timestamp:
        return datetime.now()

    # Freeze time for offline mode
    return (
            pd.to_datetime(offline_start_timestamp) -
            timedelta(days=SIMULATION_TIME_D) +
            timedelta(hours=n * RUN_PREDICTIONS_EVERY_H)
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Dash application.')
    parser.add_argument('--offline', action='store_true', help='Run the application in offline mode')
    parser.add_argument('--flow_meters_info_csv_path', '-csv', type=str, default='train_data.csv',
                        help='Path to the flow meters info CSV file (default: train_data.csv)')
    args = parser.parse_args()

    main(args.offline, args.flow_meters_info_csv_path)
