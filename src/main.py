import argparse
import logging
from apscheduler.schedulers.background import BackgroundScheduler
import pandas as pd
from datetime import datetime, timedelta

from dash.dependencies import Input, Output

from src.algorithms.correlation import CorrelationDetector
from src.algorithms.kmeans_darts import KMeansDartsDetector
from src.dataloader import DataLoader
from src.dashboard import get_signal_options, app
from src.utils.logger_config import logger, dash_handler
from src.constants import TIME_SIMULATION_TRAIN_TEST_GAP_IN_DAYS, PREDICTION_UPDATE_INTERVAL_IN_MINUTES, \
    OFFLINE_PREDICTION_UPDATE_INTERVAL_IN_MINUTES

# Set APScheduler logging level to WARNING
logging.getLogger('apscheduler').setLevel(logging.WARNING)

# Global counter for the number of prediction updates
update_counter = 0
processed = {}


def schedule_predictions(detectors, data_loader, offline):
    scheduler = BackgroundScheduler()
    job_interval = OFFLINE_PREDICTION_UPDATE_INTERVAL_IN_MINUTES if offline else PREDICTION_UPDATE_INTERVAL_IN_MINUTES

    def update_predictions():
        global update_counter
        update_counter += 1
        for signal in detectors.keys():
            detector = detectors[signal]
            t = get_current_time(offline, data_loader.config.loc[signal]['TRAIN_ENDS'], update_counter)
            new_data = data_loader.load_new_data(signal, t, detector.needed_history_in_hours)
            results = detector.predict(new_data)
            # logger.info(f"Predictions updated at {t} for signal {signal}")
            if results and results['detection']:
                logger.warning(f"Anomaly detected at {results['time_index']} for signal {signal}")

    scheduler.add_job(update_predictions, 'interval', minutes=job_interval)
    scheduler.start()

def schedule_predictions_on_complete_series(detectors, data_loader):
    scheduler = BackgroundScheduler()
    job_interval = OFFLINE_PREDICTION_UPDATE_INTERVAL_IN_MINUTES

    def update_predictions():
        global processed
        for signal in detectors.keys():
            if processed[signal]:
                continue
            logger.info(f"Performing anomaly detection on signal {signal}")
            detector = detectors[signal]
            test_data = data_loader.load_test_data(signal)
            results = detector.predict_all(test_data)
            if results and results['detection'].any():
                logger.warning(f"Anomaly detected for signal {signal}")
            processed[signal] = True

    scheduler.add_job(update_predictions, 'interval', minutes=job_interval)
    scheduler.start()

def main(offline, flow_meters_info_csv_path):
    signal_options = get_signal_options()
    data_loader = DataLoader(offline, flow_meters_info_csv_path, offline_root_dir="../data/flow-meters")

    detectors = {}
    for signal in signal_options:
        logger.info(f"Initializing (training) detector for {signal['value']}")

        # detector = CorrelationDetector(config_path="configs/correlation.yaml")
        detector = KMeansDartsDetector(config_path="configs/kmeans_darts.yaml")

        train_df = data_loader.load_train_data(signal['value'])
        detector.fit(train_df)
        detectors[signal['value']] = detector

    @app.callback(
        Output('live-update-graph', 'figure'),
        Output('log-console', 'value'),
        Input('interval-component', 'n_intervals'),
        Input('signal-dropdown', 'value'),
    )
    def refresh_chart(n, selected_signal):
        detector = detectors[selected_signal]
        figure = detector.plot(title=f"{selected_signal} (refresh {n})", show=False)
        return figure, dash_handler.get_log()

    # schedule_predictions(detectors, data_loader, offline)

    global processed
    processed = {key: False for key in detectors.keys()}
    schedule_predictions_on_complete_series(detectors, data_loader)

    app.run_server(debug=True, use_reloader=False)


def get_current_time(offline, train_end_date, n):
    if not offline:
        return datetime.now()

    # Freeze time for offline mode
    return (
            pd.to_datetime(train_end_date) +
            timedelta(days=TIME_SIMULATION_TRAIN_TEST_GAP_IN_DAYS) +
            timedelta(minutes=n * PREDICTION_UPDATE_INTERVAL_IN_MINUTES)
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Dash application.')
    parser.add_argument('--offline', action='store_true', help='Run the application in offline mode')
    parser.add_argument('--flow_meters_info_csv_path', '-csv', type=str, default='train_data.csv',
                        help='Path to the flow meters info CSV file (default: train_data.csv)')
    args = parser.parse_args()

    print(args.offline)
    main(args.offline, args.flow_meters_info_csv_path)