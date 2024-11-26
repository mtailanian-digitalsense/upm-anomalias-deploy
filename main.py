import logging
from collections import deque
from datetime import datetime

from dash.dependencies import Input, Output

from src.algorithms.correlation import CorrelationDetector
from src.dataloader import DataLoader
from src.interface import get_app, get_signal_options


class DashHandler(logging.Handler):
    def __init__(self, capacity=100):
        super().__init__()
        self.log_messages = deque(maxlen=capacity)

    def emit(self, record):
        self.log_messages.append(self.format(record))

    def get_log(self):
        return "\n".join(list(self.log_messages)[::-1])

# Configure the logger
dash_handler = DashHandler()
console_handler = logging.StreamHandler()
logging.basicConfig(level=logging.INFO, handlers=[console_handler, dash_handler])
logger = logging.getLogger(__name__)

def main():
    signal_options = get_signal_options()
    app = get_app(signal_options)

    data_loader = DataLoader(root_dir="../data/flow-meters", db_path="database.yaml")

    detectors = {}
    for signal in signal_options:
        logger.info(f"Initializing (training) detector for {signal['value']}")
        detector = CorrelationDetector(config_path="configs/correlation.yaml")
        train_df, test_df = data_loader.load(signal['value'])
        detector.fit(train_df)
        detectors[signal['value']] = detector

    @app.callback(
        Output('live-update-graph', 'figure'),
        Output('log-console', 'value'),
        Input('interval-component', 'n_intervals'),
        Input('signal-dropdown', 'value')
    )
    def update(n, selected_signal):

        # Load the data
        df_train, df_test = data_loader.load(selected_signal)

        results = detectors[selected_signal].predict(df_test)

        last_update = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"Plot updated at {last_update} (n={n})")

        figure = detectors[selected_signal].plot(df_test, results, title=f"{selected_signal} ({last_update})", show=False)
        return figure, dash_handler.get_log()

    # Run the app on the default port
    app.run_server(debug=True, use_reloader=False)

if __name__ == '__main__':
    main()