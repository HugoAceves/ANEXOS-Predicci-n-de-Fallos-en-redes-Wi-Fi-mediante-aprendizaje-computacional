import pandas as pd


FOLDER_PATH = "Data/"
E_FILE_NAME = "events2024"
M_FILE_NAME = "metrics202403"
FECHA = "2024-03-02 18:00:00"
PERIODO = "30"


def procesar_df(dataframe_e, dataframe_m, path, fecha_fija, period, specific_event = None, event_historico = None):
    """Esta función procesa los dataframes, los filtra, pivotea y une mediante un merge"""
    # Construcción de la ruta completa a los archivos
    e_file_path = str(path)+str(dataframe_e) + ".csv"
    m_file_path = str(path)+str(dataframe_m) + ".csv"


    # Carga de datos
    metrics = pd.read_csv(m_file_path)
    events = pd.read_csv(e_file_path)


    # Procesamiento de strings
    metrics['metric'] = metrics['metric'].apply(lambda x: "(Metric) " + x)
    events['classification'] = events['classification'].apply(lambda x: "(Event) " + x)
    events["firstSeen"] = pd.to_datetime(events["firstSeen"], format='ISO8601').dt.floor('s')
    metrics['timestamp'] = pd.to_datetime(metrics['timestamp'])


    if event_historico is not None:
        #events = events[events["classification"] == "(Event) " + event_historico]
        events, date_lim, metrics = filtrar_tiempo(date = fecha_fija, data = events, data_metrics=metrics, minutos= period, is_historical=event_historico)
        name_for_file = "Historico_MatrizCorrelacion_" + date_lim.replace(":", "-").replace(" ", "-")
    else:
        # Filtrar para el tiempo
        events, date_lim, metrics= filtrar_tiempo(date = fecha_fija, data = events, minutos= period, data_metrics= metrics)
        #metrics = metrics[metrics["timestamp"] == fecha_fija]
        partes = str(date_lim).split(" ")
        date_lim = partes[1]
        name_for_file = "MatrizCorrelacion_" + fecha_fija.replace(" ", "_").replace(":", "-") + "_" + date_lim.replace(":", "-")


    # Agrupar y eliminar columna de tiempo
    df_events_grouped = events.drop("firstSeen", axis = 1)
    df_metrics_grouped = metrics.drop("timestamp", axis = 1)
    df_events_grouped = df_events_grouped.groupby(["device_id", "classification"]).sum().reset_index()
    df_metrics_grouped = df_metrics_grouped.groupby(["device_id", "metric"]).mean().reset_index()


    # Filtrado y pivotado de datos
    events_filtered = df_events_grouped[df_events_grouped["classification"] != "(Event) Monitoring error"]
    if specific_event is not None:
        evento_string = "(Event) " + specific_event
        events_filtered = df_events_grouped[df_events_grouped["classification"] == evento_string]
        specific_event = specific_event.replace(" ", "-")
    pivoted_events = events_filtered.pivot_table(index="device_id", columns="classification", values="Total").reset_index()
    pivoted_metrics = df_metrics_grouped.pivot_table(index='device_id', columns='metric', values='average').reset_index()


    # Merge de los dataframes
    data_merge = pd.merge(pivoted_events, pivoted_metrics, on='device_id', how='outer') #how = "inner"
    data_merge['(Event) Access points down threshold exceeded'] = data_merge['(Event) Access points down threshold exceeded'].apply(
    lambda x: 1 if isinstance(x, (int, float)) and x > 0 else 0)


    return data_merge, name_for_file, date_lim, specific_event
def filtrar_tiempo(date, data, data_metrics, minutos = "60", is_historical = None):
    """Esta función filtra el dataframe según lo que se requiera, 30 o 60 minutos"""
    intervalo = pd.Timestamp(date) + pd.Timedelta(minutes=int(minutos))
    df = data[(data["firstSeen"] > pd.Timestamp(date) + pd.Timedelta(minutes=1)) & (data["firstSeen"] <= intervalo)]
    metrics = data_metrics[data_metrics["timestamp"] == date]
    if is_historical is not None:
        data["rounded_hour"] = data["firstSeen"].dt.floor("60T")
        df = data[(data["firstSeen"] >= data["rounded_hour"] + pd.Timedelta(minutes=1)) & (data["firstSeen"] <= data["rounded_hour"] + pd.Timedelta(minutes=int(minutos)))]
        metrics = data_metrics[data_metrics["timestamp"].isin(df["rounded_hour"])]# == df["rounded_hour"]]
        max_time = metrics["timestamp"].dt.date.max()
        min_time = metrics["timestamp"].dt.date.min()
        intervalo = "Eventos_" + str(min_time) + "_" + str(max_time) + "_Periodo " + PERIODO + " minutos"
        df = df.drop("rounded_hour", axis = 1)
    return df, intervalo, metrics




df_merged, files_name, end_date, evento_tag  = procesar_df(E_FILE_NAME, M_FILE_NAME, FOLDER_PATH, FECHA, PERIODO, event_historico="Access points down threshold exceeded", specific_event="Access points down threshold exceeded")#
df_merged = df_merged.apply(lambda col: col.fillna(col.mean()) if col.dtype.kind in 'fi' else col)
df_merged.to_csv('data.csv', index=False)
