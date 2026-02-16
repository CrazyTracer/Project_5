import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error

# --- 1. ГЕНЕРАЦИЯ / ЗАГРУЗКА ДАННЫХ ---

# Для наглядности имитируем данные, похожие на FAOSTAT (ежемесячные продажи пшеницы)
def generate_demand_data(start_date='2015-01-01', end_date='2022-12-31'):
    dates = pd.date_range(start=start_date, end=end_date, freq='M')
    n = len(dates)
    
    # Генерируем тренд (рост продаж)
    trend = np.linspace(100, 150, n)
    
    # Генерируем сезонность (годовая пиковая нагрузка, например, осенью)
    seasonality = 20 * np.sin(2 * np.pi * np.arange(n) / 12)
    
    # Случайный шум
    noise = np.random.normal(0, 5, n)
    
    sales = trend + seasonality + noise
    # Убираем отрицательные значения
    sales = np.maximum(sales, 10)
    
    df = pd.DataFrame({'ds': dates, 'y': sales})
    return df

# В моём проекте здесь:
# df = pd.read_csv('faostat_wheat_sales.csv')
# df.columns = ['ds', 'y'] # Требование Prophet
# df['ds'] = pd.to_datetime(df['ds'])

df = generate_demand_data()

# Разделяем на Train и Test (последний год на тест)
train_size = len(df) - 12
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

print(f"Всего данных: {len(df)}")
print(f"Train: {len(train_df)}, Test: {len(test_df)}")

# --- 2. МЕТРИКИ ---

def calculate_smape(actual, forecast):
    """
    SMAPE (Symmetric Mean Absolute Percentage Error)
    Более устойчив к ошибкам, когда фактическое значение близко к 0.
    """
    numerator = np.abs(forecast - actual)
    denominator = (np.abs(actual) + np.abs(forecast)) / 2
    # Избегаем деления на ноль
    denominator = np.where(denominator == 0, 1e-10, denominator)
    return np.mean(numerator / denominator) * 100

# --- 3. BASELINE МОДЕЛЬ: ARIMA ---

print("\nОбучение ARIMA...")
# ARIMA(p,d,q). Подбираем параметры (1,1,1) как отправную точку.
# Для идеального подбора обычно используют auto_arima, но здесь сделаем вручную для наглядности.
arima_model = ARIMA(train_df['y'], order=(1, 1, 1))
arima_result = arima_model.fit()

# Прогнозируем на 12 шагов вперед
arima_forecast_result = arima_result.get_forecast(steps=12)
arima_pred = arima_forecast_result.predicted_mean

# Расчет метрик ARIMA
arima_smape = calculate_smape(test_df['y'].values, arima_pred.values)
arima_mape = mean_absolute_percentage_error(test_df['y'].values, arima_pred.values) * 100

print(f"ARIMA SMAPE: {arima_smape:.2f}%")
print(f"ARIMA MAPE:  {arima_mape:.2f}%")

# --- 4. МОДЕЛЬ PROPHET ---

print("\nОбучение Prophet...")
# Создаем и настраиваем модель
model = Prophet(
    yearly_seasonality=True,   # Годовая сезонность (важно для агросектора)
    weekly_seasonality=False,  # Данные помесячные, недельная не нужна
    daily_seasonality=False,
    seasonality_mode='additive' # или 'multiplicative' если амплитуда сезонности растет
)

# Добавляем возможности для учета праздников (если бы были данные по странам)
# model.add_country_holidays(country_name='US') 

model.fit(train_df)

# Создаем датафрейм для будущего (12 месяцев)
future_dates = model.make_future_dataframe(periods=12, freq='M')

# Делаем прогноз
prophet_forecast = model.predict(future_dates)

# Извлекаем только ту часть, которая относится к тестовому набору
prophet_pred_test = prophet_forecast.iloc[-12:]['yhat']

# Расчет метрик Prophet
prophet_smape = calculate_smape(test_df['y'].values, prophet_pred_test.values)
prophet_mape = mean_absolute_percentage_error(test_df['y'].values, prophet_pred_test.values) * 100

print(f"Prophet SMAPE: {prophet_smape:.2f}%")
print(f"Prophet MAPE:  {prophet_mape:.2f}%")

# --- 5. НЕЙРОСЕТЬ: MLP (Multi-Layer Perceptron) ---

print("\n--- Об MLP (Нейросеть) ---")

# Шаг 5.1: Подготовка данных (создаем лаги на основе ВСЕХ данных, чтобы не потерять связь)
# Для обучения берем только train часть, чтобы избежать утечки данных
full_lagged = create_lags(df, lags=12)

# Нам нужно разделить lagged данные так, чтобы они совпадали с нашим train/test сплитом.
# Поскольку create_lags отбрасывает первые 12 строк, индексы смещаются.
# Простой способ: берем данные для обучения, предсказываем последние 12 точек.

# Формируем X (признаки) и y (цель)
X = full_lagged.drop(columns=['y', 'ds']).values
y = full_lagged['y'].values
dates_lagged = full_lagged['ds']

# Разделение: последние 12 значений - это тест, остальное - трейн
split_idx = len(X) - 12
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
dates_test = dates_lagged[split_idx:]

# Шаг 5.2: Масштабирование (очень важно для NN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Шаг 5.3: Обучение модели
mlp = MLPRegressor(
    hidden_layer_sizes=(64, 32), 
    activation='relu', 
    max_iter=500, 
    random_state=42
)
mlp.fit(X_train_scaled, y_train)

# Шаг 5.4: Предсказание
mlp_pred = mlp.predict(X_test_scaled)

mlp_smape = calculate_smape(y_test, mlp_pred)
print(f"MLP SMAPE: {mlp_smape:.2f}%")

# --- 6. ВИЗУАЛИЗАЦИЯ ВСЕХ ТРЕХ МОДЕЛЕЙ ---

plt.figure(figsize=(14, 7))

# История
plt.plot(df['ds'], df['y'], label='Фактические продажи', color='black', alpha=0.6)

# ARIMA
plt.plot(test_df['ds'], arima_pred, label=f'ARIMA (SMAPE: {arima_smape:.1f}%)', linestyle='--', marker='o')

# Prophet
# Отрисовываем только прогнозную часть
plt.plot(test_df['ds'], prophet_pred, label=f'Prophet (SMAPE: {prophet_smape:.1f}%)', linewidth=2)

# MLP (Нейросеть)
plt.plot(dates_test, mlp_pred, label=f'MLP Neural Net (SMAPE: {mlp_smape:.1f}%)', linestyle='-.', color='green', marker='x')

plt.title('Сравнение моделей: ARIMA vs Prophet vs Нейросеть (MLP)')
plt.xlabel('Дата')
plt.ylabel('Объем продаж')
plt.legend()
plt.grid(True)
plt.show()
