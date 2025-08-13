
# Climate Research Demo (CO₂ vs Global Temperature)

Этот репозиторий демонстрирует научный пайплайн:
данные → анализ → графики → PDF-отчёт → автоматическое обновление по расписанию.

## Структура
```
data/                # Датасеты (CSV)
reports/             # Графики и PDF-отчёты
scripts/             # Скрипты анализа
.github/workflows/   # GitHub Actions для автогенерации отчёта
requirements.txt     # Зависимости
README.md            # Это описание
```
## Быстрый старт (локально)
1. Установите Python 3.10+
2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```
3. Сгенерируйте отчёт:
   ```bash
   python scripts/generate_report.py
   ```
   Результат: `reports/research_report.pdf`

### Опционально: реальные данные
Скрипт умеет загружать реальные глобальные данные из NOAA (CO₂) и NASA GISTEMP,
а также строить региональные ряды с сайта Berkeley Earth. Для их использования
установите переменную окружения `USE_REAL_DATA=1`:
```bash
USE_REAL_DATA=1 python scripts/generate_report.py
```
Если загрузка не удалась, будут использованы локально сохранённые CSV из каталога `data/`.
В отчёт добавляется раздел *«Региональные изменения климата»* с таблицей трендов по континентам,
графиками и тепловой картой мира.

## Автоотчёт (GitHub Actions)
Файл `.github/workflows/report.yml` запускает генерацию по расписанию.
Он:
- устанавливает Python и зависимости,
- запускает `scripts/generate_report.py`,
- коммитит новые артефакты в репозиторий.

## Лицензия
MIT
