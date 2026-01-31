# StratoSearch

## Сборка приложения

### Установка репозитория

```powershell
git clone https://github.com/ribus2005/stratosearch.git
```

```powershell
cd stratosearch
```

### Установка необходимых инструментов

Для сборки приложения необходимо установить poetry и зависимости.

#### Установка poetry

Пример команды:

```powershell
%LocalAppData%\Programs\Python\Python313\python.exe -m pip insall poetry
```

#### Создание окружения и установка зависимостей

В корневой папке репозитория:

```powershell
poetry install
```

### Сборка с pyinstaller

Есть два варианта сборки:
* Onefile - на выходе получаем .exe файл и папку с весами,  
  откуда приложение подгружает модели. 
* Onedir - на выходе получаем .exe файл и папку _internal  
  со всеми необходимыми внешними пакетами и файлами.

#### Onefile

Для сборки onefile есть python скрипт, можно использовать его,  
тогда .exe файл и папка с весами сразу добавятся в выбранную папку назначения.

```powershell
python build.py <OUTPUT_DIR>
```

Или обычный способ, но тогда папку с весами нужно будет скопировать руками.

```powershell
 poetry run pyinstaller --noconfirm --windowed --onefile --collect-all PySide6 --name stratosearch stratosearch/gui/main.py
```

#### Onedir

При таком варианте сборке веса лежат в папке _internal  
и ничего руками перетаскивать не нужно.

```powershell
 poetry run pyinstaller --noconfirm --windowed --onedir --collect-all PySide6 --name stratosearch --add-data "weights/*;weights" stratosearch/gui/main.py
```

После сборки приложение будет доступно в папке dist/.  
А результаты сборки будут в build/.
