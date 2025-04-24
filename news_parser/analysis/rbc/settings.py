# settings.py

# !!! IMPORTANT !!!
# GO TO: https://www.rbc.ru/search/
# 1. Enter any relevant keywords (or leave blank for all topics).
# 2. Select the date range for the LAST 5 YEARS.
# 3. Copy the resulting URL from your browser's address bar.
# 4. Paste the copied URL below, replacing the example URL.
url = 'https://www.rbc.ru/search/?query=&project=rbcnews&dateFrom=18.04.2019&dateTo=18.04.2024' # <-- PASTE YOUR 5-YEAR URL HERE

# List of target categories (case-sensitive, check exact spelling on RBC)
TARGET_RUBRICS = ["Бизнес", "Политика"]

# Choose browser: 'firefox' or 'chrome'
# Make sure you have the corresponding WebDriver installed
# Firefox: geckodriver (https://github.com/mozilla/geckodriver/releases)
# Chrome: chromedriver (https://googlechromelabs.github.io/chrome-for-testing/)
BROWSER_CHOICE = 'firefox' # Or 'chrome'