# scrape_rbc_articles.py

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService # Or ChromeService
# from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import pandas as pd
import time
from settings import BROWSER_CHOICE # Import browser choice
from datetime import datetime # For date parsing

class Scraper():

    def scrape_links(self, url):
        """
        Uses Selenium to scroll through RBC search results and extract article links.
        """
        scroll_pause_time = 1.5 # Slightly increased pause time
        links_seen = set() # Use a set for faster checking of duplicates
        links_list = []

        print(f"Starting Selenium ({BROWSER_CHOICE}) to scrape links from: {url}")

        # --- Initialize WebDriver ---
        try:
            if BROWSER_CHOICE.lower() == 'firefox':
                # Optional: specify geckodriver path if not in PATH
                # service = FirefoxService(executable_path='/path/to/geckodriver')
                # driver = webdriver.Firefox(service=service)
                driver = webdriver.Firefox()
            elif BROWSER_CHOICE.lower() == 'chrome':
                # Optional: specify chromedriver path if not in PATH
                # service = ChromeService(executable_path='/path/to/chromedriver')
                # driver = webdriver.Chrome(service=service)
                driver = webdriver.Chrome()
            else:
                raise ValueError("Invalid BROWSER_CHOICE in settings.py. Use 'firefox' or 'chrome'.")
        except Exception as e:
            print(f"Error initializing WebDriver: {e}")
            print("Make sure the correct WebDriver (geckodriver or chromedriver) is installed and in your PATH.")
            return pd.DataFrame({'links': []}) # Return empty DataFrame on error

        try:
            driver.get(url)
            driver.maximize_window()

            last_height = driver.execute_script("return document.documentElement.scrollHeight")
            print("Scrolling down the search results page...")

            while True:
                # Scroll down to bottom
                driver.execute_script("window.scrollTo(0,document.documentElement.scrollHeight);")
                # Wait to load page
                time.sleep(scroll_pause_time)

                # Get HTML of the scrolled page
                html_source = driver.page_source
                soup = BeautifulSoup(html_source, "lxml")

                # Find article blocks - Adjusted selector for potentially more robust matching
                articles = soup.select('div.search-item.js-search-item, div.search-item') # Try both common classes

                # Get all links from the new blocks found in this scroll iteration
                current_scroll_links = set()
                for article in articles:
                    # Find the primary link within the search item
                    link_tag = article.select_one('a.search-item__link')
                    if link_tag and link_tag.get('href'):
                        article_url = link_tag.get('href')
                        # Basic validation and cleanup
                        if article_url.startswith('http') and '#ws' not in article_url:
                             current_scroll_links.add(article_url)

                # Add newly found links to the main list
                new_links_found = current_scroll_links - links_seen
                if new_links_found:
                    links_list.extend(list(new_links_found))
                    links_seen.update(new_links_found)
                    # print(f"Found {len(new_links_found)} new links. Total unique: {len(links_seen)}")
                else:
                    print("No new links found in this scroll section.")


                # Calculate new scroll height and compare with last scroll height
                new_height = driver.execute_script("return document.documentElement.scrollHeight")
                if new_height == last_height:
                    # Check for a "load more" button as a fallback (adjust selector if needed)
                    try:
                         load_more_button = driver.find_element(By.CSS_SELECTOR, 'a.button.js-load-more-button') # Example selector
                         if load_more_button.is_displayed():
                              print("Attempting to click 'Load More' button...")
                              load_more_button.click()
                              time.sleep(scroll_pause_time * 2) # Wait longer after click
                              last_height = new_height # Reset height to continue checking
                              continue # Go to next scroll cycle
                         else:
                              print("End of results reached (no new height and no visible 'Load More' button).")
                              break
                    except NoSuchElementException:
                         print("End of results reached (no new height, no 'Load More' button found).")
                         break # No button, definitely the end
                last_height = new_height

        except Exception as e:
            print(f"An error occurred during Selenium scraping: {e}")
        finally:
            print("Closing Selenium browser.")
            driver.quit()

        # Create DataFrame
        if not links_list:
             print("Warning: No links were extracted.")
             return pd.DataFrame({'links': []})

        final = pd.DataFrame({'links' : links_list})
        print(f"Extracted {len(final)} unique article links.")
        return final


    def scrape_article(self, session, url):
        """
        Scrapes title, description, date, and category from a single article URL.
        """
        article_data = {'date': None, 'headline': '', 'description': '', 'category': ''}
        try:
            # Use a reasonable timeout
            req = session.get(url, timeout=20)
            req.raise_for_status() # Check for HTTP errors
            plain_text = req.text
            soup = BeautifulSoup(plain_text, "lxml")

            # --- Extract Title ---
            # Try common title selectors
            title_tag = soup.select_one('h1.article__header__title, span.js-slide-title, .article__title')
            if title_tag:
                article_data['headline'] = title_tag.get_text(strip=True)

            # --- Extract Description/Overview ---
            # Try specific overview class first
            overview_tag = soup.select_one('div.article__text__overview')
            if overview_tag:
                article_data['description'] = overview_tag.get_text(strip=True)
            else:
                # Fallback: Take the first paragraph from the main text
                main_text_div = soup.select_one('div.article__text')
                if main_text_div:
                     first_p = main_text_div.find('p')
                     if first_p:
                          article_data['description'] = first_p.get_text(strip=True)
                # If still no description, leave it blank

            # --- Extract Date ---
            # Try common date selectors
            date_tag = soup.select_one('span.article__header__date, time.article__header__date')
            if date_tag:
                 date_text = date_tag.get_text(strip=True)
                 # Attempt to parse the date (formats might vary)
                 try:
                     # Example format: '18 апр, 00:17' or '18 апреля 2024, 00:17'
                     # Need robust parsing or store as text
                     article_data['date'] = date_text # Store raw date text for now
                 except ValueError:
                      print(f"Could not parse date: {date_text}")
                      article_data['date'] = date_text # Store raw on failure

            # --- Extract Category/Tags ---
            # Look in breadcrumbs first (often more reliable)
            breadcrumb_tags = soup.select('.article__header__breadcrumbs a')
            if breadcrumb_tags:
                 # Often the first or second breadcrumb is the main category
                 # Taking the text of the *first* relevant breadcrumb link
                 if len(breadcrumb_tags) > 0:
                      article_data['category'] = breadcrumb_tags[0].get_text(strip=True)
                 # Alternatively, join all breadcrumbs:
                 # article_data['category'] = ' | '.join([tag.get_text(strip=True) for tag in breadcrumb_tags])
            else:
                 # Fallback: Look for specific tag elements
                 tag_items = soup.select('a.article__tags__link')
                 if tag_items:
                      # Take the first tag as the primary category, or join them
                      article_data['category'] = tag_items[0].get_text(strip=True)
                      # article_data['category'] = ' | '.join([tag.get_text(strip=True) for tag in tag_items])

            # Basic logging
            # print(f"Scraped: {url} -> Category: {article_data['category']}, Title: {article_data['headline'][:30]}...")

        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
        except Exception as e:
            print(f"Error parsing {url}: {e}")
            # Optionally log traceback for detailed debugging
            # import traceback
            # print(traceback.format_exc())

        return article_data


    def scrape_all(self, df):
        """
        Iterates through links DataFrame, scrapes each article, and returns a combined DataFrame.
        """
        if 'links' not in df.columns or df.empty:
             print("No links provided to scrape_all.")
             return pd.DataFrame(columns=['link', 'date', 'headline', 'description', 'category'])

        links = list(df['links'])
        all_data = []
        session = requests.Session() # Use a session for efficiency
        total_links = len(links)

        print(f"Starting to scrape content for {total_links} articles...")
        for i, link in enumerate(links):
            if i > 0 and i % 50 == 0: # Log progress every 50 articles
                 print(f"  Processed {i}/{total_links} articles...")
            info = self.scrape_article(session, link)
            info['link'] = link # Add the link itself to the dict
            all_data.append(info)
            time.sleep(0.1) # Small delay between requests

        print(f"Finished scraping content for {total_links} articles.")
        # Define columns explicitly for the final DataFrame
        final_df = pd.DataFrame(all_data, columns=['link', 'date', 'headline', 'description', 'category'])
        return final_df