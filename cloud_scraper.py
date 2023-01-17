import cloudscraper

scraper = cloudscraper.create_scraper()
print(scraper.get('https://www.indeed.com/jobs?q=datascientist&l=remote').text)