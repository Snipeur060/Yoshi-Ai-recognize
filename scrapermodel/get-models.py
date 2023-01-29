import os
import uuid
import requests
from bs4 import BeautifulSoup

# Make a request to Google Images with the search query
query = "yoshi green drawing"

proxies = {
   'http': '',
   'https': '',
}

for i in range(1,50):
	url = f"https://www.google.com/search?q={query}&tbm=isch&start={i*10}"
	if proxies["http"] != "" and proxies["https"] != "":
		response = requests.get(url,proxies=proxies)
	else:
		response = requests.get(url)

	# Parse the HTML content
	soup = BeautifulSoup(response.content, 'html.parser')

	# Find all image tags
	img_tags = soup.find_all('img')
	print(img_tags)
	# Extract URLs of the images
	img_urls = [img['src'] for img in img_tags if 'src' in img.attrs]
	# Download the images
	path = "img/"
	if not os.path.exists(path):
		os.mkdir(path)

	for url in img_urls:
		try:
			if proxies["http"] != "" and proxies["https"] != "":
				response = requests.get(url,allow_redirects=False,proxies=proxies)
			else:
				response = requests.get(url, allow_redirects=False)
			if response.status_code == 200:
				unique_id = uuid.uuid4()
				open(f"{path}/yoshi_green_{unique_id}.jpg", "wb").write(response.content)
			else:
				print(f"Image {url} could not be downloaded")
		except:
			pass
